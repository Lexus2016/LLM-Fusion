import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import type { Logger } from "pino";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import type { Config } from "./config";
import type { CapabilityProvider, RequestUsage, UpstreamClient } from "./types";
import type { Resilience } from "./concurrency";
import { resilienceForUpstream } from "./concurrency";
import type { ProviderRouter } from "./connectors/provider_router";
import { createPanelApp } from "./panel/routes";
import { ChatCompletionRequestSchema } from "./types";
import {
  makeUsageInjectionTransform,
  toOpenAiUsage,
  UsageAccumulator,
  usageHeaderValue,
  type PricingMap,
} from "./usage";
import { createAuthMiddleware } from "./auth";
import { createAnthropicApp } from "./anthropic";
import { dispatch, entryMembers, representativeMember } from "./router";
import { BadRequestError, FusionError, toErrorResponse } from "./errors";
import { stripHopByHopHeaders } from "./headers";

/**
 * Hono application factory. Tests build the app with a mock client + in-memory
 * config (no network, no key) via this same function.
 *
 * Routes:
 *  - GET  /health             -> {status:"ok"}
 *  - GET  /ready              -> cheap upstream reachability (ok | degraded/503)
 *  - GET  /v1/models          -> configured virtual models (OpenAI list shape)
 *  - POST /v1/chat/completions -> auth -> resolve -> strategy -> JSON | SSE
 */
export interface AppDeps {
  getConfig: () => Config;
  client: UpstreamClient;
  capabilities: CapabilityProvider;
  getAuthToken: () => string | undefined;
  /**
   * Token gating the admin surface (/admin/* + panel), when it differs from the
   * client token. Optional: when absent the admin surface uses `getAuthToken`
   * (the single-token, backward-compatible behavior).
   */
  getAdminToken?: () => string | undefined;
  logger: Logger;
  /**
   * Shared resilience bundle (limiter + circuit breaker + retry policy). Built
   * once from `upstream.max_concurrency` when omitted; tests may inject a
   * deterministic one (no-op sleeper, controllable clock).
   */
  resilience?: Resilience;
  /**
   * Provider-group router backing the panel (`/panel`, `/admin/providers`) and
   * per-request routing (a model's `provider` selects its group's pool). Optional
   * so bare unit tests can build the app without one; the server always supplies it.
   */
  router?: ProviderRouter;
  /** Path to the config file, for the panel's config editor. */
  configPath?: string;
  /** Whether an api-key env var resolves (for the config editor's UI hints). */
  envHas?: (name: string) => boolean;
  /** Trigger a process restart for boot-only settings (see ConfigEditorDeps.requestRestart). */
  requestRestart?: () => void;
}

interface ModelListItem {
  id: string;
  object: "model";
  context_window?: number;
  supports_vision?: boolean;
}

/**
 * Readiness must answer fast: it pings ONE upstream model via /api/show. Bound it
 * well under the full upstream request timeout so a slow/unreachable upstream
 * returns `503 degraded` quickly instead of hanging a load-balancer probe.
 */
const READY_TIMEOUT_MS = 5_000;

export function createApp(deps: AppDeps): Hono {
  const app = new Hono();
  const auth = createAuthMiddleware(deps.getAuthToken);
  // Admin surface auth: its own token when wired, else the client token (single-
  // token, backward compatible). `getAdminTok` also drives the admin guard's
  // Host-pinning (loopback-only unless an admin-level token is enforced).
  const getAdminTok = deps.getAdminToken ?? deps.getAuthToken;
  const adminAuth = createAuthMiddleware(getAdminTok);
  // Process-lifetime resilience: the limiter is sized from the boot config's
  // max_concurrency (an upstream change needs a restart, like base_url/key).
  const resilience = deps.resilience ?? resilienceForUpstream(deps.getConfig().upstream);

  app.get("/health", (c) => c.json({ status: "ok" }));

  // Inbound body cap on the JSON API. Generous (vision requests carry base64
  // images), but an unbounded body is a memory-exhaustion DoS on any
  // non-loopback deployment. Over the cap → the OpenAI-shaped 413.
  app.use(
    "/v1/*",
    bodyLimit({
      maxSize: 50 * 1024 * 1024,
      onError: () => toErrorResponse(new FusionError("request body too large", 413, "invalid_request_error")),
    }),
  );

  // Local connector panel + admin API (mounted only when a router is wired).
  if (deps.router) {
    app.route(
      "/",
      createPanelApp({
        router: deps.router,
        auth: adminAuth,
        logger: deps.logger,
        getConfig: deps.getConfig,
        configPath: deps.configPath,
        envHas: deps.envHas,
        authEnforced: () => Boolean(getAdminTok()),
        requestRestart: deps.requestRestart,
      }),
    );
  }

  app.get("/ready", async (c) => {
    // Readiness reflects the provider pool: ready iff at least one account in any
    // provider group is currently up. Holds for any provider mix (an
    // openai-compat-only group has no native /api/show to probe).
    if (deps.router) {
      return deps.router.anyUp()
        ? c.json({ status: "ok" })
        : c.json({ status: "degraded", reason: "no account is currently up" }, 503);
    }
    const config = deps.getConfig();
    const member = firstMember(config);
    if (!member) {
      return c.json({ status: "degraded", reason: "no models configured" }, 503);
    }
    try {
      await deps.client.show(member, { signal: AbortSignal.timeout(READY_TIMEOUT_MS) });
      return c.json({ status: "ok" });
    } catch (err) {
      deps.logger.warn({ err: errMessage(err) }, "readiness check failed (upstream unreachable)");
      return c.json({ status: "degraded", reason: "upstream unreachable" }, 503);
    }
  });

  app.get("/v1/models", auth, async (c) => {
    const config = deps.getConfig();
    // Discover capabilities in PARALLEL, bounded by the shared upstream limiter
    // (respects max_concurrency, competes fairly with in-flight chat calls).
    // Promise.all preserves the configured model order.
    const entries = Object.entries(config.models);
    const data: ModelListItem[] = await Promise.all(
      entries.map(async ([name, entry]): Promise<ModelListItem> => {
        const item: ModelListItem = { id: name, object: "model" };
        const members = [...new Set(entryMembers(config.models, entry))];
        if (members.length === 0) return item;
        const discovered = await Promise.all(
          members.map(async (member) => ({
            member,
            ...(await resilience.limiterFor(member)(() => deps.capabilities.discover(member))),
          })),
        );

        // supports_vision: surfaced from the representative member (unchanged) —
        // never guessed (spec §9.1).
        const repr = representativeMember(entry);
        const reprResult = discovered.find((d) => d.member === repr);
        if (reprResult && reprResult.source !== "default") {
          item.supports_vision = reprResult.capability.vision;
        }

        // context_window: the MIN across EVERY member the virtual model can route
        // to. A fusion request fans the prompt out to every panel member (plus
        // judge + synth), so the usable window is the SMALLEST member's, never the
        // largest — advertising more would let a prompt overflow a panel model.
        // Surfaced only when EVERY member's context is known: one unknown member
        // could be smaller, so we omit rather than over-advertise (spec §9.1).
        const contexts: number[] = [];
        let allKnown = true;
        for (const d of discovered) {
          if (d.source === "default" || d.capability.context === null) {
            allKnown = false;
            break;
          }
          contexts.push(d.capability.context);
        }
        if (allKnown && contexts.length > 0) {
          item.context_window = Math.min(...contexts);
        }
        return item;
      }),
    );
    return c.json({ object: "list", data });
  });

  // Anthropic Messages API compatibility on the same base URL.
  app.route("/", createAnthropicApp(deps));

  app.post("/v1/chat/completions", auth, async (c) => {
    // Per-request correlation id + latency. Prompt CONTENT is never logged
    // (spec §12); only the virtual model name, status, and timing. The id is
    // attached to a child logger so strategy logs (e.g. the fusion panel trace)
    // correlate with the request line.
    const reqId = randomUUID();
    const startedAt = Date.now();
    const reqLogger = deps.logger.child({ req_id: reqId });

    let raw: unknown;
    try {
      raw = await c.req.json();
    } catch {
      return toErrorResponse(new BadRequestError("request body must be valid JSON"));
    }

    const parsed = ChatCompletionRequestSchema.safeParse(raw);
    if (!parsed.success) {
      const first = parsed.error.issues[0];
      const detail = first ? `${first.path.join(".") || "<root>"}: ${first.message}` : "schema validation failed";
      return toErrorResponse(new BadRequestError(`invalid chat completion request (${detail})`));
    }

    const model = parsed.data.model;
    const stream = parsed.data.stream === true;
    const config = deps.getConfig();
    const usage = new UsageAccumulator();
    try {
      const res = await dispatch({
        request: parsed.data,
        config,
        client: deps.client,
        router: deps.router,
        capabilities: deps.capabilities,
        logger: reqLogger,
        resilience,
        usage,
        // Client abort signal: a disconnect cancels in-flight upstream calls.
        signal: c.req.raw.signal,
      });
      const strategy = config.models[model]?.strategy ?? "unknown";
      const decorated = await decorateUsage(res, usage, {
        reqId,
        model,
        strategy,
        pricing: config.pricing,
        logger: reqLogger,
        clientSignal: c.req.raw.signal,
      });
      reqLogger.info(
        { model, status: decorated.status, ms: Date.now() - startedAt, stream },
        "request complete",
      );
      return decorated;
    } catch (err) {
      const status = err instanceof FusionError ? err.httpStatus : 500;
      const ms = Date.now() - startedAt;
      if (status >= 500) {
        reqLogger.error({ model, err: errMessage(err), status, ms }, "request failed");
      } else {
        reqLogger.info({ model, status, ms }, "request rejected");
      }
      return toErrorResponse(err);
    }
  });

  return app;
}

interface UsageMeta {
  reqId: string;
  model: string;
  strategy: string;
  pricing: PricingMap | undefined;
  logger: Logger;
  /** Client request signal — distinguishes a client disconnect from an upstream failure. */
  clientSignal?: AbortSignal;
}

/**
 * Attach the aggregated request usage to the client response (spec §3 / §12):
 *  - non-stream JSON success -> set the body's `usage` field to the aggregate;
 *  - stream -> emit a final `usage` chunk before `[DONE]` (composed after the
 *    fusion reasoning->content transform);
 *  - always -> set the `x-fusion-usage` header and log one `request usage` line.
 *
 * For streams the header carries the totals known at send time (the streamed
 * call's tokens land in the trailing chunk); the log fires once the stream
 * drains. Error/non-JSON bodies are passed through untouched.
 */
async function decorateUsage(res: Response, usage: UsageAccumulator, meta: UsageMeta): Promise<Response> {
  const contentType = res.headers.get("content-type") ?? "";

  if (contentType.includes("text/event-stream")) {
    const headers = new Headers(res.headers);
    // The body below is a transform stream (usage injected in flush), not the
    // upstream body — drop upstream length/encoding headers or clients truncate.
    stripHopByHopHeaders(headers);
    headers.set("x-fusion-usage", usageHeaderValue(usage.snapshot(meta.pricing)));
    const transform = makeUsageInjectionTransform(usage, {
      reqId: meta.reqId,
      model: meta.model,
      created: Math.floor(Date.now() / 1000),
    }, meta.pricing);
    // The usage log must fire exactly once, on a clean close OR a mid-stream
    // upstream error. finalize() is now hang-safe even on error (tapStreamUsage
    // settles its usage promise via cancel()), so both pipe outcomes use it;
    // snapshot() is a defensive fallback. The pre-fix flush-only path dropped the
    // log entirely when the upstream stream errored.
    let logged = false;
    const finishLog = (): void => {
      if (logged) return;
      logged = true;
      void usage
        .finalize(meta.pricing)
        .then((u) => logUsage(meta, u))
        .catch(() => logUsage(meta, usage.snapshot(meta.pricing)));
    };
    if (res.body) {
      const reader = res.body.getReader();
      const writer = transform.writable.getWriter();
      const pump = async () => {
        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            await writer.write(value);
          }
          await writer.close();
        } catch (err) {
          void reader.cancel().catch(() => {});
          if (meta.clientSignal?.aborted) {
            // The CLIENT went away — the upstream was fine and there is nobody
            // left to propagate anything to. Not an upstream failure.
            meta.logger.debug("client disconnected mid-stream");
          } else {
            // A mid-stream upstream failure must LOOK like a failure: aborting
            // the transform errors the readable, so the client's fetch body
            // rejects. A graceful close here would let flush() append a
            // synthetic usage chunk + [DONE], presenting truncation as a
            // successful end-of-turn — and silently nullifying the failover
            // strategy's deliberate committed-stream error (spec §10.5).
            meta.logger.warn(
              { err: err instanceof Error ? err.message : String(err) },
              "upstream stream failed mid-way; propagating the error to the client",
            );
          }
          try {
            await writer.abort(err);
          } catch {
            // ignore — the stream may already be errored/cancelled
          }
        }
      };
      void pump().then(finishLog, finishLog);
      return new Response(transform.readable, { status: res.status, headers });
    }
    finishLog();
    return new Response(null, { status: res.status, headers });
  }

  const aggregate = await usage.finalize(meta.pricing);
  const headers = new Headers(res.headers);
  headers.set("x-fusion-usage", usageHeaderValue(aggregate));
  logUsage(meta, aggregate);

  // Only inject `usage` into a successful JSON object body. Strip upstream
  // length/encoding headers only when we actually rewrite the body — for
  // pass-through/error bodies they still describe the untouched upstream body.
  if (contentType.includes("application/json") && res.status < 400) {
    stripHopByHopHeaders(headers);
    const text = await res.text();
    return new Response(injectUsageIntoJson(text, aggregate), { status: res.status, headers });
  }
  return new Response(res.body, { status: res.status, headers });
}

/** Set the top-level `usage` field on a JSON object body; pass through otherwise. */
function injectUsageIntoJson(text: string, aggregate: RequestUsage): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return text;
  }
  if (Array.isArray(parsed)) return text;
  const obj = z.record(z.string(), z.unknown()).safeParse(parsed);
  if (!obj.success) return text;
  return JSON.stringify({
    ...obj.data,
    usage: toOpenAiUsage({
      promptTokens: aggregate.promptTokens,
      completionTokens: aggregate.completionTokens,
      totalTokens: aggregate.totalTokens,
    }),
  });
}

/** One structured info line per request — counts only, never prompt content. */
function logUsage(meta: UsageMeta, usage: RequestUsage): void {
  meta.logger.info(
    {
      req_id: meta.reqId,
      model: meta.model,
      strategy: meta.strategy,
      upstream_calls: usage.upstreamCalls,
      prompt_tokens: usage.promptTokens,
      completion_tokens: usage.completionTokens,
      total_tokens: usage.totalTokens,
      cost_usd: usage.costUsd,
    },
    "request usage",
  );
}

function firstMember(config: Config): string | undefined {
  for (const entry of Object.values(config.models)) {
    const member = representativeMember(entry);
    if (member) return member;
  }
  return undefined;
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
