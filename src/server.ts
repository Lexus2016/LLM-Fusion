import { Hono } from "hono";
import type { Logger } from "pino";
import { randomUUID } from "node:crypto";
import type { Config } from "./config";
import type { CapabilityProvider, UpstreamClient } from "./types";
import type { Resilience } from "./concurrency";
import { createResilience } from "./concurrency";
import { ChatCompletionRequestSchema } from "./types";
import { createAuthMiddleware } from "./auth";
import { dispatch, representativeMember } from "./router";
import { BadRequestError, FusionError, toErrorResponse } from "./errors";

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
  logger: Logger;
  /**
   * Shared resilience bundle (limiter + circuit breaker + retry policy). Built
   * once from `upstream.max_concurrency` when omitted; tests may inject a
   * deterministic one (no-op sleeper, controllable clock).
   */
  resilience?: Resilience;
}

interface ModelListItem {
  id: string;
  object: "model";
  context_window?: number;
  supports_vision?: boolean;
}

export function createApp(deps: AppDeps): Hono {
  const app = new Hono();
  const auth = createAuthMiddleware(deps.getAuthToken);
  // Process-lifetime resilience: the limiter is sized from the boot config's
  // max_concurrency (an upstream change needs a restart, like base_url/key).
  const resilience =
    deps.resilience ?? createResilience({ maxConcurrency: deps.getConfig().upstream.max_concurrency });

  app.get("/health", (c) => c.json({ status: "ok" }));

  app.get("/ready", async (c) => {
    const config = deps.getConfig();
    const member = firstMember(config);
    if (!member) {
      return c.json({ status: "degraded", reason: "no models configured" }, 503);
    }
    try {
      await deps.client.show(member);
      return c.json({ status: "ok" });
    } catch (err) {
      deps.logger.warn({ err: errMessage(err) }, "readiness check failed (upstream unreachable)");
      return c.json({ status: "degraded", reason: "upstream unreachable" }, 503);
    }
  });

  app.get("/v1/models", async (c) => {
    const config = deps.getConfig();
    const data: ModelListItem[] = [];
    for (const [name, entry] of Object.entries(config.models)) {
      const item: ModelListItem = { id: name, object: "model" };
      const member = representativeMember(entry);
      if (member) {
        const { capability, source } = await deps.capabilities.discover(member);
        // Only surface fields we actually know — never guess (spec §9.1).
        if (source !== "default") {
          item.supports_vision = capability.vision;
          if (capability.context !== null) item.context_window = capability.context;
        }
      }
      data.push(item);
    }
    return c.json({ object: "list", data });
  });

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
    try {
      const res = await dispatch({
        request: parsed.data,
        config: deps.getConfig(),
        client: deps.client,
        capabilities: deps.capabilities,
        logger: reqLogger,
        resilience,
      });
      reqLogger.info(
        { model, status: res.status, ms: Date.now() - startedAt, stream },
        "request complete",
      );
      return res;
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
