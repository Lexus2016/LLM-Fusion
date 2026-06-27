import { z } from "zod";
import type {
  ChatCompletionRequest,
  ChatCompletionResult,
  ChatMessage,
  Strategy,
  StrategyContext,
} from "../types";
import type { FusionModelConfig, SingleModelConfig, SmartModelConfig } from "../config";
import type { Resilience } from "../concurrency";
import { createResilience } from "../concurrency";
import { FusionError } from "../errors";
import {
  failureKindForError,
  failureKindForStatus,
  isAvailabilityFailureStatus,
  logUpstreamFailure,
} from "../attribution";
import { singleStrategy } from "./single";
import { fusionStrategy } from "./fusion";
import { assertSingleVisionCapable } from "../vision";

/**
 * `smart` strategy — a classifier/router in front of two sub-routes (spec §5.8,
 * §7.4, §8.4):
 *
 *   router (one non-streamed JSON call)  ->  simple | fusion
 *
 * 1. ONE non-streamed call to the `router` model (temperature 0,
 *    `response_format: { type: "json_object" }`) classifies the incoming request
 *    as `{ "route": "simple" | "fusion", "reason": "..." }`. Only `route` is
 *    acted upon; `reason` is logged.
 * 2. `route === "simple"` -> the existing `single` executor on the resolved
 *    simple sub-config (tools pass through natively, normal stream/non-stream).
 *    `route === "fusion"` -> the existing `fusion` executor on the resolved
 *    fusion sub-config (full panel -> judge -> synth deliberate flow).
 * 3. Best-effort: any router error, timeout, non-OK status, or unparseable /
 *    invalid JSON (or a `route` outside `{simple, fusion}`) degrades to the
 *    configured `default` route (default `simple`). A router failure can lose
 *    its decision but never fails the request.
 * 4. The router call is NEVER streamed; the chosen sub-route streams as usual
 *    (one classifier round-trip of latency before the first token).
 *
 * The sub-strategy executors are reused verbatim — `smart` only resolves the
 * sub-config and dispatches.
 */

const ROUTER_SYSTEM_PROMPT = [
  "You are a routing classifier for an LLM proxy.",
  "Decide whether the user's request needs multi-model deliberation (\"fusion\") or whether a single capable model suffices (\"simple\").",
  'Reply with ONLY a JSON object of the form {"route": "simple" | "fusion", "reason": "<short reason>"}.',
  "",
  'Choose "fusion" when the request is any of:',
  "- complex, multi-faceted, or multi-step (several sub-problems, trade-offs, or moving parts);",
  "- ambiguous or underspecified (the best answer depends on assumptions worth cross-checking);",
  "- high-stakes or hard to reverse (correctness, safety, money, data loss, or production impact);",
  "- architecture, system design, or API/schema design;",
  "- security, threat modeling, or anything where a subtle mistake is costly;",
  "- debugging, root-cause analysis, or reasoning about why something fails;",
  "- research, comparison, or synthesis across multiple sources or options;",
  "- anything that benefits from several models cross-checking each other to catch blind spots.",
  "",
  'Choose "simple" only when the request is genuinely routine: a single-step task, a factual lookup, a short edit, a trivial transformation, boilerplate, or casual conversation that one strong model answers well on its own.',
  "",
  "When in doubt between the two, prefer \"fusion\" — the cost of under-deliberating a hard task outweighs the cost of deliberating an easy one.",
  "Output the JSON object and nothing else.",
].join("\n");

/** The router's decision. Only `route` is acted upon; `reason` is observability. */
const RouteDecisionSchema = z
  .object({
    route: z.enum(["simple", "fusion"]),
    reason: z.string().optional(),
  })
  .passthrough();
type RouteDecision = z.infer<typeof RouteDecisionSchema>;

/** OpenAI- or native-shaped completion envelope (only the content we read). */
const RouterMessageSchema = z
  .object({
    content: z.union([z.string(), z.null()]).optional(),
    // "thinking" router models may put the JSON decision here with empty content.
    reasoning: z.union([z.string(), z.null()]).optional(),
    reasoning_content: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();

const RouterCompletionSchema = z
  .object({
    choices: z.array(z.object({ message: RouterMessageSchema }).passthrough()).optional(),
    message: RouterMessageSchema.optional(),
  })
  .passthrough();

export const smartStrategy: Strategy = {
  async execute(ctx: StrategyContext): Promise<Response> {
    if (ctx.modelConfig.strategy !== "smart") {
      throw new FusionError("smart strategy invoked with a non-smart model config", 500, "internal_error");
    }
    const cfg = ctx.modelConfig;

    // Agent-loop escalation (see SmartModelSchema.escalate_on_tool_error): when
    // the latest tool result looks like a failure the model is recovering from an
    // error — exactly the step that benefits from deliberation — so route straight
    // to fusion, skipping the router round-trip. OpenRouter Fusion structurally
    // cannot do this: it is invoked out-of-loop and never sees the tool result.
    if (cfg.escalate_on_tool_error && latestToolResultIsError(ctx.request)) {
      ctx.logger.info(
        { model: ctx.request.model, route: "fusion", reason: "tool_error_escalation" },
        "smart: latest tool result looks like a failure; escalating to fusion",
      );
      // Force full deliberation: planning-turn-only would otherwise degrade this
      // mid-loop step (a tool message is present) back to synth-only, defeating
      // the escalation. The escalation IS the decision to deliberate mid-loop.
      const modelConfig = { ...resolveFusion(ctx, cfg), fusion_planning_turn_only: false };
      return fusionStrategy.execute({ ...ctx, modelConfig });
    }

    const route = await classify(ctx, cfg);

    if (route === "simple") {
      const modelConfig = resolveSimple(ctx, cfg);
      // Vision gate on the resolved single target: an image request smart-routed
      // to a non-vision target fails clean (400) instead of an opaque upstream
      // error. Mirrors the `single` dispatch gate in router.ts.
      await assertSingleVisionCapable(ctx.capabilities, ctx.request, modelConfig.target, ctx.request.model);
      return singleStrategy.execute({ ...ctx, modelConfig });
    }
    const modelConfig = resolveFusion(ctx, cfg);
    return fusionStrategy.execute({ ...ctx, modelConfig });
  },
};

/**
 * Run the best-effort classifier. Returns the configured `default` route on any
 * failure (error, timeout, non-OK status, unparseable/invalid JSON) — never
 * throws, so a router failure can never fail the request.
 */
async function classify(ctx: StrategyContext, cfg: SmartModelConfig): Promise<"simple" | "fusion"> {
  const router = cfg.router;
  const fallback = cfg.default;
  const resilience: Resilience =
    ctx.resilience ?? createResilience({ maxConcurrency: ctx.config.upstream.max_concurrency });

  if (!resilience.breaker.canAttempt(router)) {
    logUpstreamFailure(ctx.logger, { stage: "router", model: router, kind: "circuit_open", latencyMs: 0 });
    ctx.logger.warn(
      { router, model: ctx.request.model, route: fallback },
      "smart: router circuit open; using default route",
    );
    return fallback;
  }

  const body: Record<string, unknown> = {
    model: router,
    temperature: 0,
    response_format: { type: "json_object" },
    stream: false,
    messages: [
      { role: "system", content: ROUTER_SYSTEM_PROMPT },
      { role: "user", content: renderRequestForRouter(ctx.request) },
    ],
  };

  const startedAt = Date.now();
  let result: ChatCompletionResult;
  try {
    result = await resilience.limiter(() => ctx.client.chatCompletions(body, { stream: false }));
  } catch (err) {
    resilience.breaker.recordFailure(router);
    ctx.usage?.recordError(router);
    logUpstreamFailure(ctx.logger, {
      stage: "router",
      model: router,
      kind: failureKindForError(err),
      latencyMs: Date.now() - startedAt,
      reason: err instanceof Error ? err.message : String(err),
    });
    ctx.logger.warn(
      {
        router,
        model: ctx.request.model,
        route: fallback,
        reason: err instanceof Error ? err.message : String(err),
      },
      "smart: router call failed; using default route",
    );
    return fallback;
  }
  ctx.usage?.record(router, result);

  if (result.kind !== "json" || result.status >= 400) {
    // Only availability failures (non-json / 429 / 5xx) count against the router
    // model's health; a 4xx still degrades the route but leaves the breaker.
    if (result.kind !== "json" || isAvailabilityFailureStatus(result.status)) {
      resilience.breaker.recordFailure(router);
      logUpstreamFailure(ctx.logger, {
        stage: "router",
        model: router,
        kind: result.kind !== "json" ? "error" : failureKindForStatus(result.status),
        ...(result.kind === "json" ? { status: result.status } : {}),
        latencyMs: Date.now() - startedAt,
      });
    }
    ctx.logger.warn(
      { router, model: ctx.request.model, route: fallback, status: result.kind === "json" ? result.status : undefined },
      "smart: router returned a non-OK response; using default route",
    );
    return fallback;
  }

  resilience.breaker.recordSuccess(router);
  const decision = parseRouteDecision(extractContent(result.data));
  if (decision === null) {
    ctx.logger.warn(
      { router, model: ctx.request.model, route: fallback },
      "smart: router returned unparseable/invalid JSON; using default route",
    );
    return fallback;
  }

  ctx.logger.info(
    { router, model: ctx.request.model, route: decision.route, reason: decision.reason },
    "smart: router decision",
  );
  return decision.route;
}

/** Resolve the `simple` slot to a concrete single-model config. */
function resolveSimple(ctx: StrategyContext, cfg: SmartModelConfig): SingleModelConfig {
  const ref = cfg.simple;
  if (typeof ref !== "string") {
    return { strategy: "single", target: ref.target };
  }
  const target = ctx.config.models[ref];
  if (!target || target.strategy !== "single") {
    throw new FusionError(
      `smart model '${ctx.request.model}' simple ref '${ref}' does not resolve to a 'single' model`,
      500,
      "internal_error",
    );
  }
  return target;
}

/** Resolve the `fusion` slot to a concrete fusion-model config (inline blocks get strategy defaults). */
function resolveFusion(ctx: StrategyContext, cfg: SmartModelConfig): FusionModelConfig {
  const ref = cfg.fusion;
  if (typeof ref !== "string") {
    return {
      strategy: "fusion",
      panel: ref.panel,
      judge: ref.judge,
      synth: ref.synth,
      tool_mode: "deliberate",
      fusion_planning_turn_only: false,
      promote_reasoning_to_content: ref.promote_reasoning_to_content,
    };
  }
  const target = ctx.config.models[ref];
  if (!target || target.strategy !== "fusion") {
    throw new FusionError(
      `smart model '${ctx.request.model}' fusion ref '${ref}' does not resolve to a 'fusion' model`,
      500,
      "internal_error",
    );
  }
  return target;
}

/** Parse the router content as a route decision; null on any failure. */
function parseRouteDecision(content: string | null): RouteDecision | null {
  if (content === null) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(content);
  } catch {
    return null;
  }
  const parsed = RouteDecisionSchema.safeParse(raw);
  return parsed.success ? parsed.data : null;
}

/** First candidate with non-whitespace text, else null. */
function firstNonEmptyText(...values: Array<string | null | undefined>): string | null {
  for (const v of values) {
    if (typeof v === "string" && v.trim().length > 0) return v;
  }
  return null;
}

/**
 * Extract the router's decision text from an OpenAI- or native-shaped completion.
 * "Thinking" router models can return the JSON decision in `reasoning` /
 * `reasoning_content` with an empty `content` (mirrors fusion.ts `effectiveText`);
 * promote those too, otherwise a valid route is silently lost to the default.
 */
function extractContent(data: unknown): string | null {
  const parsed = RouterCompletionSchema.safeParse(data);
  if (!parsed.success) return null;
  const choiceMsg = parsed.data.choices?.[0]?.message;
  const fromChoices = choiceMsg
    ? firstNonEmptyText(choiceMsg.content, choiceMsg.reasoning, choiceMsg.reasoning_content)
    : null;
  if (fromChoices !== null) return fromChoices;
  const nativeMsg = parsed.data.message;
  return nativeMsg
    ? firstNonEmptyText(nativeMsg.content, nativeMsg.reasoning, nativeMsg.reasoning_content)
    : null;
}

/** Render the incoming conversation into a compact transcript for the classifier. */
function renderRequestForRouter(request: ChatCompletionRequest): string {
  const messages: ChatMessage[] = Array.isArray(request.messages) ? request.messages : [];
  const lines: string[] = [];
  for (const m of messages) {
    const role = typeof m.role === "string" ? m.role : "user";
    const content = m.content;
    let text: string;
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      text = "[multimodal content]";
    } else {
      text = "";
    }
    lines.push(`${role}: ${text}`);
  }
  return `Classify the route for the following request:\n\n${lines.join("\n")}`;
}

// --- Agent-loop escalation -------------------------------------------------

/**
 * High-precision signals that a tool result represents a failure. Matched
 * against only the LATEST `role:"tool"` message (the output the model is about
 * to reason over). The error/exception/traceback/fatal/panic forms are
 * LINE-ANCHORED (`^\s*` + multiline): a real failure dump opens a line with the
 * signal, whereas a benign file read that merely mentions an exception mid-line
 * (e.g. `# raises ValueError: ...`) must NOT escalate a 5x-cost deliberation.
 */
const TOOL_ERROR_PATTERNS: RegExp[] = [
  /^\s*traceback \(most recent call last\)/im,
  /^\s*[\w.]*(error|exception):/im, // line opens with TypeError:, Exception:, django.db.IntegrityError:
  /^\s*(fatal|panic):/im, // git "fatal:", go "panic:"
  /\bexit (code|status) [1-9]/i,
  /\b(command|module) not found\b/i,
  /\bno such file or directory\b/i,
  /\bpermission denied\b/i,
  /\bnpm err!/i,
  /\bsegmentation fault\b/i,
  /\b\d+ (tests? )?fail(ed|ures?)\b/i, // "3 failed", "1 failure", "2 tests failed"
  /\bFAILED\b/, // test-runner summary token (case-sensitive)
];

/** Text of a (possibly multimodal) message content; "" when there is none. */
const TextPartSchema = z.object({ text: z.string() }).passthrough();
function messageContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        const parsed = TextPartSchema.safeParse(part);
        return parsed.success ? parsed.data.text : "";
      })
      .join(" ");
  }
  return "";
}

/**
 * Does the most recent tool result in the conversation look like a failure?
 * A false positive only costs one extra deliberated step; a false negative
 * leaves error recovery to a single model, so the patterns lean toward recall.
 */
function latestToolResultIsError(request: ChatCompletionRequest): boolean {
  const messages: ChatMessage[] = Array.isArray(request.messages) ? request.messages : [];
  let latest: string | null = null;
  for (const m of messages) {
    if (m.role !== "tool") continue;
    latest = messageContentText(m.content);
  }
  if (latest === null || latest.trim().length === 0) return false;
  return TOOL_ERROR_PATTERNS.some((re) => re.test(latest));
}
