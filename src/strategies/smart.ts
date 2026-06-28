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
import { AllMembersFailedError, FusionError } from "../errors";
import {
  failureKindForError,
  failureKindForStatus,
  isAvailabilityFailureStatus,
  logUpstreamFailure,
} from "../attribution";
import { singleStrategy } from "./single";
import { fusionStrategy } from "./fusion";
import { assertSingleVisionCapable, requestHasImages } from "../vision";
import { extractJsonObject } from "../json";
import { withTimeout, realTimer } from "../timeout";
import { createHash } from "node:crypto";

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
  'Choose "fusion" only when the request is genuinely complex or high-stakes:',
  "- complex, multi-faceted, or multi-step architecture or system design decisions;",
  "- ambiguous or underspecified requirements where assumptions must be debated;",
  "- critical, high-stakes tasks where a mistake causes data loss, security bugs, or money loss;",
  "- debugging deep, unknown root-causes of failures (where it is unclear why something is broken);",
  "- research, comparison, or synthesis of conflicting trade-offs across multiple options;",
  "- security auditing, cryptography, or threat modeling.",
  "",
  'Choose "simple" for all routine coding, editing, and execution tasks:',
  "- writing unit tests, adding test cases, or creating straightforward test suites (correctness is easily checked by running them);",
  "- writing routine helper functions, boilerplate, basic scripts, or straightforward implementations;",
  "- making minor edits, refactoring local functions, or formatting code;",
  "- fixing clear-cut syntax errors, typos, or well-defined bugs with obvious solutions;",
  "- explaining code, documenting code, or writing markdown documentation;",
  "- factual lookups, casual conversation, grep/search, or mechanical operations.",
  "",
  "If this is an agent mid-loop (the latest message is a tool result), classify the model's NEXT action:",
  '- "fusion" only for substantive, complex design/architecture changes, debugging unknown errors, or high-stakes security/API modifications.',
  '- "simple" for routine actions: writing unit tests, adding tests, writing basic helpers/boilerplate, editing minor files, reading/viewing files, running commands/tests, or mechanical tool updates.',
  "",
  "When in doubt, choose \"simple\" if the output is easily verified by running tests or a compiler, as the test runner serves as the correction gate.",
  "",
  "Ground your reason in the LITERAL content of the latest message. Do NOT invent multimodal content: do NOT claim the user sent an image, screenshot, photo, or file unless an image/attachment is actually present in the message. If the latest message is plain text, treat it as plain text.",
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

/**
 * Router-decision cache. A long agent loop re-issues the classifier every turn,
 * but consecutive turns with an identical routing view (same recent window,
 * same latest tool result) yield the same decision — re-paying one router
 * round-trip of latency and one upstream call each time. The cache keys on the
 * exact request body the router is asked to classify, so a hit is provably the
 * same decision the model would return. Only successfully parsed router
 * decisions are cached; failures (timeout / non-OK / unparseable) fall through
 * to the default route and are NOT cached, so a transient blip self-heals on the
 * next identical request. In-flight coalescing (`routerPending`) collapses
 * concurrent identical requests onto a single upstream call.
 */
const MAX_ROUTER_CACHE_SIZE = 256;
const routerCache = new Map<string, "simple" | "fusion">();
const routerPending = new Map<string, Promise<"simple" | "fusion">>();

/** Stable hash of a router request body for use as a cache key. */
function routerCacheKey(body: Record<string, unknown>): string {
  return createHash("sha256").update(JSON.stringify(body)).digest("hex");
}

/** Test-only: clear the router decision cache and any in-flight promises. */
export function __resetRouterCacheForTesting(): void {
  routerCache.clear();
  routerPending.clear();
}

/** Insert a successful router decision into the LRU-bounded cache. */
function setRouterCache(key: string, route: "simple" | "fusion"): void {
  // Delete-then-set moves an existing key to the end (most-recently-used),
  // approximating LRU on Map's insertion-order iteration below.
  routerCache.delete(key);
  routerCache.set(key, route);
  if (routerCache.size > MAX_ROUTER_CACHE_SIZE) {
    const oldest = routerCache.keys().next().value;
    if (oldest !== undefined) routerCache.delete(oldest);
  }
}

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
      return executeSimple(ctx, cfg);
    }
    return executeFusionWithFallback(ctx, cfg);
  },
};

/** Run the simple sub-strategy with vision validation. */
async function executeSimple(ctx: StrategyContext, cfg: SmartModelConfig): Promise<Response> {
  const modelConfig = resolveSimple(ctx, cfg);
  // Vision gate on the resolved single target: an image request smart-routed
  // to a non-vision target fails clean (400) instead of an opaque upstream
  // error. Mirrors the `single` dispatch gate in router.ts.
  await assertSingleVisionCapable(ctx.capabilities, ctx.request, modelConfig.target, ctx.request.model);
  return singleStrategy.execute({ ...ctx, modelConfig });
}

/**
 * Run fusion with auto-fallback to simple on panel failure. When fusion fails
 * (e.g. context overflow → all panel members return 400) the smart strategy
 * degrades to simple instead of bubbling a 502 to the client. A fusion failure
 * can lose its deliberation but never fails the user's request.
 */
async function executeFusionWithFallback(ctx: StrategyContext, cfg: SmartModelConfig): Promise<Response> {
  // The router has decided this step is worth deliberation, so run the FULL panel
  // even mid-loop: a referenced fusion model's `fusion_planning_turn_only` would
  // otherwise degrade a tool-continuation step back to synth-only, silently
  // overriding the router's choice (the whole reason it routed to fusion here).
  const modelConfig = { ...resolveFusion(ctx, cfg), fusion_planning_turn_only: false };
  try {
    return await fusionStrategy.execute({ ...ctx, modelConfig });
  } catch (err) {
    // AllMembersFailedError = panel could not produce enough answers (context
    // overflow, all models down, etc.). Degrade to simple so the request still
    // gets a response — a single-model answer is better than a 502.
    if (err instanceof AllMembersFailedError) {
      ctx.logger.warn(
        { model: ctx.request.model, reason: err.message },
        "smart: fusion panel failed; falling back to simple",
      );
      return executeSimple(ctx, cfg);
    }
    throw err;
  }
}

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

  const key = routerCacheKey(body);
  const cached = routerCache.get(key);
  if (cached !== undefined) {
    ctx.logger.debug(
      { router, model: ctx.request.model, route: cached },
      "smart: router cache hit; reusing prior decision",
    );
    return cached;
  }

  // Collapse concurrent identical requests onto a single upstream call so a burst
  // of identical in-flight requests does not pay N router round-trips.
  const inFlight = routerPending.get(key);
  if (inFlight) {
    ctx.logger.debug(
      { router, model: ctx.request.model },
      "smart: router cache coalesce; awaiting in-flight classifier",
    );
    return inFlight;
  }

  const promise = classifyUncached(ctx, cfg, resilience, body, key).finally(() => {
    routerPending.delete(key);
  });
  routerPending.set(key, promise);
  return promise;
}

/**
 * The actual upstream classifier call. Isolated from `classify` so the cache
 * wrapper can short-circuit on a hit / coalesce in-flight duplicates. Returns
 * the configured `default` route on any failure (error, timeout, non-OK status,
 * unparseable/invalid JSON) — never throws. Only a successfully parsed router
 * decision is written to the cache; a failure must NOT be cached, so a
 * transient blip self-heals on the next identical request.
 */
async function classifyUncached(
  ctx: StrategyContext,
  cfg: SmartModelConfig,
  resilience: Resilience,
  body: Record<string, unknown>,
  key: string,
): Promise<"simple" | "fusion"> {
  const router = cfg.router;
  const fallback = cfg.default;

  const startedAt = Date.now();
  let result: ChatCompletionResult;
  // Stage timeout: bound the router independently of the full upstream request
  // timeout so a slow router cannot hang the whole request — on timeout it throws
  // and the catch below degrades to the default route. The abort frees the
  // limiter slot promptly; the client's abort signal (if any) is combined in so a
  // disconnect cancels the router call too.
  const timeoutMs = ctx.config.defaults.router_timeout_s * 1000;
  const stageAbort = new AbortController();
  const signal = ctx.signal ? AbortSignal.any([ctx.signal, stageAbort.signal]) : stageAbort.signal;
  try {
    result = await resilience.limiter(() =>
      withTimeout(
        ctx.client.chatCompletions(body, { stream: false, signal }),
        timeoutMs,
        realTimer,
        `smart router '${router}' timed out after ${timeoutMs}ms`,
        () => stageAbort.abort(),
      ),
    );
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

  // Hallucination guard: the router (an LLM) can confabulate multimodal input —
  // e.g. "user sent a screenshot/image" as a reason to pick `simple` — even when
  // the request is plain text with no image_url blocks. A reason that claims an
  // image which isn't there means the router did not actually ground its decision
  // in the real message, so its route is untrustworthy: fall back to the
  // configured default (consistent with how any other untrustworthy router reply
  // is handled). Do NOT cache it.
  if (claimsImage(decision.reason) && !requestHasImages(ctx.request)) {
    ctx.logger.warn(
      { router, model: ctx.request.model, route: fallback, reason: decision.reason },
      "smart: router claimed an image/screenshot that is not present; treating decision as untrustworthy, using default route",
    );
    return fallback;
  }

  ctx.logger.info(
    { router, model: ctx.request.model, route: decision.route, reason: decision.reason },
    "smart: router decision",
  );
  setRouterCache(key, decision.route);
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
      web_search: ref.web_search,
      bineval: ref.bineval,
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
/**
 * Does a free-text router reason claim the request carried an image, screenshot,
 * photo, or other visual/attached content? Used by the hallucination guard to
 * catch a router that invents multimodal input to justify its route.
 */
/**
 * Does a free-text router reason AFFIRMATIVELY claim the request carried an
 * image, screenshot, photo, or other visual/attached content? Used by the
 * hallucination guard to catch a router that invents multimodal input to
 * justify its route.
 *
 * Crucially, this returns FALSE when the router merely MENTIONS images in a
 * negating/absence context — e.g. "no actual image attachment", "plain-text,
 * not multimodal", "without a screenshot". A router that correctly observes
 * there is no image is reasoning properly, not hallucinating, and must not be
 * flagged (flagging it would override a correct decision and spam warnings on
 * every paste where the router dutifully notes the absence of an image).
 */
function claimsImage(reason: string | undefined): boolean {
  if (!reason) return false;
  const lower = reason.toLowerCase();
  // Negated/absence context: the router explicitly says there is NO image /
  // the message is plain text. Treat as "not a claim" and trust the router.
  if (
    /no (?:actual )?(?:image|screenshot|photo|picture|attachment|multimodal)/.test(lower) ||
    /without (?:an? )?(?:image|screenshot|photo|picture|attachment|multimodal)/.test(lower) ||
    /not (?:a |an )?(?:image|screenshot|photo|picture|multimodal)/.test(lower) ||
    /\bplain[- ]?text\b/.test(lower) ||
    /no (?:file |visual )?attachment/.test(lower) ||
    /\btext[- ]?only\b/.test(lower)
  ) {
    return false;
  }
  // Affirmative claim that an image/screenshot/etc. is present.
  return /\b(image|screenshot|photo|picture|figure|diagram|chart|scan|attachment|multimodal)\b/i.test(reason);
}

function parseRouteDecision(content: string | null): RouteDecision | null {
  if (content === null) return null;
  // Same fence/prose-tolerance as the judge: extract the balanced JSON object so a
  // ```json fence does not push a valid decision onto the default route.
  const jsonText = extractJsonObject(content);
  if (jsonText === null) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(jsonText);
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

/**
 * Per-message content caps for the router transcript. The router classifies the
 * NEXT action from message ROLES and recent gist — it does not need a giant file
 * dump verbatim. Head+tail keeps both the opening (what kind of content this is)
 * and the closing (where a trailing "=== 3 failed ===" / error summary lives).
 */
const ROUTER_MSG_HEAD = 1200;
const ROUTER_MSG_TAIL = 600;
/**
 * How many of the most-recent messages the router sees. The next-action signal
 * lives in the latest tool result + the assistant turn that produced it; older
 * history is represented by the first user instruction (the overall intent).
 */
const ROUTER_RECENT_WINDOW = 6;

/** A message's text, truncated head+tail with an omission marker when oversized. */
function capMessageText(text: string): string {
  const max = ROUTER_MSG_HEAD + ROUTER_MSG_TAIL;
  if (text.length <= max) return text;
  const omitted = text.length - max;
  return `${text.slice(0, ROUTER_MSG_HEAD)}\n…[${omitted} chars omitted]…\n${text.slice(-ROUTER_MSG_TAIL)}`;
}

/** One `role: <capped text>` transcript line for a message. */
function routerMessageLine(m: ChatMessage): string {
  const role = typeof m.role === "string" ? m.role : "user";
  const content = m.content;
  let text: string;
  if (typeof content === "string") {
    text = content;
  } else if (Array.isArray(content)) {
    // Extract actual text from structured content parts (OpenAI array format).
    // The old code replaced ALL array content with "[multimodal content]",
    // blinding the router to the message's actual text and making it impossible
    // to classify complexity. Now we extract text and only flag real images.
    const ImagePartSchema = z.object({ type: z.literal("image_url") }).passthrough();
    const hasImage = content.some(
      (p: unknown) => ImagePartSchema.safeParse(p).success,
    );
    text = messageContentText(content);
    if (hasImage) text = `[has image] ${text}`;
    if (text.trim().length === 0) text = "[empty content]";
  } else {
    text = "";
  }
  return `${role}: ${capMessageText(text)}`;
}

/**
 * Render a COMPACT routing view of the conversation. A naive full transcript
 * makes the router a full-context call: on a ~350k-token agent loop the router
 * (which must ingest the whole prompt just to emit one JSON line) measured ~57s
 * and intermittently hit the 120s upstream timeout, then silently fell back to
 * the default route — exactly when the agent needed a real decision. The router
 * only classifies the NEXT action, so it needs the agent's role (system), the
 * original task (first message), the ACTIVE instruction (most recent user
 * message), and the recent state (last few messages) — NOT the entire history.
 * This bounds router latency and cost to a small constant regardless of how
 * large the conversation grows; each message is content-capped so a single huge
 * tool result cannot blow up the router prompt either.
 *
 * The active-instruction anchor matters because OpenCode sessions issue NEW
 * sub-tasks mid-conversation: after more than `ROUTER_RECENT_WINDOW` mechanical
 * tool turns (grep / read / list) the substantive instruction would fall out of
 * the recent window, and the router — seeing only an obsolete first task plus
 * recent file reads — would misroute the real coding step to `simple`. That is
 * the very failure this strategy exists to prevent, so the latest user message
 * is always kept even when it predates the window.
 */
function renderRequestForRouter(request: ChatCompletionRequest): string {
  const messages: ChatMessage[] = Array.isArray(request.messages) ? request.messages : [];
  const system: ChatMessage[] = [];
  const nonSystem: ChatMessage[] = [];
  for (const m of messages) {
    if (m.role === "system") system.push(m);
    else nonSystem.push(m);
  }

  const lines: string[] = system.map(routerMessageLine);

  if (nonSystem.length <= ROUTER_RECENT_WINDOW + 1) {
    for (const m of nonSystem) lines.push(routerMessageLine(m));
    return `Classify the route for the following request:\n\n${lines.join("\n")}`;
  }

  // Build the set of indices to keep, then render them in order with an omission
  // marker spanning each gap (so the middle is summarized, never silently dropped).
  const keep = new Set<number>();
  keep.add(0); // first message — original session framing
  const recentStart = nonSystem.length - ROUTER_RECENT_WINDOW;
  for (let i = recentStart; i < nonSystem.length; i++) keep.add(i); // recent window
  for (let i = recentStart - 1; i > 0; i--) {
    if (nonSystem[i]?.role === "user") {
      keep.add(i); // most recent user instruction that predates the window
      break;
    }
  }

  let prev = -1;
  for (const i of [...keep].sort((a, b) => a - b)) {
    const gap = i - prev - 1;
    if (prev >= 0 && gap > 0) {
      lines.push(`…[${gap} earlier message${gap === 1 ? "" : "s"} omitted]…`);
    }
    const m = nonSystem[i];
    if (m) lines.push(routerMessageLine(m));
    prev = i;
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
