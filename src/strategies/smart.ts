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
  "Decide whether the user's request needs multi-model deliberation or whether a single capable model suffices.",
  'Reply with ONLY a JSON object of the form {"route": "simple" | "fusion", "reason": "<short reason>"}.',
  'Choose "fusion" for hard, ambiguous, high-stakes, multi-step, or reasoning-heavy tasks that benefit from several models cross-checking each other.',
  'Choose "simple" for straightforward requests a single strong model answers well (lookups, short edits, simple Q&A, casual conversation).',
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
const RouterCompletionSchema = z
  .object({
    choices: z
      .array(
        z
          .object({
            message: z
              .object({ content: z.union([z.string(), z.null()]).optional() })
              .passthrough(),
          })
          .passthrough(),
      )
      .optional(),
    message: z.object({ content: z.union([z.string(), z.null()]).optional() }).passthrough().optional(),
  })
  .passthrough();

export const smartStrategy: Strategy = {
  async execute(ctx: StrategyContext): Promise<Response> {
    if (ctx.modelConfig.strategy !== "smart") {
      throw new FusionError("smart strategy invoked with a non-smart model config", 500, "internal_error");
    }
    const cfg = ctx.modelConfig;
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

  let result: ChatCompletionResult;
  try {
    result = await resilience.limiter(() => ctx.client.chatCompletions(body, { stream: false }));
  } catch (err) {
    resilience.breaker.recordFailure(router);
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

  if (result.kind !== "json" || result.status >= 400) {
    resilience.breaker.recordFailure(router);
    ctx.logger.warn(
      { router, model: ctx.request.model, route: fallback, status: result.status },
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

/** Extract assistant text from an OpenAI- or native-shaped completion. */
function extractContent(data: unknown): string | null {
  const parsed = RouterCompletionSchema.safeParse(data);
  if (!parsed.success) return null;
  const fromChoices = parsed.data.choices?.[0]?.message.content;
  if (typeof fromChoices === "string" && fromChoices.length > 0) return fromChoices;
  const fromNative = parsed.data.message?.content;
  if (typeof fromNative === "string" && fromNative.length > 0) return fromNative;
  return null;
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
