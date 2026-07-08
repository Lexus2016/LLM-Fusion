import type { ChatCompletionResult, Strategy } from "../types";
import { CircuitOpenError, FusionError } from "../errors";
import {
  failureKindForError,
  failureKindForStatus,
  isAvailabilityFailureStatus,
  logUpstreamFailure,
} from "../attribution";
import { promoteReasoningNonStream, makeReasoningPromotionTransform } from "../reasoning";
import { detectIncompleteToolTurn, makeToolTurnGuardStream, retryToolTurn } from "./tool_turn_guard";

/**
 * `single` strategy — 1:1 passthrough to a single target model. Supports both
 * non-streaming (return upstream JSON) and streaming (pipe the upstream SSE
 * body straight through). This is the primitive `fusion` is built on in later
 * phases.
 *
 * When the request context carries the shared `resilience` bundle (always true
 * via the server) the upstream call is scheduled through the global concurrency
 * limiter and guarded by the per-model circuit breaker. Without it (bare unit
 * contexts) the call runs directly — identical observable behaviour.
 */

const STREAM_HEADERS_BASE: Record<string, string> = {
  "cache-control": "no-cache",
  connection: "keep-alive",
};

export const singleStrategy: Strategy = {
  async execute(ctx) {
    if (ctx.modelConfig.strategy !== "single") {
      throw new FusionError("single strategy invoked with a non-single model config", 500, "internal_error");
    }
    const target = ctx.modelConfig.target;
    const stream = ctx.request.stream === true;
    // Agentic context: the tool-turn completeness guard runs ONLY when the
    // request carried tools. Tool-less / mechanical requests stay byte-identical
    // passthrough (no extra latency, no retry surface).
    const hasTools = Array.isArray(ctx.request.tools) && ctx.request.tools.length > 0;

    // Forward the request verbatim, only rewriting the virtual model name to
    // the resolved real upstream target. Per-model request_overrides (e.g.
    // { reasoning_effort: "none" } to keep a thinking model from deliberating
    // for minutes on mechanical agent steps) are merged in, with the core
    // request keys protected so an override can never corrupt the call shape.
    const overrides: Record<string, unknown> = { ...(ctx.modelConfig.request_overrides ?? {}) };
    for (const key of ["model", "messages", "stream", "tools", "tool_choice"]) delete overrides[key];
    const body: Record<string, unknown> = { ...ctx.request, ...overrides, model: target };
    const resilience = ctx.resilience;

    if (resilience && !resilience.breaker.canAttempt(target)) {
      logUpstreamFailure(ctx.logger, { stage: "single", model: target, kind: "circuit_open", latencyMs: 0 });
      throw new CircuitOpenError(`circuit breaker open for '${target}'`);
    }

    const startedAt = Date.now();
    let result: ChatCompletionResult;
    try {
      result = resilience
        ? await resilience.limiterFor(target)(() => ctx.client.chatCompletions(body, { stream, signal: ctx.signal }))
        : await ctx.client.chatCompletions(body, { stream, signal: ctx.signal });
    } catch (err) {
      // Client disconnect is not an upstream health failure: do not trip the
      // breaker. Still release any reserved half-open probe so the model can be
      // probed again instead of sticking in half-open forever. Detect it via the
      // client signal (not the error name — a stage timeout also aborts the fetch
      // and would otherwise be misclassified as a disconnect).
      if (ctx.signal?.aborted) {
        resilience?.breaker.recordProbeAbandoned(target);
        throw err;
      }
      resilience?.breaker.recordFailure(target);
      ctx.usage?.recordError(target);
      logUpstreamFailure(ctx.logger, {
        stage: "single",
        model: target,
        kind: failureKindForError(err),
        latencyMs: Date.now() - startedAt,
        reason: err instanceof Error ? err.message : String(err),
      });
      throw err;
    }
    ctx.usage?.record(target, result);

    if (resilience) {
      if (result.status < 400) resilience.breaker.recordSuccess(target);
      // Only availability failures (429/5xx) count against the model's health; a
      // 4xx client/request error is passed through without tripping the breaker.
      else if (isAvailabilityFailureStatus(result.status)) resilience.breaker.recordFailure(target);
      // A 4xx non-availability means the model answered, so it is healthy — release
      // any half-open probe so it is not jammed open until process restart.
      else resilience.breaker.recordSuccess(target);
    }
    if (result.status >= 400 && isAvailabilityFailureStatus(result.status)) {
      logUpstreamFailure(ctx.logger, {
        stage: "single",
        model: target,
        kind: failureKindForStatus(result.status),
        status: result.status,
        latencyMs: Date.now() - startedAt,
      });
    }

    // Reasoning->content promotion (when enabled): some "thinking" target models
    // return their final answer in `reasoning` with empty `content`; content-only
    // clients (e.g. OpenCode) would otherwise render nothing. Applied only to
    // successful (2xx) responses, for both streamed and non-streamed bodies.
    const promote =
      ctx.modelConfig.promote_reasoning_to_content ?? ctx.config.defaults.promote_reasoning_to_content;

    if (result.kind === "stream") {
      const headers: Record<string, string> = {
        ...STREAM_HEADERS_BASE,
        "content-type": result.contentType ?? "text/event-stream",
      };
      let streamBody = result.body;
      // Tool-turn completeness guard (agentic requests only). Runs BEFORE
      // promotion so it sees the raw reasoning/content/tool_calls: a reasoning
      // model that narrates the next action ("let me write the file now") and
      // stops with no tool_call is caught and recovered into the tool call. A
      // healthy stream is forwarded unchanged, so first-token latency is untouched.
      // Reader-driven (not pipeThrough) so a MID-FLIGHT upstream cut — the
      // large-file "terminated" failure — is caught and recovered too.
      if (hasTools && result.status < 400 && streamBody) {
        streamBody = makeToolTurnGuardStream(ctx, resilience, target, body, streamBody);
      }
      if (promote && result.status < 400 && streamBody) {
        streamBody = streamBody.pipeThrough(makeReasoningPromotionTransform());
      }
      return new Response(streamBody, { status: result.status, headers });
    }

    // Non-stream tool-turn guard: same recovery for a narrate-and-stop turn.
    let responseData = result.data;
    if (hasTools && result.status < 400) {
      const incomplete = detectIncompleteToolTurn(responseData);
      if (incomplete !== null) {
        const recovered = await retryToolTurn(ctx, resilience, target, body, incomplete);
        if (recovered !== null) responseData = recovered;
      }
    }
    const data = promote && result.status < 400 ? promoteReasoningNonStream(responseData) : responseData;
    return new Response(JSON.stringify(data ?? null), {
      status: result.status,
      headers: { "content-type": "application/json" },
    });
  },
};
