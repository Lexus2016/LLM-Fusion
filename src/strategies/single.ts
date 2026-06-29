import type { ChatCompletionResult, Strategy } from "../types";
import { CircuitOpenError, FusionError } from "../errors";
import {
  failureKindForError,
  failureKindForStatus,
  isAvailabilityFailureStatus,
  logUpstreamFailure,
} from "../attribution";
import { promoteReasoningNonStream, makeReasoningPromotionTransform } from "../reasoning";

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

    // Forward the request verbatim, only rewriting the virtual model name to
    // the resolved real upstream target.
    const body: Record<string, unknown> = { ...ctx.request, model: target };
    const resilience = ctx.resilience;

    if (resilience && !resilience.breaker.canAttempt(target)) {
      logUpstreamFailure(ctx.logger, { stage: "single", model: target, kind: "circuit_open", latencyMs: 0 });
      throw new CircuitOpenError(`circuit breaker open for '${target}'`);
    }

    const startedAt = Date.now();
    let result: ChatCompletionResult;
    try {
      result = resilience
        ? await resilience.limiter(() => ctx.client.chatCompletions(body, { stream, signal: ctx.signal }))
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
      const body =
        promote && result.status < 400 && result.body
          ? result.body.pipeThrough(makeReasoningPromotionTransform())
          : result.body;
      return new Response(body, { status: result.status, headers });
    }

    const data = promote && result.status < 400 ? promoteReasoningNonStream(result.data) : result.data;
    return new Response(JSON.stringify(data ?? null), {
      status: result.status,
      headers: { "content-type": "application/json" },
    });
  },
};
