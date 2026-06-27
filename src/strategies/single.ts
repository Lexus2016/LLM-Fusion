import type { ChatCompletionResult, Strategy } from "../types";
import { CircuitOpenError, FusionError } from "../errors";

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
      throw new CircuitOpenError(`circuit breaker open for '${target}'`);
    }

    let result: ChatCompletionResult;
    try {
      result = resilience
        ? await resilience.limiter(() => ctx.client.chatCompletions(body, { stream }))
        : await ctx.client.chatCompletions(body, { stream });
    } catch (err) {
      resilience?.breaker.recordFailure(target);
      throw err;
    }

    if (resilience) {
      if (result.status < 400) resilience.breaker.recordSuccess(target);
      else resilience.breaker.recordFailure(target);
    }

    if (result.kind === "stream") {
      const headers: Record<string, string> = {
        ...STREAM_HEADERS_BASE,
        "content-type": result.contentType ?? "text/event-stream",
      };
      return new Response(result.body, { status: result.status, headers });
    }

    return new Response(JSON.stringify(result.data ?? null), {
      status: result.status,
      headers: { "content-type": "application/json" },
    });
  },
};
