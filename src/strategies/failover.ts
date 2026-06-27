import type { ChatCompletionResult, Strategy, StrategyContext } from "../types";
import type { Resilience } from "../concurrency";
import { backoffDelay, createResilience } from "../concurrency";
import { AllMembersFailedError, CircuitOpenError, FusionError, UpstreamNetworkError } from "../errors";

/**
 * `failover` strategy — try a `chain` of real upstream models in order with
 * explicit, correct error semantics (spec §5.6 / §10.4):
 *
 *  - 429  → exponential backoff + retry the SAME member (never advance).
 *  - 5xx / network / timeout → brief same-member retries, then ADVANCE.
 *  - 4xx other than 429 → surface immediately (request error; passthrough).
 *  - all members attempted & failed → 502 (AllMembersFailedError).
 *  - every member circuit-open (skipped) → 503 (CircuitOpenError).
 *
 * Streaming rule (spec §10.5, critical): for `stream:true` we may only switch
 * members BEFORE the first byte is forwarded to the client. We peek the first
 * upstream chunk; a failure before it arrives is retryable/advanceable, but once
 * that first chunk is committed to the client stream, any later upstream failure
 * surfaces as a stream error (`controller.error`) — never a silent re-route.
 */

const STREAM_HEADERS_BASE: Record<string, string> = {
  "cache-control": "no-cache",
  connection: "keep-alive",
};

type MemberOutcome =
  | { kind: "return"; response: Response }
  | { kind: "advance"; error: FusionError };

type Peek =
  | { kind: "error"; message: string }
  | { kind: "empty" }
  | { kind: "chunk"; reader: ReadableStreamDefaultReader<Uint8Array>; first: Uint8Array };

export const failoverStrategy: Strategy = {
  async execute(ctx: StrategyContext): Promise<Response> {
    if (ctx.modelConfig.strategy !== "failover") {
      throw new FusionError(
        "failover strategy invoked with a non-failover model config",
        500,
        "internal_error",
      );
    }
    const chain = ctx.modelConfig.chain;
    const stream = ctx.request.stream === true;
    const resilience =
      ctx.resilience ?? createResilience({ maxConcurrency: ctx.config.upstream.max_concurrency });

    let attempted = 0;
    let lastError: FusionError | undefined;

    for (const member of chain) {
      if (!resilience.breaker.canAttempt(member)) {
        ctx.logger.warn({ member, model: ctx.request.model }, "failover: skip member (circuit open)");
        lastError = new CircuitOpenError(`circuit breaker open for failover member '${member}'`);
        continue;
      }
      attempted += 1;
      const outcome = stream
        ? await attemptStreamMember(ctx, resilience, member)
        : await attemptJsonMember(ctx, resilience, member);
      if (outcome.kind === "return") return outcome.response;
      lastError = outcome.error;
      ctx.logger.warn(
        { member, status: outcome.error.httpStatus },
        "failover: member failed, advancing",
      );
    }

    if (attempted === 0) {
      throw new CircuitOpenError(
        `all failover members are circuit-open for '${ctx.request.model}'`,
      );
    }
    throw new AllMembersFailedError(
      `all failover members failed for '${ctx.request.model}'` +
        (lastError ? `: ${lastError.message}` : ""),
    );
  },
};

/** Non-streaming attempt: same-member retry loop, then return or advance. */
async function attemptJsonMember(
  ctx: StrategyContext,
  resilience: Resilience,
  member: string,
): Promise<MemberOutcome> {
  const { breaker, sleep, backoff, policy } = resilience;
  const body: Record<string, unknown> = { ...ctx.request, model: member };
  let rateLimitRetries = 0;
  let serverRetries = 0;

  for (;;) {
    let result: ChatCompletionResult;
    try {
      result = await resilience.limiter(() => ctx.client.chatCompletions(body, { stream: false }));
    } catch (err) {
      breaker.recordFailure(member);
      if (serverRetries < policy.maxServerRetries) {
        await sleep(backoffDelay(serverRetries, backoff));
        serverRetries += 1;
        continue;
      }
      return { kind: "advance", error: toFusionError(err, member) };
    }

    const status = result.status;
    if (status < 400) {
      breaker.recordSuccess(member);
      return { kind: "return", response: buildResponse(result) };
    }
    if (status === 429) {
      breaker.recordFailure(member);
      if (rateLimitRetries < policy.maxRateLimitRetries) {
        await sleep(backoffDelay(rateLimitRetries, backoff));
        rateLimitRetries += 1;
        continue; // SAME member — never advance on 429
      }
      // Exhausted waiting: surface the rate-limit (still not a chain advance).
      return { kind: "return", response: buildResponse(result) };
    }
    if (status >= 500) {
      breaker.recordFailure(member);
      if (serverRetries < policy.maxServerRetries) {
        await sleep(backoffDelay(serverRetries, backoff));
        serverRetries += 1;
        continue;
      }
      return {
        kind: "advance",
        error: new UpstreamNetworkError(`member '${member}' failed with status ${status}`),
      };
    }
    // 4xx other than 429 — client/request error: surface immediately (passthrough).
    return { kind: "return", response: buildResponse(result) };
  }
}

/** Streaming attempt: same as JSON for pre-first-byte errors, plus the peek/commit rule. */
async function attemptStreamMember(
  ctx: StrategyContext,
  resilience: Resilience,
  member: string,
): Promise<MemberOutcome> {
  const { breaker, sleep, backoff, policy } = resilience;
  const body: Record<string, unknown> = { ...ctx.request, model: member };
  let rateLimitRetries = 0;
  let serverRetries = 0;

  for (;;) {
    let result: ChatCompletionResult;
    try {
      result = await resilience.limiter(() => ctx.client.chatCompletions(body, { stream: true }));
    } catch (err) {
      breaker.recordFailure(member);
      if (serverRetries < policy.maxServerRetries) {
        await sleep(backoffDelay(serverRetries, backoff));
        serverRetries += 1;
        continue;
      }
      return { kind: "advance", error: toFusionError(err, member) };
    }

    // Error before the first byte: the client surfaces a JSON body (upstream not ok).
    if (result.kind === "json") {
      const status = result.status;
      if (status === 429) {
        breaker.recordFailure(member);
        if (rateLimitRetries < policy.maxRateLimitRetries) {
          await sleep(backoffDelay(rateLimitRetries, backoff));
          rateLimitRetries += 1;
          continue;
        }
        return { kind: "return", response: buildResponse(result) };
      }
      if (status >= 500) {
        breaker.recordFailure(member);
        if (serverRetries < policy.maxServerRetries) {
          await sleep(backoffDelay(serverRetries, backoff));
          serverRetries += 1;
          continue;
        }
        return {
          kind: "advance",
          error: new UpstreamNetworkError(`member '${member}' failed with status ${status}`),
        };
      }
      if (status >= 400) {
        return { kind: "return", response: buildResponse(result) };
      }
      breaker.recordSuccess(member);
      return { kind: "return", response: buildResponse(result) };
    }

    // kind === "stream": peek the first chunk to decide commit vs. advance.
    const peek = await peekFirstChunk(result.body);
    if (peek.kind === "error") {
      // Failure BEFORE any byte forwarded — safe to retry/advance.
      breaker.recordFailure(member);
      if (serverRetries < policy.maxServerRetries) {
        await sleep(backoffDelay(serverRetries, backoff));
        serverRetries += 1;
        continue;
      }
      return {
        kind: "advance",
        error: new UpstreamNetworkError(
          `member '${member}' stream failed before first token: ${peek.message}`,
        ),
      };
    }

    // Committed. From here a mid-stream failure surfaces as a stream error.
    breaker.recordSuccess(member);
    const headers: Record<string, string> = {
      ...STREAM_HEADERS_BASE,
      "content-type": result.contentType ?? "text/event-stream",
    };
    if (peek.kind === "empty") {
      const empty = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.close();
        },
      });
      return { kind: "return", response: new Response(empty, { status: result.status, headers }) };
    }
    const out = buildCommittedStream(peek.reader, peek.first);
    return { kind: "return", response: new Response(out, { status: result.status, headers }) };
  }
}

/** Read the first chunk of an upstream body, classifying the outcome. */
async function peekFirstChunk(body: ReadableStream<Uint8Array> | null): Promise<Peek> {
  if (!body) return { kind: "empty" };
  const reader = body.getReader();
  try {
    const { value, done } = await reader.read();
    if (done || value === undefined) {
      reader.releaseLock();
      return { kind: "empty" };
    }
    return { kind: "chunk", reader, first: value };
  } catch (err) {
    try {
      await reader.cancel();
    } catch {
      /* already errored — nothing to cancel */
    }
    return { kind: "error", message: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Build the client-facing stream: re-emit the already-read first chunk, then
 * pump the rest. A later upstream read error becomes a stream error on the
 * client (`controller.error`) — the failover loop has already returned, so no
 * other member can be substituted.
 */
function buildCommittedStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  first: Uint8Array,
): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(first);
    },
    async pull(controller) {
      try {
        const { value, done } = await reader.read();
        if (done || value === undefined) {
          controller.close();
          return;
        }
        controller.enqueue(value);
      } catch (err) {
        controller.error(err instanceof Error ? err : new Error(String(err)));
      }
    },
    async cancel(reason) {
      try {
        await reader.cancel(reason);
      } catch {
        /* ignore */
      }
    },
  });
}

/** Build a `Response` from a (non-committed) upstream result, mirroring `single`. */
function buildResponse(result: ChatCompletionResult): Response {
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
}

function toFusionError(err: unknown, member: string): FusionError {
  if (err instanceof FusionError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new UpstreamNetworkError(`member '${member}' call failed: ${message}`);
}
