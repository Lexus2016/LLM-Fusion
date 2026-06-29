import type { ChatCompletionResult, Strategy, StrategyContext } from "../types";
import type { Resilience } from "../concurrency";
import { backoffDelay, createResilience } from "../concurrency";
import { AllMembersFailedError, CircuitOpenError, FusionError, UpstreamNetworkError } from "../errors";
import {
  failureKindForError,
  logUpstreamFailure,
} from "../attribution";
import { promoteReasoningNonStream, makeReasoningPromotionTransform } from "../reasoning";

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
  | {
      kind: "chunk";
      /** Reader for the remainder; `null` when the stream already ended (prefix-only). */
      reader: ReadableStreamDefaultReader<Uint8Array> | null;
      /** Bytes read so far (leading keep-alive comments + the committing chunk), in order. */
      prefix: Uint8Array[];
    };

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
    // Reasoning->content promotion flag, threaded into the response builders so a
    // "thinking" chain member returning reasoning-only content is normalized too.
    const promote =
      ctx.modelConfig.promote_reasoning_to_content ?? ctx.config.defaults.promote_reasoning_to_content;

    let attempted = 0;
    let lastError: FusionError | undefined;

    for (const member of chain) {
      if (!resilience.breaker.canAttempt(member)) {
        logUpstreamFailure(ctx.logger, {
          stage: "failover-member",
          model: member,
          kind: "circuit_open",
          latencyMs: 0,
        });
        ctx.logger.warn({ member, model: ctx.request.model }, "failover: skip member (circuit open)");
        lastError = new CircuitOpenError(`circuit breaker open for failover member '${member}'`);
        continue;
      }
      attempted += 1;
      const outcome = stream
        ? await attemptStreamMember(ctx, resilience, member, promote)
        : await attemptJsonMember(ctx, resilience, member, promote);
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
  promote: boolean,
): Promise<MemberOutcome> {
  const { breaker, sleep, backoff, policy } = resilience;
  const body: Record<string, unknown> = { ...ctx.request, model: member };
  let rateLimitRetries = 0;
  let serverRetries = 0;

  for (;;) {
    const startedAt = Date.now();
    let result: ChatCompletionResult;
    try {
      result = await resilience.limiter(() => ctx.client.chatCompletions(body, { stream: false, signal: ctx.signal }));
    } catch (err) {
      // Client disconnect is not a member health failure: do not trip the breaker
      // and do not waste retries. Release any reserved half-open probe so the
      // model can be probed again. Detect via the client signal, not the error
      // name — a stage timeout also aborts the fetch and must still count.
      if (ctx.signal?.aborted) {
        breaker.recordProbeAbandoned(member);
        throw err;
      }
      breaker.recordFailure(member);
      ctx.usage?.recordError(member);
      logUpstreamFailure(ctx.logger, {
        stage: "failover-member",
        model: member,
        kind: failureKindForError(err),
        latencyMs: Date.now() - startedAt,
        reason: err instanceof Error ? err.message : String(err),
      });
      if (serverRetries < policy.maxServerRetries) {
        await sleep(backoffDelay(serverRetries, backoff));
        serverRetries += 1;
        continue;
      }
      return { kind: "advance", error: toFusionError(err, member) };
    }
    ctx.usage?.record(member, result);

    const status = result.status;
    if (status < 400) {
      breaker.recordSuccess(member);
      return { kind: "return", response: buildResponse(result, promote) };
    }
    if (status === 429) {
      breaker.recordFailure(member);
      logUpstreamFailure(ctx.logger, {
        stage: "failover-member",
        model: member,
        kind: "rate_limit",
        status,
        latencyMs: Date.now() - startedAt,
      });
      if (rateLimitRetries < policy.maxRateLimitRetries) {
        await sleep(backoffDelay(rateLimitRetries, backoff));
        rateLimitRetries += 1;
        continue; // SAME member — never advance on 429
      }
      // Exhausted waiting: surface the rate-limit (still not a chain advance).
      return { kind: "return", response: buildResponse(result, promote) };
    }
    if (status >= 500) {
      breaker.recordFailure(member);
      logUpstreamFailure(ctx.logger, {
        stage: "failover-member",
        model: member,
        kind: "server_error",
        status,
        latencyMs: Date.now() - startedAt,
      });
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
    return { kind: "return", response: buildResponse(result, promote) };
  }
}

/** Streaming attempt: same as JSON for pre-first-byte errors, plus the peek/commit rule. */
async function attemptStreamMember(
  ctx: StrategyContext,
  resilience: Resilience,
  member: string,
  promote: boolean,
): Promise<MemberOutcome> {
  const { breaker, sleep, backoff, policy } = resilience;
  const body: Record<string, unknown> = { ...ctx.request, model: member };
  let rateLimitRetries = 0;
  let serverRetries = 0;

  for (;;) {
    const startedAt = Date.now();
    let result: ChatCompletionResult;
    try {
      result = await resilience.limiter(() => ctx.client.chatCompletions(body, { stream: true, signal: ctx.signal }));
    } catch (err) {
      // Client disconnect is not a member health failure: do not trip the breaker
      // and do not waste retries. Release any reserved half-open probe so the
      // model can be probed again. Detect via the client signal, not the error
      // name — a stage timeout also aborts the fetch and must still count.
      if (ctx.signal?.aborted) {
        breaker.recordProbeAbandoned(member);
        throw err;
      }
      breaker.recordFailure(member);
      ctx.usage?.recordError(member);
      logUpstreamFailure(ctx.logger, {
        stage: "failover-member",
        model: member,
        kind: failureKindForError(err),
        latencyMs: Date.now() - startedAt,
        reason: err instanceof Error ? err.message : String(err),
      });
      if (serverRetries < policy.maxServerRetries) {
        await sleep(backoffDelay(serverRetries, backoff));
        serverRetries += 1;
        continue;
      }
      return { kind: "advance", error: toFusionError(err, member) };
    }
    ctx.usage?.record(member, result);

    // Error before the first byte: the client surfaces a JSON body (upstream not ok).
    if (result.kind === "json") {
      const status = result.status;
      if (status === 429) {
        breaker.recordFailure(member);
        logUpstreamFailure(ctx.logger, {
          stage: "failover-member",
          model: member,
          kind: "rate_limit",
          status,
          latencyMs: Date.now() - startedAt,
        });
        if (rateLimitRetries < policy.maxRateLimitRetries) {
          await sleep(backoffDelay(rateLimitRetries, backoff));
          rateLimitRetries += 1;
          continue;
        }
        return { kind: "return", response: buildResponse(result, promote) };
      }
      if (status >= 500) {
        breaker.recordFailure(member);
        logUpstreamFailure(ctx.logger, {
          stage: "failover-member",
          model: member,
          kind: "server_error",
          status,
          latencyMs: Date.now() - startedAt,
        });
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
        return { kind: "return", response: buildResponse(result, promote) };
      }
      breaker.recordSuccess(member);
      return { kind: "return", response: buildResponse(result, promote) };
    }

    // kind === "stream": peek until the stream COMMITS real content to decide
    // commit vs. advance. SSE keep-alive comments/blank lines do not commit.
    const peek = await peekFirstChunk(result.body);
    if (peek.kind === "error") {
      // Client disconnect surfaces here as an AbortError from the peek read. It
      // is NOT a member health failure: do not trip the breaker or waste retries.
      // Release any reserved half-open probe so the model can be probed again.
      if (ctx.signal?.aborted) {
        breaker.recordProbeAbandoned(member);
        throw new UpstreamNetworkError(`member '${member}' cancelled by client disconnect`);
      }
      // Failure BEFORE any content forwarded — safe to retry/advance.
      breaker.recordFailure(member);
      logUpstreamFailure(ctx.logger, {
        stage: "failover-member",
        model: member,
        kind: "error",
        latencyMs: Date.now() - startedAt,
        reason: peek.message,
      });
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
    const out = buildCommittedStream(peek.reader, peek.prefix);
    const committedBody = promote ? out.pipeThrough(makeReasoningPromotionTransform()) : out;
    return { kind: "return", response: new Response(committedBody, { status: result.status, headers }) };
  }
}

/**
 * Read upstream chunks until the stream COMMITS to client content. SSE keep-alive
 * comments (`:`-prefixed lines) and blank separator lines do NOT commit: a
 * failure while only those have arrived is still safe to retry/advance. The first
 * chunk carrying a real field line (e.g. `data:`) commits, and every byte read so
 * far — the leading comments plus the committing chunk — is returned as `prefix`
 * to be re-emitted verbatim, in order, so nothing is lost.
 */
async function peekFirstChunk(body: ReadableStream<Uint8Array> | null): Promise<Peek> {
  if (!body) return { kind: "empty" };
  const reader = body.getReader();
  const prefix: Uint8Array[] = [];
  const decoder = new TextDecoder();
  let text = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done || value === undefined) {
        text += decoder.decode(); // flush any partial multibyte char
        // Stream ended. A trailing (unterminated) content line still commits so a
        // body that is a single newline-less data line is never dropped; a stream
        // of only keep-alive/blank lines collapses to an inert empty stream.
        if (prefix.length > 0 && hasContentLine(text, true)) {
          reader.releaseLock();
          return { kind: "chunk", reader: null, prefix };
        }
        reader.releaseLock();
        return { kind: "empty" };
      }
      prefix.push(value);
      text += decoder.decode(value, { stream: true });
      if (hasContentLine(text, false)) {
        return { kind: "chunk", reader, prefix };
      }
      // Only comments / blank lines so far — keep reading (still pre-commit).
    }
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
 * True when `text` contains an SSE line that COMMITS to content: any complete
 * line (terminated by `\n`) that is non-blank and is not a `:` comment. With
 * `includeTrailing`, an unterminated final line is considered too (used only at
 * end-of-stream, where there will be no further `\n`).
 */
function hasContentLine(text: string, includeTrailing: boolean): boolean {
  const segments = text.split("\n");
  const completeCount = includeTrailing ? segments.length : segments.length - 1;
  for (let i = 0; i < completeCount; i += 1) {
    const seg = segments[i];
    if (seg === undefined) continue;
    const line = seg.endsWith("\r") ? seg.slice(0, -1) : seg;
    if (line.length === 0) continue; // blank separator line
    if (line.startsWith(":")) continue; // SSE comment / keep-alive
    return true; // a real field line (data:, event:, id:, ...) — commit
  }
  return false;
}

/**
 * Build the client-facing stream: re-emit the already-read `prefix` chunks (the
 * leading keep-alive comments, if any, plus the committing chunk), then pump the
 * rest from `reader`. A `null` reader means the upstream stream already ended, so
 * only the prefix is emitted. A later upstream read error becomes a stream error
 * on the client (`controller.error`) — the failover loop has already returned, so
 * no other member can be substituted.
 */
function buildCommittedStream(
  reader: ReadableStreamDefaultReader<Uint8Array> | null,
  prefix: Uint8Array[],
): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of prefix) controller.enqueue(chunk);
      if (reader === null) controller.close();
    },
    async pull(controller) {
      if (reader === null) {
        controller.close();
        return;
      }
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
      if (reader === null) return;
      try {
        await reader.cancel(reason);
      } catch {
        /* ignore */
      }
    },
  });
}

/** Build a `Response` from a (non-committed) upstream result, mirroring `single`. */
function buildResponse(result: ChatCompletionResult, promote: boolean): Response {
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
}

function toFusionError(err: unknown, member: string): FusionError {
  if (err instanceof FusionError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new UpstreamNetworkError(`member '${member}' call failed: ${message}`);
}
