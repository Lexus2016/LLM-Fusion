import { z } from "zod";
import type { ChatCompletionResult, StrategyContext } from "../types";
import type { Resilience } from "../concurrency";
import { extractAnswer, stripThinkingTags } from "../reasoning";

/**
 * Completeness guard for the SINGLE (passthrough) route — the mirror of the
 * fusion synth guard in `src/strategies/fusion.ts`, but for a bare single-model
 * turn (no panel, no synth).
 *
 * WHY: a reasoning target model (e.g. glm-5.2) inside an agentic tool-calling
 * loop sometimes spends its turn *narrating* the next action ("Let me write the
 * complete HTML file now.", "Пишу повний посібник у файл...") and ends with
 * finish_reason:"stop" and NO tool_calls — the tool call it announced is never
 * emitted. Reasoning-promotion (single.ts) then surfaces that narration as
 * `content`, so the client (OpenCode) sees a finished-looking assistant message
 * with no tool call and ENDS the agent turn; the user must type "continue".
 * The single strategy had no recovery for this (only the fusion synth did).
 *
 * This guard detects the narrate-and-stop and runs ONE stricter, non-streamed
 * retry that forces the tool call. It only runs when the request carried tools
 * (agentic context) — mechanical / tool-less requests are untouched passthrough.
 *
 * PRECISION over recall: a false positive would retry a *legitimately finished*
 * turn and could push the agent into an extra unwanted action, so detection is
 * deliberately narrow — an EMPTY delivered answer, or a non-empty answer whose
 * TAIL is a clear intent-to-act phrase. A genuine completion summary never ends
 * on those. The retry is a single attempt and fails OPEN (keep the original
 * response) if it cannot recover, so the guard can never loop.
 *
 * NOTE: the intent-marker list is a heuristic in the user's working languages
 * (EN/UA/RU). It is intentionally conservative; broaden it only with phrases a
 * completion summary would never end on.
 */

// Intent-to-act tail markers. A narrate-and-stop ends on one of these; a
// completion summary ("the file is complete", "готово") never does.
const TOOL_TURN_INTENT_MARKERS = [
  // English
  "let me write",
  "let's write",
  "let me now write",
  "now i'll write",
  "now i will write",
  "i'll write",
  "i will write",
  "let me create",
  "i'll create",
  "i will create",
  "let me now create",
  "i'll now write",
  "i'll now create",
  "let me produce",
  "let's produce",
  "let me generate",
  "let me start writing",
  // Ukrainian
  "пишу файл",
  "пишу повн",
  "зараз запишу",
  "зараз напишу",
  "напишу файл",
  "створюю",
  "зараз створю",
  "створю файл",
  "пишу посібник",
  // Russian
  "сейчас напишу",
  "создаю",
  "сейчас создам",
  "пишу полностью",
] as const;

const TurnCompletionSchema = z
  .object({
    choices: z
      .array(
        z
          .object({
            finish_reason: z.union([z.string(), z.null()]).optional(),
            message: z
              .object({
                content: z.union([z.string(), z.null()]).optional(),
                tool_calls: z.unknown().optional(),
              })
              .passthrough()
              .optional(),
          })
          .passthrough(),
      )
      .optional(),
  })
  .passthrough();

/**
 * Detect a single-model turn that "stopped" while only narrating the next action
 * (no tool call, no delivered artifact). Returns the failure reason, or null when
 * the turn is complete — which INCLUDES any stop-finish response carrying
 * tool_calls (a tool call is the action) and any non-empty answer that does NOT
 * end on an intent-to-act phrase (a real answer / completion summary is never
 * second-guessed).
 *
 * `finish_reason:"length"` (output-cap truncation) is judged too — the
 * historically confirmed large-file failure mode: the model burns the output
 * budget on reasoning or on a huge tool-call argument and gets cut mid-flight:
 *  - tool_calls present but with UNPARSEABLE (truncated) JSON arguments → the
 *    call is not runnable, the client drops it and the loop stalls → retry.
 *  - no tool_calls and no real `content` (everything died in reasoning) → no
 *    artifact was delivered at all → retry.
 *  - non-empty prose content → an honest length-cut answer; deliver as-is.
 */
export function detectIncompleteToolTurn(
  data: unknown,
): "empty" | "intent_tail" | "broken_tool_call" | null {
  const parsed = TurnCompletionSchema.safeParse(data);
  if (!parsed.success) return null;
  const choice = parsed.data.choices?.[0];
  if (!choice) return null;
  const fin = choice.finish_reason;
  const toolCalls = choice.message?.tool_calls;
  const hasCalls = Array.isArray(toolCalls) && toolCalls.length > 0;
  if (fin === "length") {
    if (hasCalls) return toolCallArgsBroken(toolCalls) ? "broken_tool_call" : null;
    const rawContent = typeof choice.message?.content === "string" ? choice.message.content : "";
    if (stripThinkingTags(rawContent).trim().length === 0) return "empty";
    return null; // honest length-cut prose is still worth delivering
  }
  if (fin !== "stop") return null;
  if (hasCalls) return null;
  // extractAnswer reads content AND reasoning, so a reasoning-only "answer" (the
  // classic thinking-model stall) is judged on its real text.
  const answer = stripThinkingTags(extractAnswer(data) ?? "").trim();
  if (answer.length === 0) return "empty";
  const tail = answer.slice(-140).toLowerCase();
  if (TOOL_TURN_INTENT_MARKERS.some((m) => tail.includes(m))) return "intent_tail";
  return null;
}

/**
 * True when at least one tool call carries a NON-EMPTY arguments string that is
 * not valid JSON — the signature of an output-cap truncation mid-arguments.
 * Empty/absent arguments are NOT judged (some models send "" for no-arg tools);
 * precision over recall, as everywhere in this guard.
 */
function toolCallArgsBroken(toolCalls: unknown[]): boolean {
  for (const tc of toolCalls) {
    const parsed = RecoveredToolCallSchema.safeParse(tc);
    const args = parsed.success ? parsed.data.function?.arguments : undefined;
    if (typeof args !== "string" || args.length === 0) continue;
    try {
      JSON.parse(args);
    } catch {
      return true;
    }
  }
  return false;
}

const TOOL_TURN_NUDGE =
  "Your previous turn described the next action in prose (e.g. \"let me write the file\", " +
  "\"пишу файл\") but ended WITHOUT emitting the tool call, so nothing actually happened. " +
  "Emit the tool call NOW to perform that action. Respond with the tool call only — do not " +
  "restate the plan or narrate what you are about to do. IMPORTANT: large payloads get cut " +
  "off by output limits and upstream stream limits — if the content is large, write only the " +
  "FIRST self-contained portion in this tool call (well under 200 lines) and continue with " +
  "further tool calls on later turns; never attempt the whole thing in one oversized call.";

/** Minimal sink the guard writes SSE bytes to (satisfied by both stream controller kinds). */
interface SseSink {
  enqueue(chunk: Uint8Array): void;
}

/** Append the strict tool-emission nudge as a trailing system turn. */
function appendToolTurnNudge(body: Record<string, unknown>, stream: boolean): Record<string, unknown> {
  const messages = Array.isArray(body.messages) ? [...body.messages] : [];
  messages.push({ role: "system", content: TOOL_TURN_NUDGE });
  return { ...body, messages, stream };
}

/**
 * One stricter, non-streamed retry on the SAME target model, nudged to emit the
 * tool call. Returns the recovered completion data, or null when the retry threw,
 * errored, or STILL narrated without acting (in which case the caller keeps the
 * original response — fail-open, never a loop). At most one upstream call. The
 * retry does not touch the circuit breaker (already recorded for the turn) but
 * its usage IS recorded so cost accounting stays honest.
 */
export async function retryToolTurn(
  ctx: StrategyContext,
  resilience: Resilience | undefined,
  target: string,
  originalBody: Record<string, unknown>,
  reason: "empty" | "intent_tail" | "broken_tool_call",
): Promise<unknown | null> {
  const body = appendToolTurnNudge({ ...originalBody, model: target }, false);
  let result: ChatCompletionResult;
  try {
    result = resilience
      ? await resilience.limiterFor(target)(() => ctx.client.chatCompletions(body, { stream: false, signal: ctx.signal }))
      : await ctx.client.chatCompletions(body, { stream: false, signal: ctx.signal });
  } catch (err) {
    ctx.logger.warn(
      { stage: "single", model: target, err: err instanceof Error ? err.message : String(err) },
      "single: tool-turn recovery retry threw",
    );
    return null;
  }
  ctx.usage?.record(target, result);
  if (result.kind !== "json" || result.status >= 400) {
    ctx.logger.warn({ stage: "single", model: target }, "single: tool-turn recovery retry not usable");
    return null;
  }
  if (detectIncompleteToolTurn(result.data) !== null) {
    ctx.logger.warn({ stage: "single", model: target }, "single: tool-turn recovery retry still narrated without acting");
    return null;
  }
  ctx.logger.info({ stage: "single", model: target, reason }, "single: tool-turn recovery retry emitted the action");
  return result.data;
}

// --- streaming helpers -----------------------------------------------------

/**
 * STREAMING recovery retry: re-ask the target with the strict nudge and forward
 * the retry's chunks to the client LIVE (upstream [DONE] dropped; the caller
 * appends its own). This replaces the old non-streamed recovery on the stream
 * path, which was silent for the whole regeneration — a large-file rewrite
 * (minutes on a cloud upstream) looked like a dead spinner to the client and
 * then died on the ~170s non-stream upstream timeout. Streaming has fast
 * first-byte, is not subject to that timeout profile, and the user SEES the
 * recovery happening. Returns true when at least one data chunk was forwarded
 * (the caller must then NOT emit the held-back original terminal chunk); false
 * when the retry could not start or produced nothing (fail open — the caller
 * delivers the original). One attempt, never recursive: the retry stream is
 * forwarded as-is, not re-guarded.
 */
async function streamRetryToolTurn(
  ctx: StrategyContext,
  resilience: Resilience | undefined,
  target: string,
  originalBody: Record<string, unknown>,
  reason: "empty" | "intent_tail" | "broken_tool_call" | "upstream_cut",
  controller: SseSink,
  encoder: TextEncoder,
): Promise<boolean> {
  const body = appendToolTurnNudge({ ...originalBody, model: target }, true);
  ctx.logger.warn(
    { stage: "single", model: target, reason },
    "single: tool turn narrated without acting; streaming recovery retry",
  );
  let result: ChatCompletionResult;
  try {
    result = resilience
      ? await resilience.limiterFor(target)(() => ctx.client.chatCompletions(body, { stream: true, signal: ctx.signal }))
      : await ctx.client.chatCompletions(body, { stream: true, signal: ctx.signal });
  } catch (err) {
    ctx.logger.warn(
      { stage: "single", model: target, err: err instanceof Error ? err.message : String(err) },
      "single: streaming recovery retry threw",
    );
    return false;
  }
  ctx.usage?.record(target, result);
  if (result.kind !== "stream" || result.status >= 400 || result.body === null) {
    ctx.logger.warn({ stage: "single", model: target, status: result.status }, "single: streaming recovery retry not usable");
    return false;
  }
  const reader = result.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let forwarded = 0;
  const forwardLine = (line: string): void => {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("data:")) {
      const payload = trimmed.slice("data:".length).trim();
      if (payload === "[DONE]") return; // the caller closes the stream itself
      if (payload.length > 0) forwarded += 1;
    }
    controller.enqueue(encoder.encode(line + "\n"));
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) !== -1) {
        forwardLine(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    }
    buf += decoder.decode();
    if (buf.length > 0) forwardLine(buf);
  } catch (err) {
    // Mid-stream failure AFTER chunks reached the client: the retry's partial
    // output is already delivered, so report "handled" — re-sending the original
    // terminal chunk now would splice two answers together.
    ctx.logger.warn(
      { stage: "single", model: target, forwarded, err: err instanceof Error ? err.message : String(err) },
      "single: streaming recovery retry broke mid-stream",
    );
    return forwarded > 0;
  }
  ctx.logger.info(
    { stage: "single", model: target, reason, forwarded },
    forwarded > 0
      ? "single: streaming recovery retry forwarded a replacement turn"
      : "single: streaming recovery retry produced no chunks; delivering the original",
  );
  return forwarded > 0;
}


const RecoveredToolCallSchema = z
  .object({
    id: z.string().optional(),
    type: z.string().optional(),
    function: z.object({ name: z.string().optional(), arguments: z.string().optional() }).passthrough().optional(),
  })
  .passthrough();

const StreamChunkSchema = z
  .object({
    id: z.string().optional(),
    created: z.number().optional(),
    model: z.string().optional(),
    choices: z
      .array(
        z
          .object({
            delta: z
              .object({
                content: z.union([z.string(), z.null()]).optional(),
                reasoning: z.union([z.string(), z.null()]).optional(),
                reasoning_content: z.union([z.string(), z.null()]).optional(),
                tool_calls: z.array(z.unknown()).optional(),
              })
              .passthrough()
              .optional(),
            finish_reason: z.union([z.string(), z.null()]).optional(),
          })
          .passthrough(),
      )
      .optional(),
  })
  .passthrough();

/**
 * Streaming completeness guard for the single route. Every chunk before the
 * terminal (finish_reason-carrying) one is forwarded live and unchanged — a
 * healthy stream is byte-identical to plain passthrough, so first-token latency
 * is untouched. Only the terminal chunk and the trailing [DONE] are held back
 * until the accumulated turn is checked; a narrate-and-stop is replaced by a
 * live-streamed recovery retry before [DONE] is finally sent. Fails OPEN.
 *
 * Deliberately a READER-DRIVEN wrapper, not a TransformStream: when the
 * upstream stream ERRORS mid-flight (Ollama Cloud terminates long generations —
 * observed "terminated" ~5 min into a large-file write), a TransformStream's
 * flush() never runs, so a pipeThrough guard is structurally blind to exactly
 * the failure that stalls the agent loop. Reading the upstream ourselves lets
 * the catch branch run the SAME streaming recovery for a mid-flight cut.
 */
export function makeToolTurnGuardStream(
  ctx: StrategyContext,
  resilience: Resilience | undefined,
  target: string,
  originalBody: Record<string, unknown>,
  upstream: ReadableStream<Uint8Array>,
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const reader = upstream.getReader();
  let buffer = "";
  let content = "";
  let reasoning = "";
  // Streaming tool-call arguments arrive in fragments across chunks (standard
  // OpenAI streaming); accumulate per index so the terminal check can judge the
  // ASSEMBLED arguments for truncation (the length-cut broken-JSON case).
  const toolCallAcc = new Map<number, { id?: string; name?: string; args: string }>();
  let terminalFinishReason: string | null = null;
  let terminalLine: string | null = null;

  const ToolCallDeltaSchema = z
    .object({
      index: z.number().optional(),
      id: z.string().optional(),
      function: z.object({ name: z.string().optional(), arguments: z.string().optional() }).passthrough().optional(),
    })
    .passthrough();

  const accumulateToolCallDelta = (tc: unknown): void => {
    const parsed = ToolCallDeltaSchema.safeParse(tc);
    if (!parsed.success) return;
    const idx = typeof parsed.data.index === "number" ? parsed.data.index : 0;
    const cur = toolCallAcc.get(idx) ?? { args: "" };
    if (parsed.data.id) cur.id = parsed.data.id;
    if (parsed.data.function?.name) cur.name = parsed.data.function.name;
    if (typeof parsed.data.function?.arguments === "string") cur.args += parsed.data.function.arguments;
    toolCallAcc.set(idx, cur);
  };

  const handleLine = (line: string, controller: SseSink): void => {
    if (terminalLine !== null) return; // holding everything after the terminal chunk (incl. [DONE])
    const trimmed = line.trimStart();
    if (!trimmed.startsWith("data:")) {
      controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    const payload = trimmed.slice("data:".length).trim();
    if (payload.length === 0 || payload === "[DONE]") {
      controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    let obj: unknown;
    try {
      obj = JSON.parse(payload);
    } catch {
      controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    const parsed = StreamChunkSchema.safeParse(obj);
    if (!parsed.success) {
      controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    const choice = parsed.data.choices?.[0];
    const delta = choice?.delta;
    if (delta) {
      if (typeof delta.content === "string") content += delta.content;
      if (typeof delta.reasoning === "string") reasoning += delta.reasoning;
      if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
      if (Array.isArray(delta.tool_calls)) for (const tc of delta.tool_calls) accumulateToolCallDelta(tc);
    }
    if (choice?.finish_reason != null) {
      terminalFinishReason = choice.finish_reason;
      terminalLine = line;
      return;
    }
    controller.enqueue(encoder.encode(line + "\n"));
  };

  /** Normal end-of-stream reconciliation (the old flush logic). */
  const finishNormally = async (controller: SseSink): Promise<void> => {
    buffer += decoder.decode();
    if (buffer.length > 0) handleLine(buffer, controller);
    const assembledCalls =
      toolCallAcc.size > 0
        ? [...toolCallAcc.values()].map((c) => ({
            id: c.id,
            type: "function",
            function: { name: c.name, arguments: c.args },
          }))
        : undefined;
    if (terminalLine === null) {
      // Stream ENDED (cleanly) with no finish_reason chunk — same shape as an
      // upstream cut, so recover the same way instead of stalling the client.
      ctx.logger.warn(
        { stage: "single", model: target, tool_calls: toolCallAcc.size, content_len: content.length, reasoning_len: reasoning.length },
        "single: tool stream ended without a terminal chunk; running streaming recovery",
      );
      await runStreamingRecoveryWithKeepalive(ctx, resilience, target, originalBody, "upstream_cut", controller, encoder);
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      return;
    }
    const reconstructed = {
      choices: [
        {
          finish_reason: terminalFinishReason,
          message: { content, reasoning, tool_calls: assembledCalls },
        },
      ],
    };
    const incomplete = detectIncompleteToolTurn(reconstructed);
    // Terminal-state instrumentation: one line per tool-carrying stream, so a
    // real-session stall is diagnosable from the log alone (finish_reason,
    // whether calls/args survived, and what the turn's tail looked like).
    ctx.logger.info(
      {
        stage: "single",
        model: target,
        finish_reason: terminalFinishReason,
        tool_calls: toolCallAcc.size,
        incomplete,
        content_len: content.length,
        reasoning_len: reasoning.length,
        tail: (content || reasoning).slice(-120),
      },
      "single: tool-turn terminal state",
    );
    if (incomplete === null) {
      // SSE events are blank-line delimited: terminalLine is a single split line
      // with its trailing "\n" already stripped, so it needs "\n\n" to close its
      // own event before [DONE] opens the next one.
      controller.enqueue(encoder.encode(terminalLine + "\n\n"));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      return;
    }
    const recovered = await runStreamingRecoveryWithKeepalive(ctx, resilience, target, originalBody, incomplete, controller, encoder);
    if (!recovered) {
      // fail open: the retry never reached the client — deliver the original turn
      controller.enqueue(encoder.encode(terminalLine + "\n\n"));
    }
    controller.enqueue(encoder.encode("data: [DONE]\n\n"));
  };

  /** Mid-flight upstream failure: recover instead of stalling the agent loop. */
  const finishAfterCut = async (controller: SseSink, err: unknown): Promise<void> => {
    ctx.logger.warn(
      {
        stage: "single",
        model: target,
        tool_calls: toolCallAcc.size,
        content_len: content.length,
        reasoning_len: reasoning.length,
        err: err instanceof Error ? err.message : String(err),
      },
      "single: upstream tool stream cut mid-flight; running streaming recovery",
    );
    if (ctx.signal?.aborted) return; // the CLIENT is gone — nobody to recover for
    if (terminalLine !== null) {
      // Cut happened after the terminal chunk was already held back — deliver it.
      controller.enqueue(encoder.encode(terminalLine + "\n\n"));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      return;
    }
    await runStreamingRecoveryWithKeepalive(ctx, resilience, target, originalBody, "upstream_cut", controller, encoder);
    controller.enqueue(encoder.encode("data: [DONE]\n\n"));
  };

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let nl: number;
          while ((nl = buffer.indexOf("\n")) !== -1) {
            handleLine(buffer.slice(0, nl), controller);
            buffer = buffer.slice(nl + 1);
          }
        }
      } catch (err) {
        try {
          await finishAfterCut(controller, err);
        } finally {
          void reader.cancel().catch(() => {});
          controller.close();
        }
        return;
      }
      try {
        await finishNormally(controller);
      } finally {
        controller.close();
      }
    },
    cancel(reason) {
      void reader.cancel(reason).catch(() => {});
    },
  });
}

/**
 * Run the STREAMING tool-turn recovery while keeping the client connection warm
 * until the retry's first bytes arrive: SSE comment lines (": keepalive") are
 * emitted on an interval — protocol-legal no-ops (they keep flowing between the
 * retry's chunks too, which is harmless). Interval override via
 * SINGLE_TOOLTURN_RECOVERY_PING_MS. Any throw fails OPEN (returns false so the
 * caller delivers the original terminal chunk).
 */
async function runStreamingRecoveryWithKeepalive(
  ctx: StrategyContext,
  resilience: Resilience | undefined,
  target: string,
  originalBody: Record<string, unknown>,
  incomplete: "empty" | "intent_tail" | "broken_tool_call" | "upstream_cut",
  controller: SseSink,
  encoder: TextEncoder,
): Promise<boolean> {
  const envPing = Number(process.env.SINGLE_TOOLTURN_RECOVERY_PING_MS ?? "");
  const pingMs = Number.isFinite(envPing) && envPing > 0 ? envPing : 5_000;
  const ping = setInterval(() => {
    try {
      controller.enqueue(encoder.encode(": keepalive\n\n"));
    } catch {
      /* stream already closed — nothing to keep alive */
    }
  }, pingMs);
  try {
    return await streamRetryToolTurn(ctx, resilience, target, originalBody, incomplete, controller, encoder);
  } catch (err) {
    ctx.logger.warn(
      { stage: "single", model: target, err: err instanceof Error ? err.message : String(err) },
      "single: tool-turn stream recovery threw; delivering the original terminal chunk",
    );
    return false;
  } finally {
    clearInterval(ping);
  }
}
