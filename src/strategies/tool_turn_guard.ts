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
 * Deep-clone an SSE chunk with any `delta.tool_calls` removed. Option B withholds
 * tool-call fragments from the live stream and re-emits ONE assembled call at the
 * end; if a terminal (finish_reason) chunk ALSO carried tool-call fragments, the
 * raw fragment must be stripped before that held chunk is forwarded — otherwise an
 * index-keyed client would concatenate it onto the assembled/recovered arguments,
 * reintroducing the exact corruption this guard prevents. Best-effort: returns the
 * clone unchanged when the shape is not the expected chunk shape.
 */
function stripToolCallsFromChunk(chunk: unknown): unknown {
  const clone = JSON.parse(JSON.stringify(chunk));
  const choice = Array.isArray(clone?.choices) ? clone.choices[0] : undefined;
  if (choice && typeof choice.delta === "object" && choice.delta !== null) {
    delete choice.delta.tool_calls;
  }
  return clone;
}

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
 * the catch branch react to a mid-flight cut at all: recover when NOTHING was
 * forwarded to the client yet, or fail the stream honestly when it was — a
 * spliced replacement turn would duplicate prose and corrupt the client's
 * tool-call argument assembly (see finishAfterCut).
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

  /**
   * True when nothing that COMMITS THE CLIENT to this turn has reached it yet —
   * only then can a recovery retry replace the turn wholesale. Recovery-eligibility
   * is judged on client-VISIBLE bytes only: content/reasoning that reached the
   * client cannot be unsent, so a replacement spliced after it would duplicate the
   * prose. Tool-call fragments are BUFFERED (option B: never forwarded live), so
   * they are NOT client-visible — a mid-stream cut after only tool fragments is
   * still cleanly recoverable (nothing to concatenate). Role-only deltas and SSE
   * comments/keepalives are content-free and likewise never block recovery.
   */
  const nothingReachedClient = (): boolean => content === "" && reasoning === "";

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

  /**
   * Message-shaped view of the buffered tool call(s) (`{id,type,function}`), or
   * undefined when none were buffered. Used to judge completeness via
   * `toolCallArgsBroken` / `detectIncompleteToolTurn` before deciding whether to
   * emit the assembled call or recover.
   */
  const buildAssembledCalls = ():
    | { id?: string; type: string; function: { name?: string; arguments: string } }[]
    | undefined =>
    toolCallAcc.size > 0
      ? [...toolCallAcc.values()].map((c) => ({
          id: c.id,
          type: "function",
          function: { name: c.name, arguments: c.args },
        }))
      : undefined;

  /**
   * STRICT runnability check for the mid-flight-cut SALVAGE path only. On a cut
   * (no finish_reason) we cannot tell an empty-arguments no-arg tool call from a
   * call truncated BEFORE its arguments began — so, unlike `toolCallArgsBroken`
   * (which ignores empty args, correct for a CLEAN finish where empty means a
   * no-arg tool), salvage requires every call to have a name AND non-empty
   * arguments that parse as JSON. Anything short of that recovers instead — the
   * safe choice, since re-asking yields a clean call rather than executing a tool
   * with empty/partial input.
   */
  const assembledCallsRunnable = (
    calls: { function: { name?: string; arguments: string } }[],
  ): boolean =>
    calls.length > 0 &&
    calls.every((c) => {
      if (!c.function.name || c.function.arguments.length === 0) return false;
      try {
        JSON.parse(c.function.arguments);
        return true;
      } catch {
        return false;
      }
    });

  /**
   * Emittability check for a CLEAN terminal/end (finishNormally / reconcile). Looser
   * than `assembledCallsRunnable`: an empty-arguments call is fine here (a genuinely
   * finished no-arg tool sends `arguments: ""`), but a call is NOT emittable if it
   * lacks a name or carries non-empty arguments that do not parse — those go to
   * recovery. Complements `detectIncompleteToolTurn`, which only inspects broken
   * args for `finish_reason:"length"`; this catches a broken/nameless call under
   * ANY finish reason (e.g. a truncated `finish_reason:"tool_calls"`).
   */
  const assembledCallsEmittable = (
    calls: { function: { name?: string; arguments: string } }[],
  ): boolean =>
    calls.length > 0 &&
    calls.every((c) => {
      if (!c.function.name) return false;
      if (c.function.arguments.length === 0) return true; // no-arg tool on a clean finish
      try {
        JSON.parse(c.function.arguments);
        return true;
      } catch {
        return false;
      }
    });

  /**
   * Emit the buffered tool call(s) as ONE reconstructed OpenAI streaming delta
   * chunk (option B). Partial fragments were withheld from the live stream, so the
   * client receives each call's `arguments` exactly once, already complete — an
   * index-keyed accumulator (openai-python, Vercel AI SDK, OpenCode) can no longer
   * concatenate a truncated fragment with a recovered one. Returns true when a
   * chunk was emitted (i.e. at least one call was buffered).
   */
  const emitAssembledToolCalls = (controller: SseSink): boolean => {
    if (toolCallAcc.size === 0) return false;
    const tool_calls = [...toolCallAcc.entries()].map(([index, c]) => ({
      index,
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.args },
    }));
    const chunk = { choices: [{ index: 0, delta: { tool_calls } }] };
    controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
    return true;
  };

  const handleLine = (line: string, controller: SseSink): void => {
    if (terminalLine !== null) return; // holding everything after the terminal chunk (incl. [DONE])
    const trimmed = line.trimStart();
    if (!trimmed.startsWith("data:")) {
      controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    const payload = trimmed.slice("data:".length).trim();
    // Swallow the upstream [DONE]: every finish branch of this guard appends its
    // own. Forwarding the upstream one would double-frame the stream when the
    // upstream ends cleanly WITHOUT a finish_reason chunk (recovery chunks and a
    // second [DONE] after the client already saw one). Found in post-release
    // review; in production the usage-injection transform downstream happened to
    // normalize it, but the guard's own framing must be canonical regardless.
    if (payload === "[DONE]") return;
    if (payload.length === 0) {
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
    let hadToolCalls = false;
    let hadVisibleText = false;
    if (delta) {
      if (typeof delta.content === "string") {
        content += delta.content;
        if (delta.content.length > 0) hadVisibleText = true;
      }
      if (typeof delta.reasoning === "string") {
        reasoning += delta.reasoning;
        if (delta.reasoning.length > 0) hadVisibleText = true;
      }
      if (typeof delta.reasoning_content === "string") {
        reasoning += delta.reasoning_content;
        if (delta.reasoning_content.length > 0) hadVisibleText = true;
      }
      if (Array.isArray(delta.tool_calls)) {
        hadToolCalls = true;
        for (const tc of delta.tool_calls) accumulateToolCallDelta(tc);
      }
    }
    if (choice?.finish_reason != null) {
      terminalFinishReason = choice.finish_reason;
      // Hold the terminal chunk until reconciliation. If it ALSO carried tool-call
      // fragments, strip them from the held line — the assembled call is emitted
      // separately (option B); forwarding the raw fragment here would let an
      // index-keyed client concatenate it onto the assembled/recovered arguments.
      terminalLine = hadToolCalls ? `data: ${JSON.stringify(stripToolCallsFromChunk(obj))}` : line;
      return;
    }
    if (hadToolCalls) {
      // Option B: NEVER forward a tool-call fragment live — a length-cut mid-args
      // truncation would otherwise reach the client and the recovery retry (which
      // restarts at index:0) would make the client concatenate truncated + recovered
      // `arguments` into invalid JSON. The buffered call is re-emitted whole at the
      // terminal reconciliation. BUT a MIXED chunk that ALSO carries content/reasoning
      // must still deliver that text (tool_calls stripped) — otherwise `content`/
      // `reasoning` state would record text the client never saw, corrupting the
      // `nothingReachedClient()` recovery decision (it would think the client was
      // committed and wrongly decline a safe recovery / error). Pure tool-call
      // fragments (no visible text) are suppressed as before. Same single-"\n"
      // framing as the raw-line path — the following blank separator line completes
      // the "\n\n" SSE frame, so this never double-frames.
      if (hadVisibleText) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(stripToolCallsFromChunk(obj))}\n`));
      }
      return;
    }
    controller.enqueue(encoder.encode(line + "\n"));
  };

  /**
   * Terminal reconciliation for a HELD finish_reason chunk. Reconstruct the turn,
   * judge completeness, then either emit the assembled call + terminal + [DONE]
   * (complete) OR run streaming recovery (broken_tool_call / empty / intent_tail),
   * failing open to the original terminal. Shared by BOTH the normal end-of-stream
   * path and finishAfterCut's post-terminal branch: once the finish_reason chunk is
   * in hand the turn is a normal finish that merely lost its trailing [DONE] (to a
   * late upstream error, in the cut case), so the SAME reconciliation applies — a
   * broken terminal turn must RECOVER, never ship a dead/actionless terminal.
   * Does NOT close the stream (the caller owns that). `terminal` is the held line.
   */
  const reconcileTerminalTurn = async (controller: SseSink, terminal: string): Promise<void> => {
    const assembledCalls = buildAssembledCalls();
    const reconstructed = {
      choices: [
        {
          finish_reason: terminalFinishReason,
          message: { content, reasoning, tool_calls: assembledCalls },
        },
      ],
    };
    const incomplete =
      detectIncompleteToolTurn(reconstructed) ??
      // detectIncompleteToolTurn only inspects broken args for finish_reason
      // "length"; a truncated/nameless call under any OTHER finish (e.g.
      // "tool_calls") would otherwise be emitted as runnable. Catch it here.
      (assembledCalls && !assembledCallsEmittable(assembledCalls) ? ("broken_tool_call" as const) : null);
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
      // Complete/runnable turn: emit the buffered tool call(s) as ONE assembled
      // chunk BEFORE the terminal + [DONE], so the client sees each call's
      // arguments exactly once, already whole (option B). A no-op when the turn
      // carried no tool calls (honest length-cut prose, narrate-and-stop that
      // passed, plain answer) — content was already streamed live.
      emitAssembledToolCalls(controller);
      // SSE events are blank-line delimited: `terminal` is a single split line with
      // its trailing "\n" already stripped, so it needs "\n\n" to close its own
      // event before [DONE] opens the next one.
      controller.enqueue(encoder.encode(terminal + "\n\n"));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      return;
    }
    // Incomplete (broken_tool_call / empty / intent_tail): the buffered call is
    // broken or absent — do NOT emit it. Because nothing was forwarded for the
    // tool call, the recovery splices a FRESH call and the client sees only the
    // clean recovered arguments (no concatenation) — this PRESERVES recovery.
    const recovered = await runStreamingRecoveryWithKeepalive(ctx, resilience, target, originalBody, incomplete, controller, encoder);
    if (!recovered) {
      // Fail open: the retry never reached the client — deliver the original
      // terminal so the turn ends honestly (its finish_reason signals the cut).
      // Only re-emit the buffered call if it is actually RUNNABLE — never hand the
      // client a nameless/truncated tool call to execute. A broken one is dropped
      // (the terminal chunk already tells the client the turn was truncated).
      if (assembledCalls && assembledCallsEmittable(assembledCalls)) emitAssembledToolCalls(controller);
      controller.enqueue(encoder.encode(terminal + "\n\n"));
    }
    controller.enqueue(encoder.encode("data: [DONE]\n\n"));
  };

  /** Normal end-of-stream reconciliation (the old flush logic). */
  const finishNormally = async (controller: SseSink): Promise<void> => {
    buffer += decoder.decode();
    if (buffer.length > 0) handleLine(buffer, controller);
    const assembledCalls = buildAssembledCalls();
    if (terminalLine === null) {
      // Stream ENDED (cleanly) with no finish_reason chunk.
      if (assembledCalls && assembledCallsEmittable(assembledCalls)) {
        // A COMPLETE buffered tool call, withheld from the live stream (option B).
        // A clean end means the call IS the turn's result, so deliver it now — no
        // recovery (that would restart at index:0 and duplicate the call).
        emitAssembledToolCalls(controller);
        ctx.logger.warn(
          { stage: "single", model: target, tool_calls: toolCallAcc.size },
          "single: tool stream ended without a terminal chunk; delivered the buffered tool call(s)",
        );
      } else if (nothingReachedClient()) {
        // Either the buffered call is TRUNCATED (clean end MID-arguments) or the
        // stream was empty — and nothing reached the client, so recover a complete
        // turn instead of emitting an unparseable call / stalling the agent loop.
        ctx.logger.warn(
          { stage: "single", model: target, tool_calls: toolCallAcc.size },
          "single: tool stream ended without a terminal chunk before anything reached the client; running streaming recovery",
        );
        await runStreamingRecoveryWithKeepalive(
          ctx,
          resilience,
          target,
          originalBody,
          assembledCalls ? "broken_tool_call" : "upstream_cut",
          controller,
          encoder,
        );
      } else {
        // Content/reasoning already reached the client: a recovery would deliver
        // the whole answer a SECOND time, so close with our own [DONE]. A broken
        // buffered call (if any) is DROPPED rather than sending the client invalid
        // JSON spliced after the prose.
        ctx.logger.warn(
          { stage: "single", model: target, content_len: content.length, reasoning_len: reasoning.length, tool_calls: toolCallAcc.size },
          "single: tool stream ended without a terminal chunk after partial output was forwarded; closing without recovery",
        );
      }
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      return;
    }
    // A finish_reason chunk was held back — run the shared terminal reconciliation.
    await reconcileTerminalTurn(controller, terminalLine);
  };

  /**
   * Mid-flight upstream failure. Recovery is safe ONLY when nothing that commits
   * the client to this turn has reached it yet (then the retry IS the whole
   * answer). Content/reasoning that was already delivered cannot be unsent, so a
   * replacement turn spliced after it would duplicate prose — that case is
   * propagated honestly via `controller.error` (matching failover.ts's
   * committed-stream semantics) and the client's own retry kicks in. Tool-call
   * fragments are BUFFERED (option B, never forwarded), so a cut after ONLY tool
   * fragments left the client uncommitted and IS recoverable. This function takes
   * over ending the stream in every path (close, or error).
   */
  const finishAfterCut = async (controller: ReadableStreamDefaultController<Uint8Array>, err: unknown): Promise<void> => {
    ctx.logger.warn(
      {
        stage: "single",
        model: target,
        tool_calls: toolCallAcc.size,
        content_len: content.length,
        reasoning_len: reasoning.length,
        err: err instanceof Error ? err.message : String(err),
      },
      "single: upstream tool stream cut mid-flight",
    );
    if (ctx.signal?.aborted) {
      controller.close(); // the CLIENT is gone — nobody to recover for
      return;
    }
    if (terminalLine !== null) {
      // Cut happened after the terminal chunk was already held back: the turn is a
      // normal finish that merely lost its trailing [DONE] to a late upstream error.
      // Run the SAME terminal reconciliation as finishNormally — emit a complete
      // call, or RECOVER a broken/empty one — instead of forwarding a dead terminal.
      await reconcileTerminalTurn(controller, terminalLine);
      controller.close();
      return;
    }
    // A COMPLETE buffered tool call survived the cut (upstream emitted the whole
    // call before dying, just not a finish_reason chunk): SALVAGE it — it was
    // withheld from the live stream (option B), so there is nothing to concatenate
    // and any content was already streamed. Mirrors finishNormally's clean-end
    // complete-call path, and covers the mixed content+complete-call case too
    // (which would otherwise error after forwarding content). Only a BROKEN/absent
    // buffered call falls through to recovery / honest error below.
    const assembledCalls = buildAssembledCalls();
    if (assembledCalls && assembledCallsRunnable(assembledCalls)) {
      emitAssembledToolCalls(controller);
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
      return;
    }
    if (nothingReachedClient()) {
      // Nothing client-visible reached the client (buffered tool fragments do not
      // count — they were never forwarded) and the buffered call, if any, is
      // broken: recover a clean turn instead of stalling the loop.
      await runStreamingRecoveryWithKeepalive(ctx, resilience, target, originalBody, "upstream_cut", controller, encoder);
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
      return;
    }
    // Content/reasoning already delivered and no complete call to salvage: fail the
    // stream honestly (a splice would duplicate the delivered prose).
    controller.error(err instanceof Error ? err : new Error(String(err)));
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
          // finishAfterCut ends the stream itself (close on recovery/abort,
          // error on an honest mid-stream failure — close() after error() would throw).
          await finishAfterCut(controller, err);
        } finally {
          void reader.cancel().catch(() => {});
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
