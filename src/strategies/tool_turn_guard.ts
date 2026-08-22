import { z } from "zod";
import type { TextEncoder as NodeTextEncoder } from "node:util";
import type { ChatCompletionResult, StrategyContext } from "../types";
import type { Resilience } from "../concurrency";
import { extractAnswer, stripThinkingTags } from "../reasoning";
import { isJsonObjectString } from "../json";

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
  // Runnability is judged under EVERY finish_reason, not just "length". A call whose
  // arguments are truncated/scalar, or that has no name to dispatch, is unrunnable
  // whatever the upstream claims the outcome was — and `finish_reason:"tool_calls"`
  // on a call the client cannot execute is precisely the stall this guard exists to
  // break. The STREAM path already enforced exactly this via `assembledCallsEmittable`;
  // leaving the shared predicate gated on "length" is what let the NON-STREAM twin
  // (single.ts) ship an unrunnable call with no retry at all.
  if (hasCalls && (toolCallArgsBroken(toolCalls) || toolCallNameMissing(toolCalls))) return "broken_tool_call";
  if (fin === "length") {
    if (hasCalls) return null; // calls present and runnable: an honest tool turn
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
 * finish_reasons for which re-prompting is never appropriate, however unrunnable the
 * turn is. `content_filter` means the upstream REFUSED: nudging it with "Emit the tool
 * call NOW" re-runs the refusal — it burns a call and at best returns the same block,
 * at worst launders a safety stop into a second attempt. The turn is still made HONEST
 * (the unrunnable call is dropped, the `content_filter` terminal is preserved); it is
 * only the retry that is suppressed.
 */
const NO_RETRY_FINISH_REASONS = new Set(["content_filter"]);

/** True when this response's finish_reason forbids a recovery retry. See above. */
export function toolTurnRetryBlocked(data: unknown): boolean {
  const parsed = TurnCompletionSchema.safeParse(data);
  if (!parsed.success) return false;
  const fin = parsed.data.choices?.[0]?.finish_reason;
  return typeof fin === "string" && NO_RETRY_FINISH_REASONS.has(fin);
}

/**
 * True when at least one tool call carries a NON-EMPTY arguments string that is
 * not a JSON OBJECT — either truncated mid-arguments (the output-cap signature)
 * or a scalar/array, which parses but is not runnable tool input.
 * Empty/absent arguments are NOT judged (some models send "" for no-arg tools);
 * precision over recall, as everywhere in this guard.
 */
function toolCallArgsBroken(toolCalls: unknown[]): boolean {
  for (const tc of toolCalls) {
    const parsed = RecoveredToolCallSchema.safeParse(tc);
    const args = parsed.success ? parsed.data.function?.arguments : undefined;
    if (typeof args !== "string" || args.length === 0) continue;
    // Not merely "does it parse": a scalar or array is not runnable tool input.
    if (!isJsonObjectString(args)) return true;
  }
  return false;
}

/**
 * True when at least one tool call has no dispatchable `function.name`. A nameless
 * call is unrunnable however well-formed its arguments are — there is nothing to look
 * up. `toolCallArgsBroken` inspects `arguments` only and misses it entirely; the
 * stream path's `assembledCallsEmittable` has always required a name.
 * NOTE: this also rejects a tool call that is not `{function:{name,arguments}}`-shaped
 * at all (e.g. OpenAI's newer `type:"custom"` calls). That is deliberate parity with
 * the stream path, whose accumulator is built from `function.name` and cannot
 * represent such a call either; supporting them is a feature, not part of this fix.
 */
function toolCallNameMissing(toolCalls: unknown[]): boolean {
  for (const tc of toolCalls) {
    const parsed = RecoveredToolCallSchema.safeParse(tc);
    if (!parsed.success) return true;
    const name = parsed.data.function?.name;
    // `.trim()`: a whitespace-only name is as undispatchable as an absent one —
    // there is no such tool to look up — and JSON round-trips it happily.
    if (typeof name !== "string" || name.trim().length === 0) return true;
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
  encoder: NodeTextEncoder,
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
  let sawFinishReason = false;
  const retryCalls: ToolCallAcc = new Map();
  /**
   * Forward ONLY `data:` lines, each re-framed with its own blank separator. Every
   * other line of the retry stream is DROPPED rather than spliced into the original
   * turn. The cosmetic half is the blank separator that used to trail the swallowed
   * `[DONE]`, leaving a stray empty line ahead of the fail-open terminal. The
   * non-cosmetic half is the SSE `event:` field: it is STICKY — it names the type of
   * the NEXT dispatched event, and on the fail-open path that next event is the
   * guard's OWN terminal chunk. A retry that died after writing `event: error` would
   * therefore re-label the original terminal as an error event, and `id:` fields would
   * likewise rewrite the client's Last-Event-ID from a stream it is not reading.
   * Neither is ours to relay: the retry is spliced INTO another turn, not proxied.
   * Re-framing also guarantees a clean frame boundary for the synthesised terminal
   * below (the payload string itself is never re-serialised, so no JSON round-trip).
   * Known limitation: a multi-line `data:` event would be split into one event per
   * line. No LLM upstream emits those, and `handleLine` cannot parse them either.
   */
  const forwardLine = (line: string): void => {
    const trimmed = line.trimStart();
    if (!trimmed.startsWith("data:")) return;
    const payload = trimmed.slice("data:".length).trim();
    if (payload === "[DONE]") return; // the caller closes the stream itself
    if (payload.length === 0) return;
    // Classify BEFORE enqueueing — the order is the point. While the line went out
    // first, a chunk carrying ONLY a terminal had to be counted as a replacement turn
    // (otherwise failing open would put a SECOND finish_reason on the wire), which
    // meant a retry answering with nothing but `finish_reason:"tool_calls"` suppressed
    // the held original terminal and delivered an EMPTY tool turn: no call to run, no
    // prose — precisely the actionless terminal this guard exists to eliminate.
    // Deciding first lets a bare terminal be DROPPED instead: not forwarded, not
    // counted, so no double-terminal is possible and the fail-open path takes over.
    // A terminal that FOLLOWS real content is still forwarded and still latches
    // `sawFinishReason` — that one is the retry's honest end, not an empty turn.
    const signal = classifyRecoveryChunk(payload, retryCalls);
    if (!signal.substantive && !(signal.terminal && forwarded > 0)) return;
    if (signal.substantive) forwarded += 1;
    if (signal.terminal) sawFinishReason = true;
    controller.enqueue(encoder.encode(`data: ${payload}\n\n`));
  };
  /**
   * The retry delivered a usable turn but never a terminal chunk — it was cut, or the
   * upstream simply closed without one. Closing on a bare `[DONE]` here would hand the
   * client a turn with no finish_reason at all, which is exactly the hole this whole
   * function's return value is supposed to cover, so synthesise one. The VALUE is
   * evidence-driven, not a blanket `"length"`:
   *   - the retry forwarded tool-call fragments that assemble into a runnable call →
   *     `"tool_calls"`. The client holds a complete, executable call; that IS why the
   *     turn ended, and `"length"` would be the lie.
   *   - fragments that do NOT assemble (truncated args, or no name) → `"length"`. The
   *     client holds broken JSON; `"tool_calls"` would order it to execute garbage,
   *     while `"length"` → Anthropic `max_tokens` prompts a clean re-ask.
   *   - no fragments, stream ERRORED mid-flight → `"length"`. Demonstrably a cut.
   *   - no fragments, stream ended CLEANLY without a terminal → `"stop"`. This is the
   *     one case with no strictly honest answer: a clean EOF is ambiguous between a
   *     sloppy upstream that never emits finish_reason and an intermediary that closed
   *     tidily on a truncated turn. `"stop"` is chosen because the content was fully
   *     forwarded and nothing reported an error; `"length"` would make every turn from
   *     such an upstream look truncated and drive endless auto-continuation.
   */
  const synthesiseTerminal = (cut: boolean): void => {
    const calls = buildAssembledCalls(retryCalls);
    const finish =
      calls !== undefined ? (assembledCallsRunnable(calls) ? "tool_calls" : "length") : cut ? "length" : "stop";
    const chunk = { choices: [{ index: 0, delta: {}, finish_reason: finish }] };
    controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
    ctx.logger.warn(
      { stage: "single", model: target, finish_reason: finish, cut },
      "single: streaming recovery retry ended without a terminal; synthesised one",
    );
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
    if (forwarded === 0) return false;
    if (!sawFinishReason) synthesiseTerminal(true);
    return true;
  }
  ctx.logger.info(
    { stage: "single", model: target, reason, forwarded },
    forwarded > 0
      ? "single: streaming recovery retry forwarded a replacement turn"
      : "single: streaming recovery retry produced no chunks; delivering the original",
  );
  if (forwarded === 0) return false;
  if (!sawFinishReason) synthesiseTerminal(false);
  return true;
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

/** Index-keyed accumulator for streamed tool-call fragments. */
type ToolCallAcc = Map<number, { id?: string; name?: string; args: string }>;

const ToolCallDeltaSchema = z
  .object({
    index: z.number().optional(),
    id: z.string().optional(),
    function: z.object({ name: z.string().optional(), arguments: z.string().optional() }).passthrough().optional(),
  })
  .passthrough();

/** Fold one `delta.tool_calls[]` entry into the accumulator, keyed by its index. */
function accumulateToolCallDelta(acc: ToolCallAcc, tc: unknown): void {
  const parsed = ToolCallDeltaSchema.safeParse(tc);
  if (!parsed.success) return;
  const idx = typeof parsed.data.index === "number" ? parsed.data.index : 0;
  const cur = acc.get(idx) ?? { args: "" };
  if (parsed.data.id) cur.id = parsed.data.id;
  if (parsed.data.function?.name) cur.name = parsed.data.function.name;
  if (typeof parsed.data.function?.arguments === "string") cur.args += parsed.data.function.arguments;
  acc.set(idx, cur);
}

/**
 * Message-shaped view of an accumulator's tool call(s) (`{id,type,function}`), or
 * undefined when none were accumulated. Used to judge completeness via
 * `toolCallArgsBroken` / `assembledCallsRunnable` before deciding whether to emit
 * the assembled call, recover, or (on the retry path) what terminal is honest.
 */
function buildAssembledCalls(
  acc: ToolCallAcc,
): { id?: string; type: string; function: { name?: string; arguments: string } }[] | undefined {
  if (acc.size === 0) return undefined;
  return [...acc.values()].map((c) => ({
    id: c.id,
    type: "function",
    function: { name: c.name, arguments: c.args },
  }));
}

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
function assembledCallsRunnable(calls: { function: { name?: string; arguments: string } }[]): boolean {
  return (
    calls.length > 0 &&
    calls.every((c) => {
      if (!c.function.name?.trim() || c.function.arguments.length === 0) return false;
      return isJsonObjectString(c.function.arguments);
    })
  );
}

/**
 * Judge ONE recovery chunk: does it carry a replacement TURN, and does it carry a
 * TERMINAL? `forwarded > 0` is what tells the caller to SUPPRESS the held original
 * terminal, so the bar has to be "the client received something it can use", not
 * merely "some bytes arrived". Two shapes cleared the old `payload.length > 0` bar
 * while delivering nothing usable:
 *   - `{"choices":[{"delta":{"role":"assistant"}}]}` — the opener nearly every
 *     upstream sends first. A retry that dies immediately after it reached the catch
 *     branch's `return forwarded > 0` and reported success.
 *   - `{"error":{...}}` on a 200 response — Ollama, vLLM and OpenRouter all signal
 *     mid-stream failure this way, so the `status >= 400` check never sees it.
 * Either one suppressed the original terminal and left the turn with NO
 * finish_reason at all (`stop_reason: null` once src/anthropic.ts translates it).
 * A terminal-only chunk is NOT substantive: an empty `finish_reason` chunk is an empty
 * turn, and `forwardLine` drops it before it reaches the wire (see the ordering note
 * there) rather than letting it suppress the held original terminal.
 * Tool-call fragments are accumulated here as a side effect, because the terminal
 * synthesised below depends on whether they assemble into a runnable call.
 */
function classifyRecoveryChunk(payload: string, acc: ToolCallAcc): { substantive: boolean; terminal: boolean } {
  let obj: unknown;
  try {
    obj = JSON.parse(payload);
  } catch {
    return { substantive: false, terminal: false };
  }
  const parsed = StreamChunkSchema.safeParse(obj);
  if (!parsed.success) return { substantive: false, terminal: false };
  const choice = parsed.data.choices?.[0];
  if (choice === undefined) return { substantive: false, terminal: false };
  const delta = choice.delta;
  const nonEmpty = (v: string | null | undefined): boolean => typeof v === "string" && v.length > 0;
  let substantive = nonEmpty(delta?.content) || nonEmpty(delta?.reasoning) || nonEmpty(delta?.reasoning_content);
  if (Array.isArray(delta?.tool_calls) && delta.tool_calls.length > 0) {
    substantive = true;
    for (const tc of delta.tool_calls) accumulateToolCallDelta(acc, tc);
  }
  const terminal = choice.finish_reason !== null && choice.finish_reason !== undefined;
  return { substantive, terminal };
}

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
 * Rewrite a terminal chunk's `finish_reason` to `"length"`.
 *
 * Only for the fail-open path, and only when the buffered tool call was DROPPED as
 * unrunnable: the turn then carries no tool call at all, and a surviving
 * `finish_reason: "tool_calls"` announces one that is not there. An OpenAI-compatible
 * client that trusts it goes looking for a call to execute, finds none, and either
 * throws or stalls waiting to append a `tool` message it cannot construct.
 * `"length"` is the honest signal: the output was cut short. Best-effort — a line
 * that is not a parseable `data: ` chunk is returned unchanged.
 */
function markTerminalLengthCut(line: string): string {
  // Parse EXACTLY as handleLine does: `line.trimStart()`, then `data:` with the
  // space OPTIONAL, then `.trim()` on the payload. The terminal handed to us is the
  // RAW upstream line whenever that chunk carried no tool_call fragments of its own,
  // so ANY divergence from handleLine silently skips the rewrite on a line handleLine
  // already accepted — and the client gets `finish_reason:"tool_calls"` for a call
  // that was dropped. Two divergences existed: the missing `trimStart()`, and the
  // missing payload `.trim()`. The latter is NOT redundant with JSON.parse's own
  // whitespace tolerance: JSON.parse accepts only space/tab/CR/LF, while
  // String.prototype.trim also strips U+00A0, U+FEFF, U+2028 and the rest of the
  // Unicode WhiteSpace set, so a payload handleLine parsed happily would throw here.
  const trimmed = line.trimStart();
  if (!trimmed.startsWith("data:")) return line;
  try {
    const obj = JSON.parse(trimmed.slice("data:".length).trim());
    const choice = Array.isArray(obj?.choices) ? obj.choices[0] : undefined;
    if (!choice || typeof choice !== "object") return line;
    // ONLY `tool_calls` is the lie worth correcting. A turn that already ended on
    // `stop` or `length` describes itself honestly even with the call dropped, and
    // rewriting it would fabricate a token cap that was never hit — `intent_tail`
    // reaches this branch carrying `stop`.
    if (choice.finish_reason !== "tool_calls") return line;
    choice.finish_reason = "length";
    return `data: ${JSON.stringify(obj)}`;
  } catch {
    return line;
  }
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
  const toolCallAcc: ToolCallAcc = new Map();
  let terminalFinishReason: string | null = null;
  let terminalLine: string | null = null;
  /**
   * Latched on the first parsed chunk of a MULTI-CHOICE stream (`n > 1`). `n` reaches
   * the upstream verbatim (ChatCompletionRequestSchema is `.passthrough()` and
   * single.ts forwards the request body), so a multi-choice stream is reachable — and
   * every part of this guard's state machine is single-choice: one `toolCallAcc`, one
   * `terminalLine`, one `content`/`reasoning` pair, and `stripToolCallsFromChunk` /
   * `markTerminalLengthCut` both look at `choices[0]` only. Half-applying it produced
   * strictly WRONG output, not merely incomplete: a terminal rewritten for choice 0
   * while choice 1 kept `finish_reason:"tool_calls"` (one chunk saying both that a call
   * is coming and that it is not), and a `choices[1]` tool-call fragment forwarded LIVE
   * at `index: 0`, which is exactly the option-B corruption the guard exists to prevent.
   * So: latch, and become a pure passthrough for the rest of the stream. Extending the
   * state machine per choice is a feature, not a fix — no client of this proxy sends
   * `n > 1` today, and an untested per-choice machine is a worse bet than none.
   */
  const requestedChoices = ((): number => {
    const n = originalBody.n;
    if (typeof n === "number") return n;
    if (typeof n === "string" && n.trim().length > 0) return Number(n);
    return 1;
  })();
  /**
   * Latched from the REQUEST, before a single byte of the response is seen. Sniffing
   * the response cannot work on the standard wire format: an OpenAI-compatible n>1
   * stream sends ONE CHOICE PER CHUNK, so `choices.length > 1` never fires and the
   * first chunk is an ordinary `index: 0` — indistinguishable from n=1. The guard
   * would therefore treat choice 0 as a single-choice turn and WITHHOLD its tool-call
   * fragments (option B), then latch on the first `index: 1` chunk, at which point
   * `finishNormally`'s multi-choice early-out emits `[DONE]` and the withheld call is
   * lost outright. `n` is in the body the client sent us, so ask it there instead.
   */
  let multiChoice = Number.isFinite(requestedChoices) && requestedChoices > 1;
  if (multiChoice) {
    ctx.logger.warn(
      { stage: "single", model: target, n: requestedChoices },
      "single: multi-choice request (n>1); tool-turn guard disabled for this turn",
    );
  }

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

  /**
   * Emittability check for a CLEAN terminal/end (finishNormally / reconcile). Looser
   * than `assembledCallsRunnable`: an empty-arguments call is fine here (a genuinely
   * finished no-arg tool sends `arguments: ""`), but a call is NOT emittable if it
   * lacks a name or carries non-empty arguments that are not a JSON object — those go to
   * recovery. Also the emit-time gate on the fail-open path, where it decides whether
   * the buffered call is handed to the client or dropped and the terminal rewritten.
   */
  const assembledCallsEmittable = (
    calls: { function: { name?: string; arguments: string } }[],
  ): boolean =>
    calls.length > 0 &&
    calls.every((c) => {
      if (!c.function.name?.trim()) return false; // whitespace-only is as undispatchable as absent
      if (c.function.arguments.length === 0) return true; // no-arg tool on a clean finish
      return isJsonObjectString(c.function.arguments);
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

  /**
   * Hand back whatever option B was holding when the multi-choice latch fires LATE
   * (an upstream that returns n>1 the client did not ask for). Passthrough starts
   * from the next line, so anything still buffered — assembled tool calls, a held
   * choice-0 terminal — has to go out now or it never does.
   */
  const flushHeldStateForLateMultiChoice = (controller: SseSink): void => {
    emitAssembledToolCalls(controller);
    toolCallAcc.clear();
    if (terminalLine !== null) {
      controller.enqueue(encoder.encode(terminalLine + "\n\n"));
      terminalLine = null;
      terminalFinishReason = null;
    }
  };

  const handleLine = (line: string, controller: SseSink): void => {
    // Holding everything after the terminal chunk (incl. [DONE]) — but the multi-choice
    // DETECTOR below still gets to see these lines. On a sequential n>1 stream choice
    // 0's terminal arrives before choice 1's first chunk, so returning here first would
    // silently discard every later choice instead of latching into passthrough.
    const held = terminalLine !== null;
    const trimmed = line.trimStart();
    if (!trimmed.startsWith("data:")) {
      if (!held) controller.enqueue(encoder.encode(line + "\n"));
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
      if (!held) controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    let obj: unknown;
    try {
      obj = JSON.parse(payload);
    } catch {
      if (!held) controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    const parsed = StreamChunkSchema.safeParse(obj);
    if (!parsed.success) {
      if (!held) controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    const choices = parsed.data.choices;
    if (!multiChoice && choices !== undefined && choices.length > 0) {
      // Secondary net only — `n` from the request is the primary latch. It catches an
      // upstream that returns n>1 unasked, and it cannot fire before the first `index`
      // that is not 0, which on a sequential stream is chunk 2 at the earliest.
      // `index` must be compared NUMERICALLY but not TYPE-strictly: a single-choice
      // stream may omit it entirely (Ollama, llama.cpp) and `undefined !== 0` would
      // latch every such stream into passthrough, disabling the guard everywhere;
      // meanwhile an upstream serialising it as the STRING "1" would slip past
      // `typeof === "number"` and leave the guard on for a real multi-choice stream.
      // A non-numeric string yields NaN, which is `!== 0` — latching off is the
      // conservative reading of an `index` we cannot interpret.
      const firstIndex = choices[0]?.index;
      const firstIndexIsNonZero =
        typeof firstIndex === "number"
          ? firstIndex !== 0
          : typeof firstIndex === "string" && firstIndex.trim().length > 0 && Number(firstIndex) !== 0;
      multiChoice = choices.length > 1 || firstIndexIsNonZero;
      if (multiChoice) {
        ctx.logger.warn(
          { stage: "single", model: target, choices: choices.length, index: firstIndex },
          "single: multi-choice stream (n>1) not declared by the request; tool-turn guard disabled",
        );
        flushHeldStateForLateMultiChoice(controller);
      }
    }
    if (multiChoice) {
      controller.enqueue(encoder.encode(line + "\n"));
      return;
    }
    if (held) return;
    const choice = choices?.[0];
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
        for (const tc of delta.tool_calls) accumulateToolCallDelta(toolCallAcc, tc);
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
    const assembledCalls = buildAssembledCalls(toolCallAcc);
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
      // Belt-and-braces. `detectIncompleteToolTurn` now judges runnability under every
      // finish_reason (it used to only do so for "length"), so this arm is not known to
      // be reachable — the two checks differ only in that `assembledCallsEmittable`
      // additionally rejects non-object JSON arguments, which `toolCallArgsBroken`
      // already covers. Kept because the two predicates are maintained separately and
      // the failure it guards against — shipping a nameless call as runnable — is worse
      // than a redundant test.
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
    // A refusal is not a malfunction to retry — see NO_RETRY_FINISH_REASONS. Skipping
    // straight to the fail-open path keeps the turn honest (unrunnable call dropped,
    // `content_filter` terminal preserved) without re-running the filtered turn.
    const retryBlocked = terminalFinishReason !== null && NO_RETRY_FINISH_REASONS.has(terminalFinishReason);
    if (retryBlocked) {
      ctx.logger.warn(
        { stage: "single", model: target, finish_reason: terminalFinishReason, incomplete },
        "single: unrunnable tool turn under a no-retry finish_reason; dropping the call without a retry",
      );
    }
    const recovered = retryBlocked
      ? false
      : await runStreamingRecoveryWithKeepalive(ctx, resilience, target, originalBody, incomplete, controller, encoder);
    if (!recovered) {
      // Fail open: the retry never reached the client — deliver the original
      // terminal so the turn ends honestly (its finish_reason signals the cut).
      // Only re-emit the buffered call if it is actually RUNNABLE — never hand the
      // client a nameless/truncated tool call to execute.
      const dropped = assembledCalls !== undefined && !assembledCallsEmittable(assembledCalls);
      if (assembledCalls !== undefined && !dropped) emitAssembledToolCalls(controller);
      // Dropping the call leaves the turn with NOTHING to execute, so the original
      // terminal's `finish_reason: "tool_calls"` would be a lie about the payload
      // the client just received — rewrite it to the cut it actually was. When no
      // call was buffered at all (narrate-and-stop) the terminal is already honest
      // and is passed through untouched.
      controller.enqueue(encoder.encode((dropped ? markTerminalLengthCut(terminal) : terminal) + "\n\n"));
    }
    controller.enqueue(encoder.encode("data: [DONE]\n\n"));
  };

  /** Normal end-of-stream reconciliation (the old flush logic). */
  const finishNormally = async (controller: SseSink): Promise<void> => {
    buffer += decoder.decode();
    if (buffer.length > 0) handleLine(buffer, controller);
    if (multiChoice) {
      // Passthrough stream: nothing was buffered or held, so there is nothing to
      // reconcile. Without this the `terminalLine === null` branch below would read
      // the empty accumulators as "the stream delivered nothing" and fire a billed
      // recovery retry against a turn that completed perfectly.
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      return;
    }
    const assembledCalls = buildAssembledCalls(toolCallAcc);
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
    if (multiChoice) {
      // Passthrough stream: every byte the upstream produced was already forwarded, so
      // the client is committed and a spliced replacement would duplicate it. Same
      // honest-failure semantics as the committed-content case at the bottom.
      controller.error(err instanceof Error ? err : new Error(String(err)));
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
    const assembledCalls = buildAssembledCalls(toolCallAcc);
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

  // Draining lives in pull(), NOT start(): start() runs to completion regardless of
  // whether anyone reads, and controller.enqueue() never blocks, so an upstream that
  // outruns the client would land in the stream queue in full — a whole generation
  // resident in memory per stalled connection. pull() is re-invoked only as the
  // consumer drains, so the queue itself becomes the backpressure signal on
  // reader.read(). Same reason fusion.ts and reasoning.ts pipe through a
  // TransformStream; this path needs the explicit form because it also has to
  // recover, splice and finish the turn.
  let ended = false;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (ended) return;
      try {
        for (;;) {
          const { done, value } = await reader.read();
          // The consumer can go away WHILE this read is in flight, and then the very
          // next handleLine enqueues into a dead controller, throws, and the catch
          // below reads that throw as an upstream cut — firing a billed recovery
          // request for a client that already left. So the check belongs HERE, ahead
          // of handleLine; the desiredSize check at the bottom of the loop is one
          // enqueue too late. Two distinct deaths, and desiredSize alone cannot tell
          // them apart: per spec it is NULL only when the stream ERRORED, while a
          // cancelled/closed stream reports 0 — indistinguishable from ordinary
          // backpressure. Cancel is caught by the `ended` latch that cancel() sets.
          if (ended || controller.desiredSize === null) {
            ended = true;
            void reader.cancel().catch(() => {});
            return;
          }
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let nl: number;
          while ((nl = buffer.indexOf("\n")) !== -1) {
            handleLine(buffer.slice(0, nl), controller);
            buffer = buffer.slice(nl + 1);
          }
          // Queue full — yield. The stream calls pull() again once the client reads,
          // and reader.read() stays un-awaited until then, stalling the upstream.
          // Lines that only buffered tool fragments leave desiredSize untouched and
          // keep the loop running, which is what recovery needs. NULL (errored) exits
          // here too; a 0 from a closed stream is caught at the top of the next
          // iteration, before anything enqueues.
          const desired = controller.desiredSize;
          if (desired === null || desired <= 0) return;
        }
      } catch (err) {
        ended = true;
        try {
          // finishAfterCut ends the stream itself (close on recovery/abort,
          // error on an honest mid-stream failure — close() after error() would throw).
          await finishAfterCut(controller, err);
        } finally {
          void reader.cancel().catch(() => {});
        }
        return;
      }
      // Backstop, not the live path. A cancel() that arrives while this pull is
      // parked on reader.read() latches `ended` AND resolves that read as done —
      // which would land here looking exactly like an upstream that finished. The
      // check at the TOP of the loop catches that first (it runs before the `done`
      // break), so this line is unreachable today; it stays as the guarantee that
      // moving or weakening the top check cannot silently turn a client walking
      // away into a reconciled turn plus a billed recovery nobody will read.
      if (ended) return;
      ended = true;
      try {
        await finishNormally(controller);
      } finally {
        controller.close();
      }
    },
    cancel(reason) {
      // Latch as well as release: pull() may be parked on backpressure, and once the
      // consumer is gone there is nothing left to finish or recover.
      ended = true;
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
  encoder: NodeTextEncoder,
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
