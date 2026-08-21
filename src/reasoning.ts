import { z } from "zod";

/**
 * Reasoning -> content normalization, shared by every strategy. Some Ollama
 * Cloud "thinking" models return their final answer in `reasoning` /
 * `reasoning_content` with an empty `content`; clients that render only
 * `message.content` would then see nothing. These helpers (a) read the effective
 * text regardless of which field carries it, and (b) optionally rewrite empty
 * content from reasoning, for both non-streamed and streamed responses.
 */

/**
 * A non-streamed assistant message, including the `reasoning` /
 * `reasoning_content` fields some Ollama Cloud "thinking" models populate
 * instead of `content`.
 */
const ReasoningMessageSchema = z
  .object({
    content: z.union([z.string(), z.null()]).optional(),
    reasoning: z.union([z.string(), z.null()]).optional(),
    reasoning_content: z.union([z.string(), z.null()]).optional(),
    tool_calls: z.unknown().optional(),
  })
  .passthrough();

export type ReasoningMessage = z.infer<typeof ReasoningMessageSchema>;

const CompletionSchema = z
  .object({
    choices: z.array(z.object({ message: ReasoningMessageSchema }).passthrough()).optional(),
    // Native /api/chat shape: { message: { content } }.
    message: ReasoningMessageSchema.optional(),
  })
  .passthrough();

/**
 * Removes inline thinking from assistant text. First strips complete
 * `<think>…</think>` blocks (models like DeepSeek-R/QwQ inline their reasoning in
 * `content` this way — without this, the whole reasoning leaks into the answer),
 * then strips any orphan opening/closing tag left over (some Ollama "thinking"
 * models put reasoning in a separate field and leave a bare `</think>` marker in
 * `content`). Block removal must run first so the inline case is handled, with the
 * orphan-tag pass preserved for the separate-field case.
 */
export function stripThinkingTags(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>/gi, "")
    .replace(/<\/think>/gi, "");
}

/**
 * Stateful <think>-tag filter for STREAMED text. Per-fragment
 * `stripThinkingTags` leaks on streams: an SSE boundary can split the tag
 * itself ("<th" + "ink>") and a block's body arrives in later fragments, so
 * both the literal tag and the private reasoning reach the client. `push()`
 * returns the visible part of a fragment, carrying a possible partial tag
 * across the boundary; text inside an open block is suppressed until its
 * close tag. `flush()` returns any leftover carry (a false partial at the
 * very end of the stream); an unterminated block stays suppressed — a
 * dangling "<think…" at stream end is a truncated tag, not prose.
 */
export interface ThinkTagStreamFilter {
  push(fragment: string): string;
  flush(): string;
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

export function createThinkTagStreamFilter(): ThinkTagStreamFilter {
  let inside = false;
  let carry = "";

  /** Longest suffix of `s` that is a case-insensitive prefix of `tag` (never the whole tag). */
  const partialSuffix = (s: string, tag: string): number => {
    const max = Math.min(s.length, tag.length - 1);
    for (let len = max; len > 0; len--) {
      if (s.slice(s.length - len).toLowerCase() === tag.slice(0, len)) return len;
    }
    return 0;
  };

  const push = (fragment: string): string => {
    let s = carry + fragment;
    carry = "";
    let out = "";
    for (;;) {
      const lower = s.toLowerCase();
      if (inside) {
        const close = lower.indexOf(THINK_CLOSE);
        if (close !== -1) {
          s = s.slice(close + THINK_CLOSE.length);
          inside = false;
          continue;
        }
        // Everything here is private; keep only a possible partial close tag.
        const keep = partialSuffix(s, THINK_CLOSE);
        carry = keep > 0 ? s.slice(s.length - keep) : "";
        return out;
      }
      const open = lower.indexOf(THINK_OPEN);
      const orphanClose = lower.indexOf(THINK_CLOSE);
      // An orphan close tag outside a block is stripped, matching stripThinkingTags.
      if (orphanClose !== -1 && (open === -1 || orphanClose < open)) {
        out += s.slice(0, orphanClose);
        s = s.slice(orphanClose + THINK_CLOSE.length);
        continue;
      }
      if (open !== -1) {
        out += s.slice(0, open);
        s = s.slice(open + THINK_OPEN.length);
        inside = true;
        continue;
      }
      const keep = Math.max(partialSuffix(s, THINK_OPEN), partialSuffix(s, THINK_CLOSE));
      if (keep > 0) {
        out += s.slice(0, s.length - keep);
        carry = s.slice(s.length - keep);
      } else {
        out += s;
      }
      return out;
    }
  };

  const flush = (): string => {
    const rest = inside ? "" : carry;
    carry = "";
    inside = false;
    return rest;
  };

  return { push, flush };
}

/** First non-empty string among the candidates, else "". */
export function firstNonEmpty(...values: Array<string | null | undefined>): string {
  for (const v of values) {
    if (typeof v === "string" && v.length > 0) return v;
  }
  return "";
}

/** The reasoning text of a message: `reasoning`, then `reasoning_content`, else "". */
function reasoningText(message: ReasoningMessage): string {
  return firstNonEmpty(message.reasoning, message.reasoning_content);
}

/**
 * Effective assistant text for a message: `content` when it has non-whitespace
 * text, otherwise the model's `reasoning` / `reasoning_content`. Some "thinking"
 * models return their final answer in `reasoning` with an empty `content`; the
 * judge (and any text consumer) needs that real text ALWAYS — independent of the
 * `promote_reasoning_to_content` flag.
 */
export function effectiveText(message: ReasoningMessage | undefined): string {
  if (!message) return "";
  const content = typeof message.content === "string" ? message.content : "";
  if (content.trim().length > 0) return content;
  return reasoningText(message);
}

/** Extract assistant text from an OpenAI- or native-shaped completion. */
export function extractAnswer(data: unknown): string | null {
  const parsed = CompletionSchema.safeParse(data);
  if (!parsed.success) return null;
  const fromChoices = effectiveText(parsed.data.choices?.[0]?.message);
  if (fromChoices.length > 0) return fromChoices;
  const fromNative = effectiveText(parsed.data.message);
  if (fromNative.length > 0) return fromNative;
  return null;
}

/**
 * Non-stream normalization: when a message has empty/whitespace `content`,
 * no tool calls, and non-empty reasoning, promote the reasoning into `content`
 * so content-only clients render the answer. Returns the (possibly rewritten)
 * data; all unrelated fields are preserved.
 */
export function promoteReasoningNonStream(data: unknown): unknown {
  const parsed = CompletionSchema.safeParse(data);
  if (!parsed.success) return data;
  const messages: ReasoningMessage[] = [];
  for (const choice of parsed.data.choices ?? []) messages.push(choice.message);
  if (parsed.data.message) messages.push(parsed.data.message);
  let mutated = false;
  for (const message of messages) {
    let content = typeof message.content === "string" ? message.content : "";
    if (content.length > 0) {
      const cleaned = stripThinkingTags(content);
      if (cleaned !== content) {
        message.content = cleaned;
        mutated = true;
      }
    }
    const promotedContent = typeof message.content === "string" ? message.content : "";
    const hasToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
    const reasoning = stripThinkingTags(reasoningText(message));
    if (promotedContent.trim().length === 0 && !hasToolCalls && reasoning.length > 0) {
      message.content = reasoning;
      mutated = true;
    }
    // Strip the raw fields whatever happened above, matching the streaming
    // transform: on the promotion path a reasoning field left on the wire is a
    // leak into any client that renders that channel.
    if (message.reasoning !== undefined || message.reasoning_content !== undefined) {
      delete message.reasoning;
      delete message.reasoning_content;
      mutated = true;
    }
  }
  return mutated ? parsed.data : data;
}

// --- Streaming reasoning->content normalization ---------------------------

const StreamDeltaSchema = z
  .object({
    content: z.union([z.string(), z.null()]).optional(),
    reasoning: z.union([z.string(), z.null()]).optional(),
    reasoning_content: z.union([z.string(), z.null()]).optional(),
    tool_calls: z.unknown().optional(),
  })
  .passthrough();

const StreamChunkSchema = z
  .object({
    choices: z
      .array(
        z
          .object({
            index: z.number().int().optional(),
            delta: StreamDeltaSchema.optional(),
            finish_reason: z.union([z.string(), z.null()]).optional(),
          })
          .passthrough(),
      )
      .optional(),
  })
  .passthrough();

/**
 * Cap on buffered reasoning text. A reasoning-only reply is the answer, so the
 * buffer is kept whole up to this size; past it the TAIL is preserved — a
 * runaway chain-of-thought puts its conclusion last, so dropping the opening
 * costs less than dropping the answer.
 */
export const REASONING_BUFFER_MAX = 1_048_576;

/**
 * SSE transform that surfaces `delta.reasoning` / `delta.reasoning_content` as
 * `delta.content` ONLY when the stream never produces a real `delta.content` —
 * the reasoning-only case this normalization exists for.
 *
 * Reasoning fragments are BUFFERED, never streamed inline. Which kind of model
 * is on the other end is not knowable while the reasoning arrives: a
 * reasoning-only model puts its ANSWER in that field, while a thinking model
 * puts private chain-of-thought there and the answer in later `content` deltas.
 * Emitting eagerly got the second case wrong — the chain-of-thought was streamed
 * into the same text the client renders, glued to the front of the real answer
 * with no separator. So the buffer is dropped the moment real content appears,
 * and flushed as content only at end of stream.
 *
 * `tool_calls` deltas are passed through untouched, but their PRESENCE is
 * latched: a tool turn's reasoning is never the answer. `finish_reason` values
 * are likewise never rewritten — they only mark where a choice's buffered
 * answer must be flushed. Only a partial trailing line is buffered for
 * parsing — never the whole response.
 */
export function makeReasoningPromotionTransform(): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  // Chunk metadata captured from the stream, so synthetic tail chunks carry
  // the same id/model shape as the real ones (strict parsers reject bare deltas).
  const meta: { id?: string; created?: number; model?: string } = {};

  interface ChoiceState {
    /** A non-whitespace answer fragment has been seen on the content channel. */
    realContentSeen: boolean;
    /**
     * A tool turn carries no prose: the model reasons, then calls a tool. Its
     * reasoning must NEVER be flushed as the answer — that would inject private
     * chain-of-thought into an agent loop's visible transcript. Mirrors the
     * tool_calls guard in promoteReasoningNonStream.
     */
    toolCallsSeen: boolean;
    /** Cut off by the token limit before producing any answer text. */
    truncated: boolean;
    /**
     * This choice already flushed at its finish_reason. Anything arriving after
     * that is trailing chain-of-thought: it must never re-buffer under a fresh
     * latch and reach the client at [DONE].
     */
    flushed: boolean;
    /** Reasoning held back until end of stream — see the transform's doc comment. */
    reasoningBuffer: string;
    // Separate stream-filters per source: a tag split across fragments is only
    // meaningful WITHIN one field, and an unterminated block in `reasoning` must
    // not suppress later real `content`.
    contentFilter: ThinkTagStreamFilter;
    reasoningFilter: ThinkTagStreamFilter;
  }

  // Per choice index, never shared: with `n>1` the choices are independent
  // streams, and one choice's content must not suppress another's reasoning.
  const states = new Map<number, ChoiceState>();
  const stateFor = (index: number): ChoiceState => {
    let s = states.get(index);
    if (s === undefined) {
      s = {
        realContentSeen: false,
        toolCallsSeen: false,
        truncated: false,
        flushed: false,
        reasoningBuffer: "",
        contentFilter: createThinkTagStreamFilter(),
        reasoningFilter: createThinkTagStreamFilter(),
      };
      states.set(index, s);
    }
    return s;
  };
  // A chunk this transform could not parse but that mentions tool_calls still
  // marks the turn as a tool turn: failing open would flush chain-of-thought
  // into an agent transcript, which is the exact leak this transform prevents.
  let unparsedToolCalls = false;

  const syntheticChunkLine = (index: number, content: string): string =>
    `data: ${JSON.stringify({
      ...(meta.id !== undefined ? { id: meta.id } : {}),
      object: "chat.completion.chunk",
      ...(meta.created !== undefined ? { created: meta.created } : {}),
      ...(meta.model !== undefined ? { model: meta.model } : {}),
      choices: [{ index, delta: { content } }],
    })}`;

  /**
   * Leftover carry flushed as one synthetic content chunk per choice. Pass the
   * indices that just finished; omit to flush every choice (end of stream). A
   * flushed choice is MARKED, not removed — deleting it would let a trailing
   * chunk recreate it with a fresh latch and leak late chain-of-thought.
   *
   * Accepted tradeoff: `unparsedToolCalls` is a substring test on a chunk this
   * transform could not read, so an unreadable chunk that merely mentions
   * tool_calls suppresses a reasoning-only answer for the whole stream. Failing
   * closed loses an answer; failing open leaks chain-of-thought.
   */
  const tailChunkLine = (only?: number[]): string | null => {
    const lines: string[] = [];
    for (const [index, s] of states) {
      if (only !== undefined && !only.includes(index)) continue;
      if (s.flushed) continue;
      // The buffered reasoning is the answer only when this choice produced no
      // real content AND no tool call; otherwise it was chain-of-thought.
      const suppress = s.realContentSeen || s.toolCallsSeen || s.truncated || unparsedToolCalls;
      const reasoningTail = suppress ? "" : s.reasoningBuffer + s.reasoningFilter.flush();
      s.reasoningBuffer = "";
      if (suppress) s.reasoningFilter.flush();
      const tail = s.contentFilter.flush() + reasoningTail;
      // Kept, not deleted: a deleted entry would be recreated with a fresh
      // latch by any trailing chunk, letting late chain-of-thought through.
      s.flushed = true;
      if (tail.length > 0) lines.push(syntheticChunkLine(index, tail));
    }
    return lines.length > 0 ? lines.join("\n\n") : null;
  };

  const handleLine = (line: string): string => {
    // `trimStart()` before the prefix test: without it a `data:` line carrying any
    // leading whitespace misses this transform entirely and is forwarded VERBATIM,
    // reasoning fields and all. That is a fail-OPEN in a transform whose whole
    // purpose is to fail closed (see `unparsedToolCalls` below) — the leading-space
    // form is not spec-legal SSE, but the cost of accepting it is one `trimStart()`
    // and the cost of rejecting it is chain-of-thought in an agent transcript.
    const trimmed = line.trimStart();
    if (!trimmed.startsWith("data:")) return line; // blank separators, comments, etc.
    const payload = trimmed.slice("data:".length).trim();
    if (payload.length === 0) return line;
    if (payload === "[DONE]") {
      const tail = tailChunkLine();
      // SSE events are blank-line delimited — the synthetic tail chunk must
      // close its OWN event before [DONE] opens the next one (a single \n
      // would merge them into one event and break client-side JSON.parse).
      return tail !== null ? `${tail}\n\n${line}` : line;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      if (payload.includes('"tool_calls"')) unparsedToolCalls = true;
      return line;
    }
    const parsed = StreamChunkSchema.safeParse(chunk);
    if (!parsed.success || !parsed.data.choices) {
      if (payload.includes('"tool_calls"')) unparsedToolCalls = true;
      return line;
    }
    const raw = chunk as { id?: unknown; created?: unknown; model?: unknown };
    if (typeof raw.id === "string") meta.id = raw.id;
    if (typeof raw.created === "number") meta.created = raw.created;
    if (typeof raw.model === "string") meta.model = raw.model;
    let modified = false;
    const finished: number[] = [];
    for (const choice of parsed.data.choices) {
      const index = choice.index ?? 0;
      const s = stateFor(index);
      if (choice.finish_reason != null) {
        finished.push(index);
        // A stream cut by the token limit before it ever produced content has
        // no answer to show: what sits in the buffer is a truncated train of
        // thought. Fail closed, as with an unreadable tool_calls chunk.
        if (choice.finish_reason === "length") s.truncated = true;
      }
      const delta = choice.delta;
      if (!delta) continue;
      if (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) s.toolCallsSeen = true;
      else if (delta.tool_calls != null && !Array.isArray(delta.tool_calls)) s.toolCallsSeen = true;
      const content = typeof delta.content === "string" ? delta.content : "";
      if (content.length > 0) {
        const cleaned = s.contentFilter.push(content);
        // Latch on the FILTERED text, and on non-whitespace only. Content that
        // is entirely an inline <think> block is chain-of-thought, not the
        // answer — latching on the raw delta would drop a reasoning-only
        // reply's buffer and then strip that same content, leaving nothing.
        // The non-whitespace rule matches promoteReasoningNonStream.
        if (cleaned.trim().length > 0) {
          s.realContentSeen = true;
          // The answer arrived, so everything buffered from `reasoning` was
          // private chain-of-thought. Drop it (buffer AND the filter's carry)
          // instead of prepending it to the text the client renders.
          s.reasoningBuffer = "";
          s.reasoningFilter.flush();
        }
        if (cleaned !== content) {
          delta.content = cleaned;
          modified = true;
        }
      }
      if (choice.finish_reason != null) {
        // Consume the content filter's carry HERE, appended to this chunk's own
        // text. Leaving it to the synthetic tail would emit it BEFORE this
        // chunk (the tail precedes the terminating chunk), so "answer <thi"
        // would reach the client as "<thi" + "answer ".
        const carry = s.contentFilter.flush();
        if (carry.length > 0) {
          delta.content = (typeof delta.content === "string" ? delta.content : "") + carry;
          modified = true;
        }
      }
      // Strip the raw reasoning fields from EVERY chunk on this path, buffering
      // only while the answer channel is still silent. Promotion means the
      // client asked for content-only normalization, so a reasoning field left
      // on the wire is a leak into any client that renders that channel —
      // including the late fragments that arrive after the answer starts.
      const rawReasoning = firstNonEmpty(delta.reasoning, delta.reasoning_content);
      if (delta.reasoning !== undefined || delta.reasoning_content !== undefined) {
        delete delta.reasoning;
        delete delta.reasoning_content;
        modified = true;
      }
      // Buffer only while this choice is still live and its answer channel is
      // silent. After the choice flushed at its finish_reason, trailing
      // reasoning is stripped but never re-buffered — it is late
      // chain-of-thought, not a second answer.
      if (rawReasoning.length > 0 && !s.realContentSeen && !s.flushed) {
        s.reasoningBuffer += s.reasoningFilter.push(rawReasoning);
        if (s.reasoningBuffer.length > REASONING_BUFFER_MAX) {
          s.reasoningBuffer = s.reasoningBuffer.slice(-REASONING_BUFFER_MAX);
        }
      }
    }
    const out = modified ? `data: ${JSON.stringify(parsed.data)}` : line;
    if (finished.length > 0) {
      // Deliver a reasoning-only answer BEFORE the terminating chunk: clients
      // that stop accumulating at finish_reason would otherwise never see it.
      // Only the choices that actually finished are flushed — with `n>1` the
      // others may still be streaming.
      const tail = tailChunkLine(finished);
      if (tail !== null) return `${tail}\n\n${out}`;
    }
    return out;
  };

  return new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      let out = "";
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        out += handleLine(line) + "\n";
      }
      if (out.length > 0) controller.enqueue(encoder.encode(out));
    },
    flush(controller) {
      buffer += decoder.decode();
      // A stream that ends mid-line leaves that line unterminated; the
      // synthetic tail below would concatenate onto it and produce one
      // unparseable `data:` line instead of two events.
      if (buffer.length > 0) controller.enqueue(encoder.encode(`${handleLine(buffer)}\n\n`));
      // Stream ended without [DONE]: still surface a false-partial tail as its
      // own blank-line-closed SSE event.
      const tail = tailChunkLine();
      if (tail !== null) controller.enqueue(encoder.encode(`${tail}\n\n`));
    },
  });
}
