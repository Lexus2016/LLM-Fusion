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
    if (promotedContent.trim().length > 0) continue; // real content already present
    if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) continue; // tool path
    const reasoning = stripThinkingTags(reasoningText(message));
    if (reasoning.length === 0) continue;
    message.content = reasoning;
    mutated = true;
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
    choices: z.array(z.object({ delta: StreamDeltaSchema.optional() }).passthrough()).optional(),
  })
  .passthrough();

/**
 * SSE transform that re-emits `delta.reasoning` / `delta.reasoning_content`
 * fragments as `delta.content`, but ONLY until a real `delta.content` fragment
 * appears; once real content arrives, every later event passes through verbatim
 * (no duplication). `tool_calls` deltas and `finish_reason` are never touched.
 * Only a partial trailing line is buffered — never the whole response.
 */
export function makeReasoningPromotionTransform(): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let realContentSeen = false;
  // Separate stream-filters per source: a tag split across fragments is only
  // meaningful WITHIN one field, and an unterminated block in `reasoning` must
  // not suppress later real `content`.
  const contentFilter = createThinkTagStreamFilter();
  const reasoningFilter = createThinkTagStreamFilter();
  // Chunk metadata captured from the stream, so synthetic tail chunks carry
  // the same id/model shape as the real ones (strict parsers reject bare deltas).
  const meta: { id?: string; created?: number; model?: string } = {};
  let reasoningTailMerged = false;

  const syntheticChunkLine = (content: string): string =>
    `data: ${JSON.stringify({
      ...(meta.id !== undefined ? { id: meta.id } : {}),
      object: "chat.completion.chunk",
      ...(meta.created !== undefined ? { created: meta.created } : {}),
      ...(meta.model !== undefined ? { model: meta.model } : {}),
      choices: [{ index: 0, delta: { content } }],
    })}`;

  /** Leftover carry flushed as one synthetic content chunk (own SSE event). */
  const tailChunkLine = (): string | null => {
    const tail = contentFilter.flush() + reasoningFilter.flush();
    if (tail.length === 0) return null;
    return syntheticChunkLine(tail);
  };

  const handleLine = (line: string): string => {
    if (!line.startsWith("data:")) return line; // blank separators, comments, etc.
    const payload = line.slice("data:".length).trim();
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
      return line;
    }
    const parsed = StreamChunkSchema.safeParse(chunk);
    if (!parsed.success || !parsed.data.choices) return line;
    const raw = chunk as { id?: unknown; created?: unknown; model?: unknown };
    if (typeof raw.id === "string") meta.id = raw.id;
    if (typeof raw.created === "number") meta.created = raw.created;
    if (typeof raw.model === "string") meta.model = raw.model;
    let modified = false;
    for (const choice of parsed.data.choices) {
      const delta = choice.delta;
      if (!delta) continue;
      const content = typeof delta.content === "string" ? delta.content : "";
      if (content.length > 0) {
        realContentSeen = true; // real content — leave this and every later event alone
        let cleaned = contentFilter.push(content);
        // The reasoning phase is over: a false-partial tag carried at its end
        // is literal narration text and belongs BEFORE the first content, not
        // reordered to the end of the stream.
        if (!reasoningTailMerged) {
          reasoningTailMerged = true;
          cleaned = reasoningFilter.flush() + cleaned;
        }
        if (cleaned !== content) {
          delta.content = cleaned;
          modified = true;
        }
      } else if (!realContentSeen) {
        const rawReasoning = firstNonEmpty(delta.reasoning, delta.reasoning_content);
        if (rawReasoning.length > 0) {
          // Promote even when the filtered text is empty (tag-only fragment):
          // leaving the raw reasoning field in place would leak the partial tag.
          delta.content = reasoningFilter.push(rawReasoning);
          delete delta.reasoning;
          delete delta.reasoning_content;
          modified = true;
        }
      }
    }
    return modified ? `data: ${JSON.stringify(parsed.data)}` : line;
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
      if (buffer.length > 0) controller.enqueue(encoder.encode(handleLine(buffer)));
      // Stream ended without [DONE]: still surface a false-partial tail as its
      // own blank-line-closed SSE event.
      const tail = tailChunkLine();
      if (tail !== null) controller.enqueue(encoder.encode(`${tail}\n\n`));
    },
  });
}
