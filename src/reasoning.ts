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

/** Strips <think> and </think> tags and trims the result. */
export function stripThinkingTags(text: string): string {
  return text
    .replace(/<think>/gi, "")
    .replace(/<\/think>/gi, "");
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

  const handleLine = (line: string): string => {
    if (!line.startsWith("data:")) return line; // blank separators, comments, etc.
    const payload = line.slice("data:".length).trim();
    if (payload.length === 0 || payload === "[DONE]") return line;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return line;
    }
    const parsed = StreamChunkSchema.safeParse(chunk);
    if (!parsed.success || !parsed.data.choices) return line;
    let modified = false;
    for (const choice of parsed.data.choices) {
      const delta = choice.delta;
      if (!delta) continue;
      const content = typeof delta.content === "string" ? delta.content : "";
      if (content.length > 0) {
        realContentSeen = true; // real content — leave this and every later event alone
        const cleaned = stripThinkingTags(content);
        if (cleaned !== content) {
          delta.content = cleaned;
          modified = true;
        }
      } else if (!realContentSeen) {
        const reasoning = stripThinkingTags(firstNonEmpty(delta.reasoning, delta.reasoning_content));
        if (reasoning.length > 0) {
          delta.content = reasoning;
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
    },
  });
}
