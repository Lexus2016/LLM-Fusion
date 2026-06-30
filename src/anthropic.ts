import { Hono } from "hono";
import { z } from "zod";
import { randomUUID } from "node:crypto";
import type { Logger } from "pino";
import type { Config } from "./config";
import type {
  CapabilityProvider,
  ChatCompletionRequest,
  ChatMessage,
  ContentPart,
  RequestUsage,
  UpstreamClient,
  Usage,
} from "./types";
import type { Resilience } from "./concurrency";
import { dispatch } from "./router";
import { UsageAccumulator, usageHeaderValue, toOpenAiUsage } from "./usage";
import { createAuthMiddleware } from "./auth";
import { BadRequestError, FusionError, toAnthropicErrorResponse } from "./errors";
import { stripThinkingTags } from "./reasoning";
import { stripHopByHopHeaders } from "./headers";

/**
 * Anthropic Messages API compatibility layer.
 *
 * Exposes the standard Anthropic path `POST /v1/messages` on the same base URL
 * as the OpenAI-compatible `/v1/chat/completions`. Claude Code (and any other
 * Anthropic SDK client) can therefore point `ANTHROPIC_BASE_URL` at the proxy
 * without a special prefix.
 *
 * The implementation is intentionally best-effort / agent-loop focused: it
 * translates between Anthropic content blocks (text / image / tool_use /
 * tool_result) and the OpenAI chat-completions shape, then reuses the existing
 * strategy dispatcher so `single`, `fusion`, `smart`, etc. work unchanged.
 */

// --- Zod schemas ----------------------------------------------------------

const TextBlockSchema = z
  .object({ type: z.literal("text"), text: z.string() })
  .passthrough();

const ImageBlockSchema = z
  .object({
    type: z.literal("image"),
    source: z
      .object({
        type: z.string(),
        media_type: z.string().optional(),
        data: z.string().optional(),
        url: z.string().optional(),
      })
      .passthrough(),
  })
  .passthrough();

const ThinkingBlockSchema = z
  .object({
    type: z.literal("thinking"),
    thinking: z.string().optional().nullable(),
  })
  .passthrough();

const RedactedThinkingBlockSchema = z
  .object({
    type: z.literal("redacted_thinking"),
    data: z.string().optional().nullable(),
  })
  .passthrough();

const ToolUseBlockSchema = z
  .object({
    type: z.literal("tool_use"),
    id: z.string(),
    name: z.string(),
    input: z.union([z.record(z.string(), z.unknown()), z.string(), z.null()]).optional().default({}),
  })
  .passthrough();

// Catch-all for content blocks the proxy does not natively translate but which
// Claude Code / the Anthropic API can emit in long agent loops (server_tool_use,
// web_search_tool_result, code_execution_tool_*, document, container_upload, …).
// The schema MUST accept them so the request is not rejected wholesale with 400
// "invalid Anthropic messages request (messages.N.content: Invalid input)" —
// which breaks the agent loop as soon as one such block appears. The translators
// re-parse each block against the known schemas via safeParse, so unknown blocks
// are silently ignored (best-effort), and adding this catch-all here is safe.
const UnknownBlockSchema = z.object({ type: z.string() }).passthrough();

const ToolResultBlockSchema = z
  .object({
    type: z.literal("tool_result"),
    tool_use_id: z.string(),
    // Allow null and image arrays because the Anthropic SDK / Claude Code can
    // emit them in real agent loops (e.g. a tool that returns nothing, or a
    // vision tool that returns a screenshot). They are translated best-effort
    // into the OpenAI chat-completions shape.
    content: z
      .union([
        z.string(),
        z.null(),
        z.array(
          z.union([
            TextBlockSchema,
            ImageBlockSchema,
            ThinkingBlockSchema,
            RedactedThinkingBlockSchema,
            UnknownBlockSchema,
          ]),
        ),
      ])
      .default(""),
    is_error: z.boolean().optional(),
  })
  .passthrough();

const ContentBlockSchema = z.union([
  TextBlockSchema,
  ImageBlockSchema,
  ToolUseBlockSchema,
  ToolResultBlockSchema,
  ThinkingBlockSchema,
  RedactedThinkingBlockSchema,
  UnknownBlockSchema,
]);

const MessageSchema = z
  .object({
    role: z.enum(["user", "assistant", "system"]),
    // null content is accepted because Claude Code occasionally emits it for
    // assistant/tool messages; it is normalised to an empty string upstream.
    content: z.union([z.string(), z.null(), ContentBlockSchema, z.array(ContentBlockSchema)]),
  })
  .passthrough();

const ToolSchema = z
  .object({
    name: z.string(),
    description: z.string().optional(),
    input_schema: z.record(z.string(), z.unknown()).default({}),
  })
  .passthrough();

const ToolChoiceSchema = z.union([
  z.enum(["auto", "any"]),
  z.object({ type: z.enum(["auto", "any", "tool"]), name: z.string().optional() }).passthrough(),
]);

export const AnthropicRequestSchema = z
  .object({
    model: z.string().min(1),
    messages: z.array(MessageSchema),
    system: z.union([z.string(), z.array(TextBlockSchema)]).optional(),
    tools: z.array(ToolSchema).optional(),
    tool_choice: ToolChoiceSchema.optional(),
    max_tokens: z.number().int().positive().optional(),
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    top_k: z.number().int().optional(),
    stream: z.boolean().optional(),
    metadata: z.unknown().optional(),
    thinking: z.unknown().optional(),
  })
  .passthrough();

export type AnthropicRequest = z.infer<typeof AnthropicRequestSchema>;
export type AnthropicContentBlock = z.infer<typeof ContentBlockSchema>;
export type AnthropicMessage = z.infer<typeof MessageSchema>;

// --- Dependencies ---------------------------------------------------------

export interface AnthropicDeps {
  getConfig: () => Config;
  client: UpstreamClient;
  capabilities: CapabilityProvider;
  getAuthToken: () => string | undefined;
  logger: Logger;
  resilience?: Resilience;
}

// --- Request translation: Anthropic -> OpenAI -----------------------------

export function anthropicToOpenAiRequest(req: AnthropicRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = [];

  if (req.system) {
    const text =
      typeof req.system === "string"
        ? req.system
        : req.system.map((b) => b.text).join("\n");
    messages.push({ role: "system", content: text });
  }

  for (const m of req.messages) {
    if (m.role === "user") {
      messages.push(...anthropicUserContentToOpenAi(m.content));
    } else if (m.role === "system") {
      messages.push(anthropicSystemContentToOpenAi(m.content));
    } else {
      messages.push(anthropicAssistantContentToOpenAi(m.content));
    }
  }

  const openAi: ChatCompletionRequest = {
    model: req.model,
    messages,
    stream: req.stream,
  };

  if (req.tools && req.tools.length > 0) {
    openAi.tools = req.tools.map((t) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      },
    }));
  }

  if (req.tool_choice != null) {
    openAi.tool_choice = anthropicToolChoiceToOpenAi(req.tool_choice);
  }

  if (req.max_tokens != null) openAi.max_tokens = req.max_tokens;
  if (req.temperature != null) openAi.temperature = req.temperature;
  if (req.top_p != null) openAi.top_p = req.top_p;

  return openAi;
}

function anthropicUserContentToOpenAi(
  content: string | AnthropicContentBlock | AnthropicContentBlock[] | null,
): ChatMessage[] {
  if (content == null) {
    return [{ role: "user", content: "" }];
  }
  if (typeof content === "string") {
    return [{ role: "user", content }];
  }

  const result: ChatMessage[] = [];
  let currentUser: { role: "user"; content: ContentPart[] } | null = null;

  const flushUser = (): void => {
    if (!currentUser) return;
    if (currentUser.content.length === 1 && typeof currentUser.content[0] === "string") {
      result.push({ role: "user", content: currentUser.content[0] });
    } else {
      result.push({ role: "user", content: currentUser.content });
    }
    currentUser = null;
  };

  const blocks = Array.isArray(content) ? content : [content];
  for (const raw of blocks) {
    // Re-parse each block against the KNOWN schemas rather than narrowing on
    // block.type: ContentBlockSchema also accepts unknown block types
    // (server_tool_use, web_search_tool_result, document, …) so the request is
    // not rejected, but those blocks carry no type-safe fields — safeParse lets
    // us handle the known ones and silently ignore the rest.
    const tb = TextBlockSchema.safeParse(raw);
    if (tb.success) {
      if (!currentUser) currentUser = { role: "user", content: [] };
      currentUser.content.push({ type: "text", text: tb.data.text });
      continue;
    }
    const ib = ImageBlockSchema.safeParse(raw);
    if (ib.success) {
      if (!currentUser) currentUser = { role: "user", content: [] };
      const url = anthropicImageUrl(ib.data.source);
      currentUser.content.push({ type: "image_url", image_url: { url } });
      continue;
    }
    const tr = ToolResultBlockSchema.safeParse(raw);
    if (tr.success) {
      flushUser();
      const block = tr.data;
      const toolContent =
        block.content == null
          ? ""
          : typeof block.content === "string"
            ? block.content
            : block.content
                .map((b) => {
                  const itb = TextBlockSchema.safeParse(b);
                  if (itb.success) return itb.data.text;
                  const iib = ImageBlockSchema.safeParse(b);
                  if (iib.success) return anthropicImageUrl(iib.data.source);
                  const thb = ThinkingBlockSchema.safeParse(b);
                  if (thb.success) return thb.data.thinking || "";
                  return "";
                })
                .filter(Boolean)
                .join("\n");
      result.push({
        role: "tool",
        content: toolContent,
        tool_call_id: block.tool_use_id,
      });
      continue;
    }
    // Unknown block type — ignore (best-effort, keeps the agent loop alive).
  }
  flushUser();
  return result;
}

function anthropicSystemContentToOpenAi(
  content: string | AnthropicContentBlock | AnthropicContentBlock[] | null,
): ChatMessage {
  if (content == null) {
    return { role: "system", content: "" };
  }
  if (typeof content === "string") {
    return { role: "system", content };
  }
  const blocks = Array.isArray(content) ? content : [content];
  let text = "";
  for (const raw of blocks) {
    const tb = TextBlockSchema.safeParse(raw);
    if (tb.success) {
      text += (text.length > 0 ? "\n" : "") + tb.data.text;
    }
  }
  return { role: "system", content: text };
}

function anthropicAssistantContentToOpenAi(
  content: string | AnthropicContentBlock | AnthropicContentBlock[] | null,
): ChatMessage {
  if (content == null) {
    return { role: "assistant", content: "" };
  }
  if (typeof content === "string") {
    return { role: "assistant", content };
  }

  const textParts: string[] = [];
  const toolCalls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }> = [];
  const blocks = Array.isArray(content) ? content : [content];

  for (const raw of blocks) {
    // Re-parse against the KNOWN schemas (see anthropicUserContentToOpenAi for
    // why): thinking/redacted_thinking and any unknown block type are ignored.
    const tb = TextBlockSchema.safeParse(raw);
    if (tb.success) {
      textParts.push(tb.data.text);
      continue;
    }
    const tub = ToolUseBlockSchema.safeParse(raw);
    if (tub.success) {
      const block = tub.data;
      let args = "{}";
      if (block.input != null) {
        if (typeof block.input === "string") {
          args = block.input;
        } else {
          args = JSON.stringify(block.input);
        }
      }
      toolCalls.push({
        id: block.id,
        type: "function",
        function: {
          name: block.name,
          arguments: args,
        },
      });
      continue;
    }
  }

  const message: ChatMessage = { role: "assistant" };
  if (textParts.length > 0) message.content = textParts.join("\n");
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  if (message.content == null && message.tool_calls == null) message.content = "";
  return message;
}

function anthropicImageUrl(source: { type?: string; media_type?: string; data?: string; url?: string }): string {
  if (source.url) return source.url;
  if (source.data) {
    const mime = source.media_type ?? "application/octet-stream";
    return `data:${mime};base64,${source.data}`;
  }
  return "";
}

function anthropicToolChoiceToOpenAi(
  choice: z.infer<typeof ToolChoiceSchema>,
): unknown {
  if (choice === "auto") return "auto";
  if (choice === "any") return "required";
  if (typeof choice === "object") {
    if (choice.type === "auto") return "auto";
    if (choice.type === "any") return "required";
    if (choice.type === "tool" && choice.name) {
      return { type: "function", function: { name: choice.name } };
    }
  }
  return undefined;
}

// --- Response translation: OpenAI -> Anthropic (non-stream) -------------

export function openAiToAnthropicResponse(
  openAiData: unknown,
  model: string,
  usage: RequestUsage,
): unknown {
  const parsed = z
    .object({
      id: z.string().default(""),
      choices: z
        .array(
          z
            .object({
              message: z
                .object({
                  role: z.string().optional(),
                  content: z.union([z.string(), z.null()]).optional(),
                  tool_calls: z
                    .array(
                      z.object({
                        id: z.string(),
                        function: z.object({ name: z.string(), arguments: z.string() }).passthrough(),
                      }).passthrough(),
                    )
                    .optional(),
                })
                .passthrough(),
              finish_reason: z.string().nullable().optional(),
            })
            .passthrough(),
        )
        .default([]),
    })
    .passthrough()
    .safeParse(openAiData);

  if (!parsed.success) {
    return openAiData;
  }

  const choice = parsed.data.choices[0];
  const message = choice?.message;
  const finishReason = choice?.finish_reason ?? null;

  const contentBlocks: Array<Record<string, unknown>> = [];
  const textContent = typeof message?.content === "string" ? stripThinkingTags(message.content) : "";
  if (textContent.length > 0) {
    contentBlocks.push({ type: "text", text: textContent });
  }

  for (const tc of message?.tool_calls ?? []) {
    let input: Record<string, unknown> = {};
    try {
      input = JSON.parse(tc.function.arguments);
    } catch {
      input = {};
    }
    contentBlocks.push({
      type: "tool_use",
      id: tc.id,
      name: tc.function.name,
      input,
    });
  }

  return {
    id: parsed.data.id,
    type: "message",
    role: "assistant",
    model,
    content: contentBlocks,
    stop_reason: finishReasonToAnthropic(finishReason, contentBlocks),
    stop_sequence: null,
    usage: {
      input_tokens: usage.promptTokens,
      output_tokens: usage.completionTokens,
    },
  };
}

function finishReasonToAnthropic(
  reason: string | null | undefined,
  contentBlocks: Array<Record<string, unknown>>,
): string | null {
  if (reason === "stop") return "end_turn";
  if (reason === "length") return "max_tokens";
  if (reason === "tool_calls" || contentBlocks.some((b) => b.type === "tool_use")) {
    return "tool_use";
  }
  return null;
}

// --- Stream translation: OpenAI SSE -> Anthropic SSE ---------------------

interface AnthropicStreamOpts {
  /** Virtual model name to report back to the Anthropic client. */
  model: string;
  /** Correlation id for the response `id` field. */
  reqId: string;
  /** Unix timestamp for the response. */
  created: number;
  usage: UsageAccumulator;
  pricing?: Record<string, { input_per_mtok: number; output_per_mtok: number }>;
  logger: Logger;
}

export function anthropicStreamTransform(opts: AnthropicStreamOpts): TransformStream<Uint8Array, Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let buffer = "";

  interface ActiveBlock {
    index: number;
    kind: "text" | "tool";
    toolIndex?: number;
  }

  let nextBlockIndex = 0;
  let activeBlock: ActiveBlock | null = null;
  const toolBlockIndex = new Map<number, number>();
  let finishReason: string | null = null;
  let started = false;

  const emit = (
    controller: TransformStreamDefaultController<Uint8Array>,
    event: string,
    data: unknown,
  ): void => {
    controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
  };

  const stopActiveBlock = (controller: TransformStreamDefaultController<Uint8Array>): void => {
    if (!activeBlock) return;
    emit(controller, "content_block_stop", { type: "content_block_stop", index: activeBlock.index });
    activeBlock = null;
  };

  const startTextBlock = (controller: TransformStreamDefaultController<Uint8Array>): number => {
    stopActiveBlock(controller);
    const index = nextBlockIndex++;
    emit(controller, "content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "text", text: "" },
    });
    activeBlock = { index, kind: "text" };
    return index;
  };

  const startToolBlock = (
    controller: TransformStreamDefaultController<Uint8Array>,
    toolIndex: number,
    id: string,
    name: string,
  ): number => {
    stopActiveBlock(controller);
    const index = nextBlockIndex++;
    toolBlockIndex.set(toolIndex, index);
    emit(controller, "content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id, name, input: {} },
    });
    activeBlock = { index, kind: "tool", toolIndex };
    return index;
  };

  const handleChunk = (
    obj: unknown,
    controller: TransformStreamDefaultController<Uint8Array>,
  ): void => {
    const parsed = z
      .object({
        id: z.string().optional(),
        choices: z
          .array(
            z
              .object({
                delta: z
                  .object({
                    role: z.string().optional(),
                    content: z.union([z.string(), z.null()]).optional(),
                    reasoning: z.union([z.string(), z.null()]).optional(),
                    tool_calls: z
                      .array(
                        z
                          .object({
                            index: z.number().int().optional(),
                            id: z.string().optional(),
                            function: z
                              .object({ name: z.string().optional(), arguments: z.string().optional() })
                              .passthrough(),
                          })
                          .passthrough(),
                      )
                      .optional(),
                  })
                  .passthrough(),
                finish_reason: z.string().nullable().optional(),
              })
              .passthrough(),
          )
          .optional(),
      })
      .passthrough()
      .safeParse(obj);

    if (!parsed.success) return;

    const chunk = parsed.data;
    const choice = chunk.choices?.[0];
    if (!choice) return;

    if (choice.finish_reason != null) {
      finishReason = choice.finish_reason ?? null;
    }

    const delta = choice.delta;
    if (!delta) return;

    const rawText = delta.content ?? delta.reasoning;
    if (typeof rawText === "string" && rawText.length > 0) {
      const text = stripThinkingTags(rawText);
      if (text.length > 0) {
        if (!activeBlock || activeBlock.kind !== "text") {
          startTextBlock(controller);
        }
        emit(controller, "content_block_delta", {
          type: "content_block_delta",
          index: activeBlock!.index,
          delta: { type: "text_delta", text },
        });
      }
    }

    for (const tc of delta.tool_calls ?? []) {
      const idx = tc.index ?? 0;
      const hasStart = tc.id != null || tc.function?.name != null;
      let blockIdx = toolBlockIndex.get(idx);
      if (blockIdx == null && hasStart) {
        blockIdx = startToolBlock(controller, idx, tc.id ?? "", tc.function?.name ?? "");
      }
      if (blockIdx == null) continue;
      if (!activeBlock || activeBlock.kind !== "tool" || activeBlock.toolIndex !== idx) {
        stopActiveBlock(controller);
        activeBlock = { index: blockIdx, kind: "tool", toolIndex: idx };
      }
      const partial = tc.function?.arguments ?? "";
      emit(controller, "content_block_delta", {
        type: "content_block_delta",
        index: blockIdx,
        delta: { type: "input_json_delta", partial_json: partial },
      });
    }
  };

  const transformer: Transformer<Uint8Array, Uint8Array> & { cancel?(reason?: unknown): void } = {
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        const trimmed = line.trimStart();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice("data:".length).trim();
        if (payload === "[DONE]") continue;
        try {
          const obj = JSON.parse(payload);
          if (!started) {
            started = true;
            emit(controller, "message_start", {
              type: "message_start",
              message: {
                id: opts.reqId,
                type: "message",
                role: "assistant",
                model: opts.model,
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: 0, output_tokens: 0 },
              },
            });
          }
          handleChunk(obj, controller);
        } catch {
          /* ignore malformed SSE lines */
        }
      }
    },
    async flush(controller) {
      buffer += decoder.decode();
      if (buffer.length > 0) {
        const trimmed = buffer.trimStart();
        if (trimmed.startsWith("data:")) {
          const payload = trimmed.slice("data:".length).trim();
          if (payload !== "[DONE]") {
            try {
              const obj = JSON.parse(payload);
              if (!started) {
                started = true;
                emit(controller, "message_start", {
                  type: "message_start",
                  message: {
                    id: opts.reqId,
                    type: "message",
                    role: "assistant",
                    model: opts.model,
                    content: [],
                    stop_reason: null,
                    stop_sequence: null,
                    usage: { input_tokens: 0, output_tokens: 0 },
                  },
                });
              }
              handleChunk(obj, controller);
            } catch {
              /* ignore */
            }
          }
        }
      }

      stopActiveBlock(controller);
      const finalUsage = await opts.usage.finalize(opts.pricing);
      const stopReason = finishReasonToAnthropic(finishReason, []);
      emit(controller, "message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: {
          input_tokens: finalUsage.promptTokens,
          output_tokens: finalUsage.completionTokens,
        },
      });
      emit(controller, "message_stop", { type: "message_stop" });

      opts.logger.info(
        {
          req_id: opts.reqId,
          model: opts.model,
          upstream_calls: finalUsage.upstreamCalls,
          prompt_tokens: finalUsage.promptTokens,
          completion_tokens: finalUsage.completionTokens,
          total_tokens: finalUsage.totalTokens,
          cost_usd: finalUsage.costUsd,
        },
        "request usage",
      );
    },
    cancel() {
      // Best-effort: if the stream is cancelled before flush, still try to log.
      void opts.usage
        .finalize(opts.pricing)
        .then((u) =>
          opts.logger.info(
            {
              req_id: opts.reqId,
              model: opts.model,
              upstream_calls: u.upstreamCalls,
              prompt_tokens: u.promptTokens,
              completion_tokens: u.completionTokens,
              total_tokens: u.totalTokens,
              cost_usd: u.costUsd,
            },
            "request usage",
          ),
        )
        .catch(() => undefined);
    },
  };
  return new TransformStream<Uint8Array, Uint8Array>(transformer);
}

// --- Route wiring ---------------------------------------------------------

export function createAnthropicApp(deps: AnthropicDeps): Hono {
  const app = new Hono();
  const auth = createAuthMiddleware(deps.getAuthToken);

  app.post("/v1/messages", auth, async (c) => {
    const reqId = randomUUID();
    const startedAt = Date.now();
    const reqLogger = deps.logger.child({ req_id: reqId });

    let raw: unknown;
    try {
      raw = await c.req.json();
    } catch {
      return toAnthropicErrorResponse(new BadRequestError("request body must be valid JSON"));
    }

    const parsed = AnthropicRequestSchema.safeParse(raw);
    if (!parsed.success) {
      const first = parsed.error.issues[0];
      const detail = first ? `${first.path.join(".") || "<root>"}: ${first.message}` : "schema validation failed";
      // DIAG (root-cause investigation): surface the exact shape of the rejected
      // message so we can see WHICH content block type Claude Code emits that the
      // schema does not accept. No message text is logged — only structural types
      // and block type names, so no prompt/tool-result content leaks to the log.
      try {
        const isRecord = (v: unknown): v is Record<string, unknown> =>
          v !== null && typeof v === "object";
        const idx = first && typeof first.path[1] === "number" ? first.path[1] : -1;
        const blockTypes: string[] = [];
        let role: unknown;
        if (isRecord(raw) && Array.isArray(raw.messages) && idx >= 0) {
          const rejected = raw.messages[idx];
          if (isRecord(rejected)) {
            role = rejected.role;
            const c = rejected.content;
            const pushType = (b: unknown): void => {
              if (isRecord(b) && typeof b.type === "string") blockTypes.push(b.type);
            };
            if (Array.isArray(c)) for (const b of c) pushType(b);
            else pushType(c);
          }
        }
        reqLogger.warn(
          { idx, role, blockTypes, issuePath: first?.path, issueMsg: first?.message },
          "anthropic: request rejected by schema (diagnostic)",
        );
      } catch {
        // never let diagnostics mask the original error
      }
      return toAnthropicErrorResponse(new BadRequestError(`invalid Anthropic messages request (${detail})`));
    }

    const request = parsed.data;
    const model = request.model;
    const stream = request.stream === true;
    const config = deps.getConfig();
    const usage = new UsageAccumulator();

    try {
      const openAiRequest = anthropicToOpenAiRequest(request);
      const res = await dispatch({
        request: openAiRequest,
        config,
        client: deps.client,
        capabilities: deps.capabilities,
        logger: reqLogger,
        resilience: deps.resilience,
        usage,
        // Client abort signal: a disconnect (Esc in Claude Code) cancels in-flight
        // upstream work instead of running a whole fusion fan-out for a gone client.
        signal: c.req.raw.signal,
      });

      const headers = new Headers(res.headers);
      headers.set("x-fusion-usage", usageHeaderValue(usage.snapshot(config.pricing)));

      if (res.status >= 400) {
        reqLogger.info({ model, status: res.status, ms: Date.now() - startedAt, stream }, "Anthropic request rejected");
        return new Response(res.body, { status: res.status, headers });
      }

      if (stream && res.body) {
        const transform = anthropicStreamTransform({
          model,
          reqId,
          created: Math.floor(Date.now() / 1000),
          usage,
          pricing: config.pricing,
          logger: reqLogger,
        });
        // The SSE body is our transform stream, not the upstream body — drop
        // upstream length/encoding headers so they cannot disagree.
        stripHopByHopHeaders(headers);
        const sseHeaders: Record<string, string> = {
          "content-type": "text/event-stream",
          "cache-control": "no-cache",
          connection: "keep-alive",
        };
        for (const [k, v] of headers.entries()) {
          if (k.toLowerCase() !== "content-type") sseHeaders[k] = v;
        }

        void res.body.pipeTo(transform.writable).then(
          () => undefined,
          () => undefined,
        );
        reqLogger.info({ model, status: res.status, ms: Date.now() - startedAt, stream }, "request complete");
        return new Response(transform.readable, { status: res.status, headers: sseHeaders });
      }

      const openAiData = await res.json();
      const finalUsage = await usage.finalize(config.pricing);
      const anthropicBody = openAiToAnthropicResponse(openAiData, model, finalUsage);
      // The body is re-serialized below — drop upstream length/encoding headers.
      stripHopByHopHeaders(headers);
      headers.set("content-type", "application/json");
      reqLogger.info(
        { model, status: res.status, ms: Date.now() - startedAt, stream },
        "request complete",
      );
      return new Response(JSON.stringify(anthropicBody), { status: res.status, headers });
    } catch (err) {
      const status = err instanceof FusionError ? err.httpStatus : 500;
      const ms = Date.now() - startedAt;
      if (status >= 500) {
        reqLogger.error({ model, err: errMessage(err), status, ms }, "request failed");
      } else {
        reqLogger.info({ model, status, ms }, "request rejected");
      }
      return toAnthropicErrorResponse(err);
    }
  });

  return app;
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
