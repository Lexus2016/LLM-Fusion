import { z } from "zod";
import type { Logger } from "pino";
import type { Config, ModelConfig } from "./config";
import type { Resilience } from "./concurrency";

/**
 * Shared types: OpenAI-compatible request/response/stream shapes, the upstream
 * client interface, the capability record, and the strategy interface.
 */

// --- Capability discovery -------------------------------------------------

/** Discovered (or fallen-back) capabilities for a real upstream model. */
export interface Capability {
  vision: boolean;
  tools: boolean;
  context: number | null;
}

export type CapabilitySource = "discovered" | "override" | "default";

export interface DiscoveryResult {
  capability: Capability;
  /**
   * Where the capability came from. `default` means we are guessing (discovery
   * missed and no override existed) — callers that must not guess (e.g.
   * `/v1/models`) omit the fields in that case.
   */
  source: CapabilitySource;
}

export interface CapabilityProvider {
  discover(model: string): Promise<DiscoveryResult>;
  clear(): void;
}

// --- Upstream client ------------------------------------------------------

export type FetchFn = typeof globalThis.fetch;

/** Result of an upstream chat-completions call. */
export type ChatCompletionResult =
  | { kind: "json"; status: number; data: unknown }
  | {
      kind: "stream";
      status: number;
      body: ReadableStream<Uint8Array> | null;
      contentType: string | null;
    };

export interface UpstreamClient {
  /**
   * OpenAI-compat path: `POST {baseUrl}/v1/chat/completions`.
   * Non-stream -> parsed JSON; stream -> raw body stream.
   */
  chatCompletions(body: Record<string, unknown>, opts: { stream: boolean }): Promise<ChatCompletionResult>;
  /** Native discovery path: `POST {baseUrl}/api/show {model}` -> raw JSON. */
  show(model: string): Promise<unknown>;
  /**
   * Native chat path (`/api/chat`, NDJSON transport). Used for vision when
   * `api_mode === "native"`. The non-stream path is implemented; native
   * streaming is deferred (throws `NativeStreamingNotImplementedError`).
   */
  chatNative(body: Record<string, unknown>, opts: { stream: boolean }): Promise<ChatCompletionResult>;
}

// --- OpenAI-compatible request shapes -------------------------------------

const ContentPartSchema = z.object({ type: z.string() }).passthrough();

const MessageSchema = z
  .object({
    role: z.string(),
    content: z.union([z.string(), z.array(ContentPartSchema), z.null()]).optional(),
  })
  .passthrough();

/**
 * Incoming chat-completions request. Only the fields the proxy needs to route
 * are typed; everything else is preserved (`.passthrough()`) and forwarded
 * verbatim to the upstream.
 */
export const ChatCompletionRequestSchema = z
  .object({
    model: z.string().min(1),
    stream: z.boolean().optional(),
    messages: z.array(MessageSchema).optional(),
    tools: z.array(z.unknown()).optional(),
    tool_choice: z.unknown().optional(),
  })
  .passthrough();

export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequestSchema>;
export type ChatMessage = z.infer<typeof MessageSchema>;
export type ContentPart = z.infer<typeof ContentPartSchema>;

// --- OpenAI-compatible response shapes (descriptive) ----------------------

export interface ChatCompletionChoice {
  index: number;
  message: { role: string; content: string | null };
  finish_reason: string | null;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: { role?: string; content?: string };
  finish_reason: string | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
}

// --- Strategy execution context ------------------------------------------

export interface RequestContext {
  request: ChatCompletionRequest;
  config: Config;
  client: UpstreamClient;
  capabilities: CapabilityProvider;
  logger: Logger;
  /**
   * Shared resilience primitives (concurrency limiter + per-model circuit
   * breaker + retry/backoff policy). Optional so unit tests can build a bare
   * context; the server always supplies it.
   */
  resilience?: Resilience;
}

export interface StrategyContext extends RequestContext {
  modelConfig: ModelConfig;
}

export interface Strategy {
  execute(ctx: StrategyContext): Promise<Response>;
}
