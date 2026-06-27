import type { ChatCompletionResult, FetchFn, UpstreamClient } from "../types";
import { tapStreamUsage, usageFromBody } from "../usage";
import {
  NativeStreamingNotImplementedError,
  UpstreamNetworkError,
  UpstreamTimeoutError,
} from "../errors";

export interface OllamaClientOptions {
  baseUrl: string;
  apiKey?: string;
  /** Injectable fetch (default = global fetch). Tests pass a mock — no network, no key. */
  fetchFn?: FetchFn;
  /** Per-call timeout in ms. Must stay below the ~182 s Ollama Cloud ceiling. */
  timeoutMs?: number;
}

/**
 * Ollama Cloud client.
 *
 * Implements the OpenAI-compat path (`/v1/chat/completions`), the native
 * discovery path (`/api/show`), and the native chat path (`/api/chat`). The
 * native chat path serves vision requests when `api_mode === "native"`; its
 * non-stream path is wired, while native NDJSON streaming is deferred (a typed
 * `NativeStreamingNotImplementedError` is thrown).
 */
export class OllamaClient implements UpstreamClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchFn: FetchFn;
  private readonly timeoutMs: number;

  constructor(opts: OllamaClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.fetchFn = opts.fetchFn ?? globalThis.fetch;
    this.timeoutMs = opts.timeoutMs ?? 170_000;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "content-type": "application/json" };
    if (this.apiKey) h.authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  private async doFetch(url: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      return await this.fetchFn(url, { ...init, signal: controller.signal });
    } catch (err) {
      if (controller.signal.aborted) {
        throw new UpstreamTimeoutError(`upstream request to ${url} timed out after ${this.timeoutMs}ms`);
      }
      const message = err instanceof Error ? err.message : "upstream request failed";
      throw new UpstreamNetworkError(`upstream request to ${url} failed: ${message}`);
    } finally {
      clearTimeout(timer);
    }
  }

  async chatCompletions(
    body: Record<string, unknown>,
    opts: { stream: boolean },
  ): Promise<ChatCompletionResult> {
    const url = `${this.baseUrl}/v1/chat/completions`;
    const payload = withIncludeUsage({ ...body, stream: opts.stream }, opts.stream);
    const res = await this.doFetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
    });
    // Stream only when the upstream actually succeeded; an error before the
    // first byte is surfaced as a JSON body with the upstream status.
    if (opts.stream && res.ok) {
      const { stream, usage } = tapStreamUsage(res.body);
      return {
        kind: "stream",
        status: res.status,
        body: stream,
        contentType: res.headers.get("content-type"),
        usage,
      };
    }
    const data = await readBody(res);
    return { kind: "json", status: res.status, data, usage: usageFromBody(data) };
  }

  async show(model: string): Promise<unknown> {
    const url = `${this.baseUrl}/api/show`;
    const res = await this.doFetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ model }),
    });
    if (!res.ok) {
      const data = await readBody(res);
      const detail = typeof data === "string" ? data : JSON.stringify(data);
      throw new UpstreamNetworkError(`/api/show failed for '${model}' (status ${res.status}): ${detail}`);
    }
    return readBody(res);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async chatNative(
    body: Record<string, unknown>,
    opts: { stream: boolean },
  ): Promise<ChatCompletionResult> {
    if (opts.stream) {
      // Native NDJSON streaming is intentionally deferred — callers that hit
      // this (api_mode "native" + images + stream:true) get a clean 501 rather
      // than a half-wired stream. The OpenAI-compat streaming path is unaffected.
      throw new NativeStreamingNotImplementedError(
        "native /api/chat streaming (NDJSON) is not yet wired; use api_mode openai/auto for streaming image requests",
      );
    }
    const url = `${this.baseUrl}/api/chat`;
    const payload = { ...body, stream: false };
    const res = await this.doFetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
    });
    const data = await readBody(res);
    return { kind: "json", status: res.status, data, usage: usageFromBody(data) };
  }
}

/**
 * Add `stream_options:{include_usage:true}` for streaming requests so Ollama
 * emits a final SSE chunk carrying `usage`. Merges with any caller-supplied
 * `stream_options`. A no-op for non-stream requests.
 */
function withIncludeUsage(
  payload: Record<string, unknown>,
  stream: boolean,
): Record<string, unknown> {
  if (!stream) return payload;
  const existing = payload.stream_options;
  const base = typeof existing === "object" && existing !== null ? existing : {};
  return { ...payload, stream_options: { ...base, include_usage: true } };
}

/** Read a response body as JSON when possible, falling back to text or null. */
async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (text.length === 0) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
