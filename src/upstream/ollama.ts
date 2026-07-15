import type { ChatCompletionResult, FetchFn } from "../types";
import { usageFromBody } from "../usage";
import { NativeStreamingNotImplementedError, UpstreamNetworkError } from "../errors";
import { OpenAiCompatClient, readBody } from "./openai_compat";

export interface OllamaClientOptions {
  baseUrl: string;
  apiKey?: string;
  /** Injectable fetch (default = global fetch). Tests pass a mock — no network, no key. */
  fetchFn?: FetchFn;
  /** Per-call timeout in ms. Must stay below the ~182 s Ollama Cloud ceiling. */
  timeoutMs?: number;
  /** Extra request headers merged into every call (rarely needed for Ollama). */
  extraHeaders?: Record<string, string>;
}

/**
 * Ollama Cloud client — an OpenAI-compatible client (inherited) that additionally
 * implements the native discovery path (`/api/show`) and native chat path
 * (`/api/chat`). The native chat path serves vision requests when
 * `api_mode === "native"`; its non-stream path is wired, while native NDJSON
 * streaming is deferred (a typed `NativeStreamingNotImplementedError` is thrown).
 */
export class OllamaClient extends OpenAiCompatClient {
  constructor(opts: OllamaClientOptions) {
    super({ ...opts, authScheme: "Bearer" });
  }

  /** Ollama is the one provider that can answer native `/api/show` discovery. */
  override get supportsNativeShow(): boolean {
    return true;
  }

  override async show(model: string, opts: { signal?: AbortSignal } = {}): Promise<unknown> {
    const url = `${this.baseUrl}/api/show`;
    const res = await this.doFetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ model }),
      signal: opts.signal,
    });
    if (!res.ok) {
      const data = await readBody(res);
      const detail = typeof data === "string" ? data : JSON.stringify(data);
      throw new UpstreamNetworkError(`/api/show failed for '${model}' (status ${res.status}): ${detail}`);
    }
    return readBody(res);
  }

  override async chatNative(
    body: Record<string, unknown>,
    opts: { stream: boolean; signal?: AbortSignal },
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
      signal: opts.signal,
    });
    const data = await readBody(res);
    return { kind: "json", status: res.status, data, usage: usageFromBody(data) };
  }
}
