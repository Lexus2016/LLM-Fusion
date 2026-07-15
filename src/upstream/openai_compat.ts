import type { ChatCompletionResult, FetchFn, UpstreamClient } from "../types";
import { tapStreamUsage, usageFromBody } from "../usage";
import { NotImplementedError, UpstreamNetworkError, UpstreamTimeoutError } from "../errors";

/**
 * Generic OpenAI-compatible upstream client.
 *
 * Speaks the OpenAI `/v1/chat/completions` path (streaming + non-streaming) with
 * a configurable auth scheme and optional extra headers. This is the shared base
 * for every provider whose chat API is a drop-in OpenAI endpoint — OpenRouter,
 * DeepInfra, Together, Novita, Nebius, Groq, Cerebras, DeepSeek, Mistral, … —
 * which differ only in `base_url`, model-id format (handled by the connector's
 * `model_map`, not here), and optional headers.
 *
 * The native discovery (`/api/show`) and native chat (`/api/chat`) paths are
 * NOT part of the OpenAI-compat surface, so they throw `NotImplementedError`
 * here; the Ollama subclass re-adds them.
 */

/** How the credential is presented. Ollama/OpenRouter/most = `Bearer`; fal.ai = `Key`. */
export type AuthScheme = "Bearer" | "Key";

export interface OpenAiCompatClientOptions {
  baseUrl: string;
  apiKey?: string;
  /** Injectable fetch (default = global fetch). Tests pass a mock — no network, no key. */
  fetchFn?: FetchFn;
  /** Per-call timeout in ms. Must stay below the ~182 s Ollama Cloud ceiling. */
  timeoutMs?: number;
  /** Credential presentation scheme. Default `Bearer`. */
  authScheme?: AuthScheme;
  /** Extra request headers merged into every call (e.g. OpenRouter ranking headers). */
  extraHeaders?: Record<string, string>;
}

export class OpenAiCompatClient implements UpstreamClient {
  protected baseUrl: string;
  protected apiKey: string | undefined;
  protected readonly fetchFn: FetchFn;
  protected timeoutMs: number;
  protected authScheme: AuthScheme;
  protected extraHeaders: Record<string, string>;

  constructor(opts: OpenAiCompatClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.fetchFn = opts.fetchFn ?? globalThis.fetch;
    this.timeoutMs = opts.timeoutMs ?? 170_000;
    this.authScheme = opts.authScheme ?? "Bearer";
    this.extraHeaders = opts.extraHeaders ?? {};
  }

  /**
   * Whether this client can serve native `/api/show` capability discovery. Only
   * the Ollama subclass can; the pooled client uses this to route discovery to an
   * Ollama connector rather than degrading unnecessarily.
   */
  get supportsNativeShow(): boolean {
    return false;
  }

  updateConfig(opts: {
    baseUrl: string;
    apiKey?: string;
    timeoutMs?: number;
    extraHeaders?: Record<string, string>;
  }): void {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    if (opts.timeoutMs !== undefined) this.timeoutMs = opts.timeoutMs;
    if (opts.extraHeaders !== undefined) this.extraHeaders = opts.extraHeaders;
  }

  protected headers(): Record<string, string> {
    // extraHeaders first so `content-type` always wins; authorization set last.
    const h: Record<string, string> = { ...this.extraHeaders, "content-type": "application/json" };
    if (this.apiKey) h.authorization = `${this.authScheme} ${this.apiKey}`;
    return h;
  }

  protected async doFetch(
    url: string,
    init: RequestInit,
    opts: { phaseTimeoutOnly?: boolean } = {},
  ): Promise<Response> {
    // Per-call timeout. Two shapes:
    //  - default (non-stream): the timeout covers the whole request incl. body
    //    read, via `AbortSignal.timeout` (unref'd, does not keep the event loop
    //    alive). Combined with any caller signal so a stage timeout/shutdown can
    //    cancel the in-flight request and free its concurrency slot promptly.
    //  - phaseTimeoutOnly (stream): the timeout is a CONNECTION/first-response
    //    timeout only. It is cleared the instant `fetch()` resolves (the headers
    //    are back), so a slow-but-progressing stream is NOT hard-cut mid-delivery
    //    — the model already produced its answer and is delivering it. The stream
    //    is still cancellable by the caller's signal (client disconnect / stage
    //    abort). This turns the hard request_timeout_s into a dynamic one: it
    //    bounds time-to-first-response, not total response time.
    if (opts.phaseTimeoutOnly) {
      const phaseAbort = new AbortController();
      const handle = setTimeout(() => phaseAbort.abort(), this.timeoutMs);
      handle.unref?.();
      const signal = init.signal ? AbortSignal.any([init.signal, phaseAbort.signal]) : phaseAbort.signal;
      try {
        const res = await this.fetchFn(url, { ...init, signal });
        // Headers received — stop the hard timeout so the streaming body is not
        // truncated once the model starts delivering.
        clearTimeout(handle);
        return res;
      } catch (err) {
        clearTimeout(handle);
        if (phaseAbort.signal.aborted) {
          throw new UpstreamTimeoutError(`upstream request to ${url} timed out after ${this.timeoutMs}ms`);
        }
        if (init.signal?.aborted) {
          throw new UpstreamTimeoutError(`upstream request to ${url} was cancelled by the caller`);
        }
        const message = err instanceof Error ? err.message : "upstream request failed";
        throw new UpstreamNetworkError(`upstream request to ${url} failed: ${message}`);
      }
    }
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    try {
      return await this.fetchFn(url, { ...init, signal });
    } catch (err) {
      if (timeout.aborted) {
        throw new UpstreamTimeoutError(`upstream request to ${url} timed out after ${this.timeoutMs}ms`);
      }
      if (init.signal?.aborted) {
        throw new UpstreamTimeoutError(`upstream request to ${url} was cancelled by the caller`);
      }
      const message = err instanceof Error ? err.message : "upstream request failed";
      throw new UpstreamNetworkError(`upstream request to ${url} failed: ${message}`);
    }
  }

  async chatCompletions(
    body: Record<string, unknown>,
    opts: { stream: boolean; signal?: AbortSignal },
  ): Promise<ChatCompletionResult> {
    const url = `${this.baseUrl}/v1/chat/completions`;
    const payload = withIncludeUsage({ ...body, stream: opts.stream }, opts.stream);
    const res = await this.doFetch(
      url,
      {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(payload),
        signal: opts.signal,
      },
      // Stream: hard timeout is connection/first-response only — once the model
      // starts delivering, do not cut it. See doFetch.
      { phaseTimeoutOnly: opts.stream },
    );
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
    const retry = parseRetryAfterMs(res);
    return {
      kind: "json",
      status: res.status,
      data,
      usage: usageFromBody(data),
      ...(retry !== undefined ? { retryAfterMs: retry } : {}),
    };
  }

  /**
   * List the provider's available model ids via the OpenAI-compat `GET /v1/models`
   * (shape `{ data: [{ id }] }`). Used by the panel's config editor to offer real
   * model choices instead of free-typed ids. Throws on a non-OK response.
   */
  async listModels(opts: { signal?: AbortSignal } = {}): Promise<string[]> {
    const url = `${this.baseUrl}/v1/models`;
    const res = await this.doFetch(url, { method: "GET", headers: this.headers(), signal: opts.signal });
    if (!res.ok) {
      throw new UpstreamNetworkError(`/v1/models failed for ${this.baseUrl} (status ${res.status})`);
    }
    return parseModelList(await readBody(res));
  }

  async show(_model: string, _opts: { signal?: AbortSignal } = {}): Promise<unknown> {
    throw new NotImplementedError(
      "native /api/show capability discovery is not supported by the openai-compat provider",
    );
  }

  async chatNative(
    _body: Record<string, unknown>,
    _opts: { stream: boolean; signal?: AbortSignal },
  ): Promise<ChatCompletionResult> {
    throw new NotImplementedError(
      "native /api/chat is not supported by the openai-compat provider (use api_mode openai/auto)",
    );
  }
}

/**
 * Add `stream_options:{include_usage:true}` for streaming requests so the upstream
 * emits a final SSE chunk carrying `usage`. Merges with any caller-supplied
 * `stream_options`. A no-op for non-stream requests.
 */
export function withIncludeUsage(
  payload: Record<string, unknown>,
  stream: boolean,
): Record<string, unknown> {
  if (!stream) return payload;
  const existing = payload.stream_options;
  const base = typeof existing === "object" && existing !== null ? existing : {};
  return { ...payload, stream_options: { ...base, include_usage: true } };
}

/**
 * Extract model ids from an OpenAI-compat `GET /v1/models` body (`{ data: [{ id }] }`).
 * Tolerant of the two common id fields (`id`, some servers use `name`); returns a
 * de-duplicated, sorted list of non-empty string ids. Never throws on a malformed
 * body — an unrecognised shape yields `[]`.
 */
export function parseModelList(data: unknown): string[] {
  const rows = extractModelRows(data);
  const ids = new Set<string>();
  for (const row of rows) {
    if (typeof row === "string") {
      if (row.length > 0) ids.add(row);
      continue;
    }
    if (row && typeof row === "object") {
      const rec = row as Record<string, unknown>;
      const id = rec.id ?? rec.name ?? rec.model;
      if (typeof id === "string" && id.length > 0) ids.add(id);
    }
  }
  return [...ids].sort((a, b) => a.localeCompare(b));
}

function extractModelRows(data: unknown): unknown[] {
  if (Array.isArray(data)) return data;
  if (data && typeof data === "object") {
    const rec = data as Record<string, unknown>;
    // OpenAI-compat: { data: [...] }; Ollama /api/tags: { models: [...] }.
    if (Array.isArray(rec.data)) return rec.data;
    if (Array.isArray(rec.models)) return rec.models;
  }
  return [];
}

/** Read a response body as JSON when possible, falling back to text or null. */
export async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (text.length === 0) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/**
 * Parse an HTTP `Retry-After` header into milliseconds. Supports both forms:
 * a delta in seconds (`Retry-After: 30`) and an HTTP date. Returns undefined
 * when absent or unparseable. Never negative.
 */
export function parseRetryAfterMs(res: Response): number | undefined {
  const raw = res.headers.get("retry-after");
  if (!raw) return undefined;
  const secs = Number(raw.trim());
  if (Number.isFinite(secs)) return Math.max(0, Math.round(secs * 1000));
  const when = Date.parse(raw);
  if (Number.isNaN(when)) return undefined;
  return Math.max(0, when - Date.now());
}

