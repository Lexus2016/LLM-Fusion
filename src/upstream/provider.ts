import type { FetchFn } from "../types";
import type { ConnectorClient } from "../connectors/registry";
import { OllamaClient } from "./ollama";
import { OpenAiCompatClient } from "./openai_compat";

/**
 * Provider factory — maps a connector's `provider` string to a concrete
 * `UpstreamClient`. Adding a provider is one case here; every OpenAI-compatible
 * vendor (OpenRouter, DeepInfra, Together, Novita, Nebius, Groq, DeepSeek, …)
 * already works through `openai-compat` with no code change.
 */

export type ProviderKind = "ollama" | "openai-compat";

export interface CreateClientOptions {
  provider: ProviderKind;
  baseUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  extraHeaders?: Record<string, string>;
  fetchFn?: FetchFn;
}

export function createUpstreamClient(o: CreateClientOptions): ConnectorClient {
  if (o.provider === "ollama") {
    return new OllamaClient({
      baseUrl: o.baseUrl,
      apiKey: o.apiKey,
      timeoutMs: o.timeoutMs,
      extraHeaders: o.extraHeaders,
      fetchFn: o.fetchFn,
    });
  }
  return new OpenAiCompatClient({
    baseUrl: o.baseUrl,
    apiKey: o.apiKey,
    timeoutMs: o.timeoutMs,
    extraHeaders: o.extraHeaders,
    authScheme: "Bearer",
    fetchFn: o.fetchFn,
  });
}
