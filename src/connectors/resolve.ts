import type { Logger } from "pino";
import type { Config, ConnectorConfig } from "../config";
import type { FetchFn } from "../types";
import { createUpstreamClient } from "../upstream/provider";
import type { ConnectorClient, ResolvedConnector } from "./registry";

/**
 * Resolve the config's connector pool into `{ cfg, client }` entries ready for
 * the `ConnectorRegistry`. Reads the API key for each connector from its
 * `api_key_env` env var (never stores the value in `cfg`). When no explicit
 * `connectors:` list is configured, a single connector is synthesised from the
 * legacy `upstream.base_url` + `upstream.api_key_env` (backward compatible).
 */

export interface ResolveOptions {
  env: Record<string, string | undefined>;
  logger?: Logger;
  /** Injectable fetch for tests (no network). */
  fetchFn?: FetchFn;
}

export interface ResolvedEntry {
  cfg: ResolvedConnector;
  client: ConnectorClient;
}

export function connectorDefs(config: Config): ConnectorConfig[] {
  if (config.connectors && config.connectors.length > 0) return config.connectors;
  const u = config.upstream;
  if (!u.base_url || !u.api_key_env) {
    // Guaranteed present by config validation when no connectors are set; guard
    // so the types narrow without a non-null assertion.
    throw new Error(
      "no connector source: neither `connectors:` nor `upstream.base_url`+`upstream.api_key_env` is set",
    );
  }
  return [
    {
      id: "default",
      provider: "ollama",
      base_url: u.base_url,
      api_key_env: u.api_key_env,
      treat_403_as: "passthrough",
      quota_markers: [],
    },
  ];
}

export function resolveConnectors(config: Config, opts: ResolveOptions): ResolvedEntry[] {
  const u = config.upstream;
  return connectorDefs(config).map((c) => {
    const apiKey = opts.env[c.api_key_env];
    if (!apiKey) {
      opts.logger?.warn(
        { connector: c.id, env: c.api_key_env },
        "connector API key env var is unset; this connector will start DOWN on first auth failure",
      );
    }
    const timeoutMs = (c.request_timeout_s ?? u.request_timeout_s) * 1000;
    const client = createUpstreamClient({
      provider: c.provider,
      baseUrl: c.base_url,
      apiKey,
      timeoutMs,
      extraHeaders: c.extra_headers,
      fetchFn: opts.fetchFn,
    });
    const cfg: ResolvedConnector = {
      id: c.id,
      provider: c.provider,
      baseUrl: c.base_url,
      host: hostOf(c.base_url),
      hasKey: Boolean(apiKey),
      treat403As: c.treat_403_as,
      quotaMarkers: c.quota_markers,
      modelMap: c.model_map ?? {},
    };
    return { cfg, client };
  });
}

/** Display host (scheme + host), never the full URL/path/secret. */
function hostOf(url: string): string {
  try {
    const u = new URL(url);
    return `${u.protocol}//${u.host}`;
  } catch {
    return url;
  }
}
