import type { Logger } from "pino";
import type { AccountConfig, Config } from "../config";
import type { FetchFn } from "../types";
import { createUpstreamClient } from "../upstream/provider";
import type { ConnectorClient, ResolvedConnector } from "./registry";

/**
 * Resolve the config's provider groups into ready-to-run structures. Each group
 * is one upstream provider (Ollama Cloud, OpenRouter, …) and its ordered accounts;
 * failover happens WITHIN a group. Reads each account's API key from its
 * `api_key_env` env var (never stores the value in `cfg`). When no explicit
 * `providers:` map is configured, a single `default` group is synthesised from
 * the legacy `upstream.base_url` + `upstream.api_key_env` (backward compatible).
 */

export interface ResolveOptions {
  env: Record<string, string | undefined>;
  logger?: Logger;
  /** Injectable fetch for tests (no network). */
  fetchFn?: FetchFn;
}

export interface ResolvedAccount {
  cfg: ResolvedConnector;
  client: ConnectorClient;
}

export interface ResolvedGroup {
  id: string;
  type: "ollama" | "openai-compat";
  accounts: ResolvedAccount[];
}

interface GroupDef {
  id: string;
  type: "ollama" | "openai-compat";
  base_url: string | undefined;
  accounts: AccountConfig[];
}

/** Config's provider groups (or the single synthesised `default` group). */
export function groupDefs(config: Config): GroupDef[] {
  if (config.providers && Object.keys(config.providers).length > 0) {
    return Object.entries(config.providers).map(([id, g]) => ({
      id,
      type: g.type,
      base_url: g.base_url,
      accounts: g.accounts,
    }));
  }
  const u = config.upstream;
  if (!u.base_url || !u.api_key_env) {
    throw new Error(
      "no provider source: neither `providers:` nor `upstream.base_url`+`upstream.api_key_env` is set",
    );
  }
  return [
    {
      id: "default",
      type: "ollama",
      base_url: u.base_url,
      accounts: [
        { id: "default", api_key_env: u.api_key_env, treat_403_as: "passthrough", quota_markers: [] },
      ],
    },
  ];
}

export function resolveProviders(config: Config, opts: ResolveOptions): ResolvedGroup[] {
  const u = config.upstream;
  return groupDefs(config).map((g) => {
    const accounts = g.accounts.map((acc): ResolvedAccount => {
      const apiKey = opts.env[acc.api_key_env];
      if (!apiKey) {
        opts.logger?.warn(
          { provider: g.id, account: acc.id, env: acc.api_key_env },
          "account API key env var is unset; this account will go DOWN on first auth failure",
        );
      }
      const baseUrl = acc.base_url ?? g.base_url;
      if (!baseUrl) {
        // Guaranteed present by config validation; guard so the type narrows.
        throw new Error(`account '${acc.id}' in provider '${g.id}' has no base_url`);
      }
      const timeoutMs = (acc.request_timeout_s ?? u.request_timeout_s) * 1000;
      const client = createUpstreamClient({
        provider: g.type,
        baseUrl,
        apiKey,
        timeoutMs,
        extraHeaders: acc.extra_headers,
        fetchFn: opts.fetchFn,
      });
      const cfg: ResolvedConnector = {
        id: acc.id,
        group: g.id,
        provider: g.type,
        baseUrl,
        host: hostOf(baseUrl),
        hasKey: Boolean(apiKey),
        treat403As: acc.treat_403_as,
        quotaMarkers: acc.quota_markers,
        modelMap: acc.model_map ?? {},
      };
      return { cfg, client };
    });
    return { id: g.id, type: g.type, accounts };
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
