import type { Logger } from "pino";
import type { UpstreamClient, UpstreamRouter } from "../types";
import { ConnectorRegistry, type ConnectorSnapshot } from "./registry";
import { PooledUpstreamClient } from "./pooled_client";
import type { ResolvedGroup } from "./resolve";
import { NoConnectorAvailableError } from "../errors";

/**
 * Owns one `ConnectorRegistry` + `PooledUpstreamClient` PER provider group, and
 * routes a virtual model to the pool of the group it is bound to. Failover thus
 * stays WITHIN a provider (same models), and a model never silently jumps to a
 * provider with a different catalog. Strategies are unaffected: `dispatch` sets
 * `ctx.client` to `poolFor(model.provider)` before the strategy runs.
 */

export interface ProviderGroupSnapshot {
  id: string;
  type: string;
  activeId: string | null;
  accounts: ConnectorSnapshot[];
}

export interface ProviderRouterOptions {
  cooldownMs?: number;
  downRecheckMs?: number;
  logger?: Logger;
  now?: () => number;
}

interface GroupRuntime {
  id: string;
  type: "ollama" | "openai-compat";
  registry: ConnectorRegistry;
  pool: PooledUpstreamClient;
}

export class ProviderRouter implements UpstreamRouter {
  private groups = new Map<string, GroupRuntime>();
  private readonly opts: ProviderRouterOptions;

  constructor(groups: ResolvedGroup[], opts: ProviderRouterOptions = {}) {
    this.opts = opts;
    this.groups = this.build(groups);
  }

  private build(groups: ResolvedGroup[]): Map<string, GroupRuntime> {
    const map = new Map<string, GroupRuntime>();
    for (const g of groups) {
      const registry = new ConnectorRegistry(g.accounts, {
        cooldownMs: this.opts.cooldownMs,
        downRecheckMs: this.opts.downRecheckMs,
        now: this.opts.now,
      });
      const pool = new PooledUpstreamClient(registry, { logger: this.opts.logger, now: this.opts.now });
      map.set(g.id, { id: g.id, type: g.type, registry, pool });
    }
    return map;
  }

  /**
   * Rebuild the groups from a fresh resolution (used when the config editor
   * changes `providers:`). Per-account health resets — acceptable for an operator
   * edit, and only invoked when the providers section actually changed.
   */
  reload(groups: ResolvedGroup[]): void {
    this.groups = this.build(groups);
  }

  /** The only group's id, or undefined when there are 0 or 2+ groups. */
  get soleGroupId(): string | undefined {
    return this.groups.size === 1 ? [...this.groups.keys()][0] : undefined;
  }

  /** The first group's pool — used for health-neutral capability discovery. */
  get defaultPool(): UpstreamClient {
    for (const g of this.groups.values()) return g.pool;
    throw new Error("provider router has no groups");
  }

  /** Pool for a model bound to `group` (falls back to the sole/first group). */
  poolFor(group?: string): UpstreamClient {
    if (group !== undefined) {
      const rt = this.groups.get(group);
      if (!rt) throw new NoConnectorAvailableError(`unknown provider group '${group}'`);
      return rt.pool;
    }
    for (const g of this.groups.values()) return g.pool; // sole/first
    throw new NoConnectorAvailableError("no provider group is configured");
  }

  /** The registry that owns `accountId` (account ids are globally unique). */
  registryForAccount(accountId: string): ConnectorRegistry | undefined {
    for (const g of this.groups.values()) {
      if (g.registry.cfgFor(accountId)) return g.registry;
    }
    return undefined;
  }

  /** True when at least one account in any group is currently up. */
  anyUp(): boolean {
    for (const g of this.groups.values()) {
      if (g.registry.snapshot().some((s) => s.state === "up")) return true;
    }
    return false;
  }

  /** The ids of all configured provider groups, in declaration order. */
  groupIds(): string[] {
    return [...this.groups.keys()];
  }

  /**
   * Ask a group's provider for its available model ids (for the panel's model
   * picker). Uses any account that carries a key — model catalogs are identical
   * across a group's accounts, so a cooling/down account still answers discovery.
   * Throws when the group is unknown, has no keyed account, or the client cannot
   * list models. The caller (config editor) turns these into a friendly message.
   */
  async listGroupModels(groupId: string, opts: { signal?: AbortSignal } = {}): Promise<string[]> {
    const rt = this.groups.get(groupId);
    if (!rt) throw new NoConnectorAvailableError(`unknown provider group '${groupId}'`);
    const snaps = rt.registry.snapshot();
    const chosen = snaps.find((s) => s.hasKey) ?? snaps[0];
    if (!chosen) throw new NoConnectorAvailableError(`provider group '${groupId}' has no accounts`);
    const client = rt.registry.clientFor(chosen.id);
    if (!client?.listModels) {
      throw new NoConnectorAvailableError(`provider group '${groupId}' cannot list models`);
    }
    return client.listModels(opts);
  }

  /** Grouped snapshot for the panel: providers, each with its accounts + active. */
  snapshot(): { providers: ProviderGroupSnapshot[] } {
    const providers: ProviderGroupSnapshot[] = [];
    for (const g of this.groups.values()) {
      providers.push({
        id: g.id,
        type: g.type,
        activeId: g.registry.activeId() ?? null,
        accounts: g.registry.snapshot(),
      });
    }
    return { providers };
  }
}
