import type { UpstreamClient } from "../types";
import {
  isHardReason,
  type ConnectorReason,
  type ConnectorState,
} from "./health";

/**
 * Connector registry — the single source of truth for connector health, counters,
 * selection order, and manual controls. Shared by the pooled upstream client
 * (which drives failover) and the panel (which reads snapshots and applies manual
 * actions). Holds no secrets: only a display host, never the API key.
 *
 * Concurrency (single-threaded event loop, but interleaving at every `await`):
 *  - **epoch guard** — an attempt captures the connector's epoch at `acquire`;
 *    a failure is applied only if the epoch is unchanged. `recordSuccess` and
 *    every manual action bump the epoch, so a late failure from an attempt that
 *    started earlier cannot overwrite a newer success/decision.
 *  - **single-flight probe** — a `cooling`/`down` connector whose cooldown has
 *    elapsed admits exactly one probe (`probeInFlight`); concurrent callers skip
 *    it. Prevents a probe stampede.
 *  - **monotonic cooldown** — `cooldownUntil` only ever moves later, and a hard
 *    `down` is never downgraded to `cooling` by a subsequent soft failure.
 */

/** A connector definition resolved from config (no secrets). */
export interface ResolvedConnector {
  id: string;
  provider: "ollama" | "openai-compat";
  /** Full base URL (used for the client; the panel shows only `host`). */
  baseUrl: string;
  /** Display host (scheme + host, no path/secrets). */
  host: string;
  /** True when the connector's api-key env var resolved to a non-empty value. */
  hasKey: boolean;
  treat403As: "passthrough" | "down";
  quotaMarkers: string[];
  /** Per-connector logical→upstream model-id map (identity when absent). */
  modelMap: Record<string, string>;
}

/** The live client for a connector, plus its native-show capability. */
export interface ConnectorClient extends UpstreamClient {
  readonly supportsNativeShow: boolean;
}

interface ConnectorRuntime {
  cfg: ResolvedConnector;
  client: ConnectorClient;
  state: ConnectorState;
  reason: ConnectorReason | null;
  lastError: string | null;
  stateChangedAt: number;
  /** Epoch/generation: bumped on success and on every manual action. */
  epoch: number;
  /** Wall-clock (ms) until which the connector is banned; 0 when serving. */
  cooldownUntil: number;
  probeInFlight: boolean;
  consecutiveFailures: number;
  totalRequests: number;
  totalFailures: number;
  lastSuccessAt: number | null;
  lastFailureAt: number | null;
  lastLatencyMs: number | null;
}

export interface RegistryOptions {
  /** Cooldown for soft (`cooling`) failures before a probe. Default 60 s. */
  cooldownMs?: number;
  /** Recheck window for hard (`down`) failures before a probe. Default 900 s. */
  downRecheckMs?: number;
  /** Injectable clock (epoch ms). Default `Date.now`. */
  now?: () => number;
}

/** A reservation token returned by `acquire`. */
export type Acquire =
  | { ok: true; epoch: number; probe: boolean }
  | { ok: false };

/** A plain, secret-free snapshot of one connector for the panel/JSON. */
export interface ConnectorSnapshot {
  id: string;
  provider: string;
  host: string;
  hasKey: boolean;
  state: ConnectorState;
  reason: ConnectorReason | null;
  active: boolean;
  pinned: boolean;
  lastError: string | null;
  stateChangedAt: number;
  cooldownUntil: number | null;
  cooldownRemainingMs: number | null;
  consecutiveFailures: number;
  totalRequests: number;
  totalFailures: number;
  lastSuccessAt: number | null;
  lastFailureAt: number | null;
  lastLatencyMs: number | null;
}

const DEFAULT_COOLDOWN_MS = 60_000;
const DEFAULT_DOWN_RECHECK_MS = 900_000;

export class ConnectorRegistry {
  private readonly runtimes: ConnectorRuntime[];
  private readonly byId = new Map<string, ConnectorRuntime>();
  private readonly cooldownMs: number;
  private readonly downRecheckMs: number;
  private readonly now: () => number;
  /** Operator-pinned preferred connector; tried first when usable. */
  private pinnedId: string | undefined;

  constructor(
    entries: Array<{ cfg: ResolvedConnector; client: ConnectorClient }>,
    opts: RegistryOptions = {},
  ) {
    this.cooldownMs = opts.cooldownMs ?? DEFAULT_COOLDOWN_MS;
    this.downRecheckMs = opts.downRecheckMs ?? DEFAULT_DOWN_RECHECK_MS;
    this.now = opts.now ?? Date.now;
    const started = this.now();
    this.runtimes = entries.map(({ cfg, client }) => {
      const r: ConnectorRuntime = {
        cfg,
        client,
        state: "up",
        reason: null,
        lastError: null,
        stateChangedAt: started,
        epoch: 0,
        cooldownUntil: 0,
        probeInFlight: false,
        consecutiveFailures: 0,
        totalRequests: 0,
        totalFailures: 0,
        lastSuccessAt: null,
        lastFailureAt: null,
        lastLatencyMs: null,
      };
      this.byId.set(cfg.id, r);
      return r;
    });
  }

  /** Connectors in selection order: pinned-first (when set), then config order. */
  private selectionOrder(): ConnectorRuntime[] {
    if (!this.pinnedId) return this.runtimes;
    const pinned = this.byId.get(this.pinnedId);
    if (!pinned) return this.runtimes;
    return [pinned, ...this.runtimes.filter((r) => r !== pinned)];
  }

  /** Ordered ids the pool should try this call (pinned-first). */
  order(): string[] {
    return this.selectionOrder().map((r) => r.cfg.id);
  }

  clientFor(id: string): ConnectorClient | undefined {
    return this.byId.get(id)?.client;
  }

  cfgFor(id: string): ResolvedConnector | undefined {
    return this.byId.get(id)?.cfg;
  }

  /** All ollama-capable connector ids in selection order (for native show/chat). */
  nativeShowOrder(): string[] {
    return this.selectionOrder()
      .filter((r) => r.client.supportsNativeShow && r.state !== "off")
      .map((r) => r.cfg.id);
  }

  /**
   * Try to reserve `id` for an attempt. `up` admits every caller; a `cooling`/
   * `down` connector whose cooldown has elapsed admits exactly one probe;
   * everything else is skipped. Returns the epoch to pass back to record*.
   */
  acquire(id: string): Acquire {
    const r = this.byId.get(id);
    if (!r) return { ok: false };
    if (r.state === "off") return { ok: false };
    if (r.state === "up") return { ok: true, epoch: r.epoch, probe: false };
    // cooling or down: single-flight probe once the cooldown has elapsed.
    if (this.now() >= r.cooldownUntil && !r.probeInFlight) {
      r.probeInFlight = true;
      return { ok: true, epoch: r.epoch, probe: true };
    }
    return { ok: false };
  }

  /** A call succeeded: the connector is authoritative-up (unless manually off). */
  recordSuccess(id: string, _epoch: number, latencyMs: number): void {
    const r = this.byId.get(id);
    if (!r) return;
    const now = this.now();
    r.probeInFlight = false;
    r.totalRequests += 1;
    r.consecutiveFailures = 0;
    r.lastSuccessAt = now;
    r.lastLatencyMs = latencyMs;
    r.epoch += 1; // invalidate any in-flight attempt that started earlier
    if (r.state === "off") return; // respect a manual disable that raced this call
    if (r.state !== "up") {
      r.state = "up";
      r.stateChangedAt = now;
    }
    r.reason = null;
    r.lastError = null;
    r.cooldownUntil = 0;
  }

  /**
   * A call failed with `reason`. Applied only if `epoch` still matches (else a
   * newer success/action supersedes it). Soft → `cooling`; hard → `down`.
   * Cooldown is monotonic and a hard `down` is never downgraded to `cooling`.
   */
  recordFailure(
    id: string,
    epoch: number,
    reason: ConnectorReason,
    opts: { error?: string; cooldownMs?: number } = {},
  ): void {
    const r = this.byId.get(id);
    if (!r) return;
    r.probeInFlight = false;
    if (r.epoch !== epoch) return; // stale attempt — a newer success/action won
    const now = this.now();
    r.totalRequests += 1;
    r.totalFailures += 1;
    r.consecutiveFailures += 1;
    r.lastFailureAt = now;
    r.lastError = opts.error ?? reason;
    if (r.state === "off") return; // respect a manual disable that raced this call
    const hard = isHardReason(reason);
    const cooldownMs = opts.cooldownMs ?? (hard ? this.downRecheckMs : this.cooldownMs);
    const nextUntil = Math.max(r.cooldownUntil, now + cooldownMs);
    if (r.state === "down" && !hard) {
      // Worst-of: a soft failure never downgrades a hard `down`; only extend.
      r.cooldownUntil = nextUntil;
      return;
    }
    const nextState: ConnectorState = hard ? "down" : "cooling";
    if (r.state !== nextState) {
      r.state = nextState;
      r.stateChangedAt = now;
    }
    r.reason = reason;
    r.cooldownUntil = nextUntil;
  }

  /**
   * A probe/attempt was abandoned by a client disconnect — release the probe
   * slot without recording a health failure (the connector's health is unknown,
   * not bad). Bumps nothing; leaves counters untouched.
   */
  recordAbandoned(id: string, epoch: number): void {
    const r = this.byId.get(id);
    if (!r) return;
    if (r.epoch === epoch) r.probeInFlight = false;
  }

  // --- manual controls ----------------------------------------------------

  /** Operator disables a connector; it is skipped until re-enabled. */
  disable(id: string): boolean {
    const r = this.byId.get(id);
    if (!r) return false;
    r.state = "off";
    r.reason = "manual";
    r.probeInFlight = false;
    r.cooldownUntil = 0;
    r.stateChangedAt = this.now();
    r.epoch += 1;
    if (this.pinnedId === id) this.pinnedId = undefined;
    return true;
  }

  /** Operator re-enables an `off` connector (no-op if not off). */
  enable(id: string): boolean {
    const r = this.byId.get(id);
    if (!r) return false;
    if (r.state !== "off") return true;
    this.forceUp(r);
    return true;
  }

  /** Force a connector back to `up` from any state (e.g. after topping up billing). */
  reset(id: string): boolean {
    const r = this.byId.get(id);
    if (!r) return false;
    this.forceUp(r);
    return true;
  }

  /** Pin a connector as the preferred active one (tried first when usable). */
  pin(id: string): boolean {
    if (!this.byId.has(id)) return false;
    this.pinnedId = id;
    return true;
  }

  /** Clear any active-connector pin. */
  unpin(): void {
    this.pinnedId = undefined;
  }

  private forceUp(r: ConnectorRuntime): void {
    r.state = "up";
    r.reason = null;
    r.lastError = null;
    r.probeInFlight = false;
    r.cooldownUntil = 0;
    r.consecutiveFailures = 0;
    r.stateChangedAt = this.now();
    r.epoch += 1;
  }

  // --- read side (panel / JSON) -------------------------------------------

  /** The connector currently serving steady traffic: pinned-if-up, else first up. */
  activeId(): string | undefined {
    if (this.pinnedId) {
      const p = this.byId.get(this.pinnedId);
      if (p && p.state === "up") return p.cfg.id;
    }
    for (const r of this.runtimes) if (r.state === "up") return r.cfg.id;
    return undefined;
  }

  snapshot(): ConnectorSnapshot[] {
    const now = this.now();
    const active = this.activeId();
    return this.runtimes.map((r) => {
      const cooling = r.state === "cooling" || r.state === "down";
      const remaining = cooling ? Math.max(0, r.cooldownUntil - now) : null;
      return {
        id: r.cfg.id,
        provider: r.cfg.provider,
        host: r.cfg.host,
        hasKey: r.cfg.hasKey,
        state: r.state,
        reason: r.reason,
        active: r.cfg.id === active,
        pinned: r.cfg.id === this.pinnedId,
        lastError: r.lastError,
        stateChangedAt: r.stateChangedAt,
        cooldownUntil: cooling ? r.cooldownUntil : null,
        cooldownRemainingMs: remaining,
        consecutiveFailures: r.consecutiveFailures,
        totalRequests: r.totalRequests,
        totalFailures: r.totalFailures,
        lastSuccessAt: r.lastSuccessAt,
        lastFailureAt: r.lastFailureAt,
        lastLatencyMs: r.lastLatencyMs,
      };
    });
  }
}
