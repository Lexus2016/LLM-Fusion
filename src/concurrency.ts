import pLimit from "p-limit";
import type { Logger } from "pino";
import { CircuitOpenError } from "./errors";

/**
 * Resilience primitives shared by all upstream-facing strategies:
 *
 *  - a single global concurrency limiter (`p-limit`) bounding in-flight upstream
 *    calls across every request and every fusion stage;
 *  - a per-model circuit breaker that fast-fails a model after repeated
 *    failures, then probes recovery;
 *  - an exponential-backoff delay helper for 429/5xx retries.
 *
 * All time-dependent behaviour is funnelled through injectable seams (`now()`
 * for the breaker, a `Sleeper` for retries, an `rng()` for jitter) so tests are
 * fully deterministic without real timers or sleeps.
 */

// --- Concurrency limiter --------------------------------------------------

/** Callable limiter: `limiter(() => doWork())` resolves once a slot is free. */
export type Limiter = ReturnType<typeof pLimit>;

/** Build the global limiter sized by `upstream.max_concurrency`. */
export function createLimiter(maxConcurrency: number): Limiter {
  return pLimit(Math.max(1, maxConcurrency));
}

/** Per-model concurrency budgets, keyed by REAL upstream model name. */
export interface PerModelConcurrency {
  /** Budget for models without an explicit override. Default: the global cap
   *  (i.e. no extra gate — behavior identical to a single global limiter). */
  defaultPerModel?: number;
  /** Explicit per-model budgets (e.g. cap a background small model at 2). */
  overrides?: Record<string, number>;
}

/**
 * Notified when a call has to WAIT at its own model gate — i.e. the model's
 * per-model budget is fully spent and this call is queuing behind it. This is
 * the ONLY externally visible signal that the gating is biting; without it a
 * saturated budget looks exactly like a slow upstream from the outside.
 */
export type GateWaitObserver = (info: {
  model: string;
  budget: number;
  queued: number;
  /**
   * Which limit is actually binding. `sizeFor` falls back to the GLOBAL cap for
   * any model without a per-model budget, so that model's gate is a no-op
   * wrapper around the global limiter — reporting its saturation as a per-model
   * problem would point the operator at `per_model_concurrency`, a knob that
   * cannot help. `scope` lets the logger name the right one.
   */
  scope: "per_model" | "global";
}) => void;

/**
 * Keyed limiter: every real upstream model gets its own gate IN FRONT of the
 * global limiter. Acquisition order is strictly model-gate -> global-slot, so
 * a saturated model queues at its OWN gate and can occupy at most its budget
 * of global-queue positions — a burst of background small-model calls can no
 * longer head-of-line-block interactive fusion turns. Uniform ordering across
 * all callers means no lock cycle is possible.
 */
/** Called when a model's gate fully drains, so a finished burst can flush its peak. */
export type GateDrainObserver = (model: string) => void;

export function createKeyedLimiter(
  global: Limiter,
  maxConcurrency: number,
  perModel: PerModelConcurrency = {},
  onGateWait?: GateWaitObserver,
  onGateDrain?: GateDrainObserver,
): (model: string) => <T>(fn: () => Promise<T> | T) => Promise<T> {
  // `outstanding` is counted here rather than read off p-limit's
  // `activeCount`/`pendingCount`: p-limit only starts a job on a microtask, so
  // during a synchronous burst every call still sees `activeCount === 0` and the
  // saturation signal would never fire. Counting at SUBMIT time is also the
  // semantically right moment — a call holds its model budget from submit until
  // it settles, including the stretch where it is waiting for a global slot.
  interface Gate {
    limiter: Limiter;
    budget: number;
    /** Submitted and not yet settled = slots claimed at this model's gate. */
    outstanding: number;
  }
  const gates = new Map<string, Gate>();
  // Submitted-and-unsettled across EVERY model. Without it, global-cap
  // contention between DIFFERENT models is invisible: with a cap of 2 held by
  // models A and B, a call to C waits at the global limiter while C's own gate
  // shows a single outstanding call, so no per-model gate ever reports it.
  let globalOutstanding = 0;
  const sizeFor = (model: string): number =>
    Math.max(1, perModel.overrides?.[model] ?? perModel.defaultPerModel ?? maxConcurrency);
  return (model: string) => {
    let gate = gates.get(model);
    if (!gate) {
      const budget = sizeFor(model);
      gate = { limiter: pLimit(budget), budget, outstanding: 0 };
      gates.set(model, gate);
    }
    const g = gate;
    // A budget at or above the global cap means no per-model budget is configured
    // for this model: its gate can never bind before the global limiter does, so
    // queueing there is really global-cap congestion.
    const perModelBinds = g.budget < maxConcurrency;
    // Telemetry must never break the data path: an observer that throws would
    // otherwise escape synchronously out of `limiterFor(m)(fn)` — before the
    // upstream call was ever submitted — and leave the counters inflated.
    const notify = (info: Parameters<GateWaitObserver>[0]): void => {
      if (onGateWait === undefined) return;
      try {
        onGateWait(info);
      } catch {
        /* observer is best-effort */
      }
    };
    return <T>(fn: () => Promise<T> | T): Promise<T> => {
      // Verdict A — decided at SUBMIT, because that is the moment the call starts
      // holding model budget and the model gate is what it will queue at. The
      // scope depends on whether this model actually has a budget of its own: if
      // it does not, `sizeFor` gave it the global cap, the gate and the cap are
      // the same width, and the knob that would help is `max_concurrency`.
      const queuedAtModel = g.outstanding >= g.budget;
      g.outstanding += 1;
      if (queuedAtModel) {
        notify(
          perModelBinds
            ? { model, budget: g.budget, queued: g.outstanding - g.budget, scope: "per_model" }
            : { model, budget: maxConcurrency, queued: g.outstanding - g.budget, scope: "global" },
        );
      }
      return g
        .limiter(() => {
          // GLOBAL verdict, decided HERE and not at submit. `globalOutstanding`
          // must count calls that have reached the global limiter, not calls
          // merely submitted: with a global cap of 4 and a model budget of 1,
          // six submissions to that model occupy ONE global slot, yet a
          // submit-time counter would read 6 and cry global saturation at the
          // next model — a false warning with a fabricated depth.
          const queuedGlobally = globalOutstanding >= maxConcurrency;
          globalOutstanding += 1;
          // Verdict B — CROSS-MODEL contention: this call cleared its own gate
          // and still has to wait because other models hold every global slot.
          // Suppressed when verdict A already fired for this call, so one wait
          // never produces two lines.
          if (queuedGlobally && !queuedAtModel) {
            notify({
              model,
              budget: maxConcurrency,
              queued: globalOutstanding - maxConcurrency,
              scope: "global",
            });
          }
          return Promise.resolve(global(fn)).finally(() => {
            globalOutstanding -= 1;
          });
        })
        .finally(() => {
          g.outstanding -= 1;
          // The gate emptied: a burst that ended inside the throttle window would
          // otherwise never report the depth it actually reached.
          if (g.outstanding === 0 && onGateDrain !== undefined) {
            try {
              onGateDrain(model);
            } catch {
              /* observer is best-effort */
            }
          }
        });
    };
  };
}

// --- Circuit breaker ------------------------------------------------------

/** Injectable wall-clock seam (epoch ms). */
export type Clock = () => number;

export type BreakerState = "closed" | "open" | "half-open";

export interface CircuitBreakerOptions {
  /** Consecutive failures that trip the breaker open. Default 5. */
  failureThreshold?: number;
  /** How long the breaker stays open before a half-open probe. Default 30 s. */
  cooldownMs?: number;
  /** Time source; default `Date.now`. Tests pass a controllable clock. */
  now?: Clock;
}

interface ModelState {
  state: BreakerState;
  consecutiveFailures: number;
  openedAt: number;
  /** A half-open probe is currently in flight (no concurrent probes allowed). */
  probeInFlight: boolean;
}

const DEFAULT_FAILURE_THRESHOLD = 5;
const DEFAULT_COOLDOWN_MS = 30_000;

/**
 * Per-model circuit breaker. Keyed by the *real* upstream model name so a
 * degraded model fast-fails for every virtual model that routes to it.
 */
export class CircuitBreaker {
  private readonly states = new Map<string, ModelState>();
  private readonly threshold: number;
  private readonly cooldownMs: number;
  private readonly now: Clock;

  constructor(opts: CircuitBreakerOptions = {}) {
    this.threshold = opts.failureThreshold ?? DEFAULT_FAILURE_THRESHOLD;
    this.cooldownMs = opts.cooldownMs ?? DEFAULT_COOLDOWN_MS;
    this.now = opts.now ?? Date.now;
  }

  private get(model: string): ModelState {
    let s = this.states.get(model);
    if (!s) {
      s = { state: "closed", consecutiveFailures: 0, openedAt: 0, probeInFlight: false };
      this.states.set(model, s);
    }
    return s;
  }

  /** Promote an expired `open` breaker to `half-open` (cooldown elapsed). */
  private refresh(s: ModelState): void {
    if (s.state === "open" && this.now() - s.openedAt >= this.cooldownMs) {
      s.state = "half-open";
      s.probeInFlight = false;
    }
  }

  /** Current state, after applying any pending cooldown transition. */
  getState(model: string): BreakerState {
    const s = this.get(model);
    this.refresh(s);
    return s.state;
  }

  /**
   * Whether a call to `model` may proceed. `closed` always allows; `open` always
   * denies; `half-open` allows exactly one probe at a time. Reserves the probe
   * slot as a side effect when it returns true in the half-open state.
   */
  canAttempt(model: string): boolean {
    const s = this.get(model);
    this.refresh(s);
    if (s.state === "closed") return true;
    if (s.state === "open") return false;
    if (s.probeInFlight) return false;
    s.probeInFlight = true;
    return true;
  }

  /** A call succeeded: close the breaker and reset the failure count. */
  recordSuccess(model: string): void {
    const s = this.get(model);
    s.state = "closed";
    s.consecutiveFailures = 0;
    s.probeInFlight = false;
  }

  /** A call failed: count it, and open (or re-open a failed probe). */
  recordFailure(model: string): void {
    const s = this.get(model);
    s.consecutiveFailures += 1;
    const wasProbe = s.state === "half-open";
    s.probeInFlight = false;
    if (wasProbe) {
      s.state = "open";
      s.openedAt = this.now();
      return;
    }
    if (s.consecutiveFailures >= this.threshold) {
      s.state = "open";
      s.openedAt = this.now();
    }
  }

  /**
   * Release a reserved half-open probe without recording a failure. Called when
   * a probe is cancelled by the client (abort) before it could succeed or fail:
   * the model's health is unchanged, but `probeInFlight` must be freed so the
   * next call can probe again. Without this, a cancelled probe leaves the
   * breaker stuck in half-open forever.
   */
  recordProbeAbandoned(model: string): void {
    const s = this.get(model);
    this.refresh(s);
    if (s.state === "half-open") s.probeInFlight = false;
  }

  /** Reset all breaker state (test helper). */
  reset(): void {
    this.states.clear();
  }
}

// --- Exponential backoff --------------------------------------------------

export interface BackoffOptions {
  /** Delay for the first retry. Default 200 ms. */
  baseMs?: number;
  /** Growth factor per attempt. Default 2. */
  factor?: number;
  /** Hard cap on the (pre-jitter) delay. Default 5 s. */
  maxMs?: number;
  /** Fraction of the delay subject to random jitter, 0..1. Default 0.2. */
  jitter?: number;
  /** [0,1) source; default `Math.random`. Tests pass a deterministic value. */
  rng?: () => number;
}

const DEFAULT_BACKOFF: Required<Omit<BackoffOptions, "rng">> = {
  baseMs: 200,
  factor: 2,
  maxMs: 5_000,
  jitter: 0.2,
};

/**
 * Delay (ms) for a 0-based retry `attempt`. The capped exponential value
 * `min(maxMs, baseMs * factor^attempt)` has up to `jitter` of its magnitude
 * subtracted at random, so the result always lands in
 * `[raw * (1 - jitter), raw]`.
 */
export function backoffDelay(attempt: number, opts: BackoffOptions = {}): number {
  const baseMs = opts.baseMs ?? DEFAULT_BACKOFF.baseMs;
  const factor = opts.factor ?? DEFAULT_BACKOFF.factor;
  const maxMs = opts.maxMs ?? DEFAULT_BACKOFF.maxMs;
  const jitter = Math.min(1, Math.max(0, opts.jitter ?? DEFAULT_BACKOFF.jitter));
  const rng = opts.rng ?? Math.random;

  const raw = Math.min(maxMs, baseMs * Math.pow(factor, Math.max(0, attempt)));
  const delta = raw * jitter * rng();
  return Math.max(0, Math.round(raw - delta));
}

// --- Sleeper --------------------------------------------------------------

/** Injectable delay primitive; tests pass a no-op or a recorder. */
export type Sleeper = (ms: number) => Promise<void>;

export const realSleep: Sleeper = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --- Failover policy + bundled resilience ---------------------------------

export interface FailoverPolicy {
  /** Same-member retries on 429 before surfacing the rate-limit. Default 5. */
  maxRateLimitRetries: number;
  /** Same-member retries on 5xx/network before advancing the chain. Default 1. */
  maxServerRetries: number;
}

const DEFAULT_POLICY: FailoverPolicy = {
  maxRateLimitRetries: 5,
  maxServerRetries: 1,
};

/** Everything the strategies need to be resilient, built once per process. */
export interface Resilience {
  /** Global limiter — for calls with no specific upstream model (e.g. capability discovery). */
  limiter: Limiter;
  /** Per-model gate composed with the global limiter — the default for model-bound upstream calls. */
  limiterFor: (model: string) => <T>(fn: () => Promise<T> | T) => Promise<T>;
  breaker: CircuitBreaker;
  sleep: Sleeper;
  backoff: BackoffOptions;
  policy: FailoverPolicy;
}

export interface ResilienceOptions {
  maxConcurrency: number;
  perModel?: PerModelConcurrency;
  failureThreshold?: number;
  cooldownMs?: number;
  now?: Clock;
  sleep?: Sleeper;
  backoff?: BackoffOptions;
  policy?: Partial<FailoverPolicy>;
  /** Notified when a call queues at its model gate (see GateWaitObserver). */
  onGateWait?: GateWaitObserver;
  /** Notified when a model's gate drains (see GateDrainObserver). */
  onGateDrain?: GateDrainObserver;
}

/** Compose a `Resilience` bundle with sane defaults. */
export function createResilience(opts: ResilienceOptions): Resilience {
  const limiter = createLimiter(opts.maxConcurrency);
  return {
    limiter,
    limiterFor: createKeyedLimiter(limiter, opts.maxConcurrency, opts.perModel, opts.onGateWait, opts.onGateDrain),
    breaker: new CircuitBreaker({
      failureThreshold: opts.failureThreshold,
      cooldownMs: opts.cooldownMs,
      now: opts.now,
    }),
    sleep: opts.sleep ?? realSleep,
    backoff: opts.backoff ?? {},
    policy: {
      maxRateLimitRetries: opts.policy?.maxRateLimitRetries ?? DEFAULT_POLICY.maxRateLimitRetries,
      maxServerRetries: opts.policy?.maxServerRetries ?? DEFAULT_POLICY.maxServerRetries,
    },
  };
}

/**
 * Throttled `GateWaitObserver` that logs at most one line per model per
 * `intervalMs`. Throttling is the whole point: the observer fires on EVERY
 * queued call, and a Claude-Code-style burst (80–130 calls/min) would otherwise
 * turn a real diagnostic into log spam nobody reads.
 */
export function throttledGateWaitLogger(
  logger: Logger,
  opts: { intervalMs?: number; now?: Clock } = {},
): { onWait: GateWaitObserver; onDrain: GateDrainObserver } {
  const intervalMs = opts.intervalMs ?? 30_000;
  const now = opts.now ?? Date.now;
  interface ModelWindow {
    /** When this model last emitted a line; `undefined` = never. */
    lastLogged?: number;
    /** Deepest queue seen since that line — the number that actually matters. */
    peak: number;
    /** The peak value the last emitted line actually carried. */
    reported: number;
    budget: number;
    scope: "per_model" | "global";
  }
  // Keyed by model AND scope. One model can report both kinds of wait — its own
  // budget on one call, the global cap on the next — and a single per-model
  // window would flush a peak accumulated under one scope while labelled with
  // the other, naming the wrong limit and the wrong budget.
  const SCOPES = ["per_model", "global"] as const;
  const keyOf = (model: string, scope: "per_model" | "global"): string => `${model}\u0000${scope}`;
  const windows = new Map<string, ModelWindow>();
  const windowFor = (model: string, scope: "per_model" | "global"): ModelWindow => {
    const k = keyOf(model, scope);
    let w = windows.get(k);
    if (w === undefined) {
      w = { peak: 0, reported: 0, budget: 0, scope };
      windows.set(k, w);
    }
    return w;
  };
  const onWait: GateWaitObserver = ({ model, budget, queued, scope }) => {
    const w = windowFor(model, scope);
    w.budget = budget;
    // Track the peak on EVERY event, including the throttled ones. Reporting only
    // `queued` would make the log actively misleading: a burst's first sample is
    // always `queued: 1`, so a gate that went on to back up 80 deep would be
    // recorded as "1 call queued" and then stay silent for the rest of the window.
    w.peak = Math.max(w.peak, queued);
    const t = now();
    if (w.lastLogged !== undefined && t - w.lastLogged < intervalMs) return;
    emit(model, w, queued);
    w.lastLogged = t;
  };

  function emit(model: string, w: ModelWindow, queued: number): void {
    logger.warn(
      { model, budget: w.budget, queued, peak_queued: w.peak, scope: w.scope },
      w.scope === "per_model"
        ? "upstream: per-model concurrency budget saturated; calls are queuing at the model gate (raise upstream.per_model_concurrency for this model)"
        : "upstream: concurrency saturated for this model, which has NO per-model budget — the binding limit is upstream.max_concurrency",
    );
    w.reported = w.peak;
    w.peak = 0;
  }

  /**
   * The gate emptied. A burst that spiked to 80 and drained inside the throttle
   * window would otherwise leave only its first line on the record — `queued: 1`
   * — and the real depth would never be logged at all, because the peak is only
   * carried out by the NEXT wait event. Flush it here instead.
   */
  const onDrain: GateDrainObserver = (model) => {
    for (const scope of SCOPES) {
      const w = windows.get(keyOf(model, scope));
      if (w === undefined) continue;
      // Reset unconditionally. Returning early on a peak that was already
      // reported used to LEAVE it in place, so a stale depth belonging to a
      // burst that finished long ago could still be emitted by the next wait
      // once the throttle interval elapsed.
      const unreported = w.peak > w.reported;
      const peak = w.peak;
      w.peak = 0;
      if (!unreported) continue;
      w.reported = peak;
      logger.warn(
        { model, budget: w.budget, peak_queued: peak, scope, drained: true },
        "upstream: concurrency burst drained; this is the depth it actually reached",
      );
    }
  };

  return { onWait, onDrain };
}

/**
 * Build a `Resilience` bundle straight from an `upstream` config block, so the
 * server and every strategy fallback wire the SAME per-model budgets — a
 * fallback that silently dropped them would disable the keyed gating on any
 * non-server call path. Pass a `logger` to get the throttled gate-saturation
 * warning; without one the gating still works, it is just invisible.
 */
export function resilienceForUpstream(
  upstream: {
    max_concurrency: number;
    per_model_concurrency?: Record<string, number>;
    per_model_concurrency_default?: number;
  },
  opts: { logger?: Logger } = {},
): Resilience {
  return createResilience({
    maxConcurrency: upstream.max_concurrency,
    perModel: {
      defaultPerModel: upstream.per_model_concurrency_default,
      overrides: upstream.per_model_concurrency,
    },
    ...(opts.logger
      ? (() => {
          const obs = throttledGateWaitLogger(opts.logger);
          return { onGateWait: obs.onWait, onGateDrain: obs.onDrain };
        })()
      : {}),
  });
}

// Re-export so callers can construct the typed fast-fail error without reaching
// into errors.ts for breaker-specific concerns.
export { CircuitOpenError };
