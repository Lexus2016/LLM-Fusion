import pLimit from "p-limit";
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
 * Keyed limiter: every real upstream model gets its own gate IN FRONT of the
 * global limiter. Acquisition order is strictly model-gate -> global-slot, so
 * a saturated model queues at its OWN gate and can occupy at most its budget
 * of global-queue positions — a burst of background small-model calls can no
 * longer head-of-line-block interactive fusion turns. Uniform ordering across
 * all callers means no lock cycle is possible.
 */
export function createKeyedLimiter(
  global: Limiter,
  maxConcurrency: number,
  perModel: PerModelConcurrency = {},
): (model: string) => <T>(fn: () => Promise<T> | T) => Promise<T> {
  const gates = new Map<string, Limiter>();
  const sizeFor = (model: string): number =>
    Math.max(1, perModel.overrides?.[model] ?? perModel.defaultPerModel ?? maxConcurrency);
  return (model: string) => {
    let gate = gates.get(model);
    if (!gate) {
      gate = pLimit(sizeFor(model));
      gates.set(model, gate);
    }
    const g = gate;
    return <T>(fn: () => Promise<T> | T): Promise<T> => g(() => global(fn));
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
}

/** Compose a `Resilience` bundle with sane defaults. */
export function createResilience(opts: ResilienceOptions): Resilience {
  const limiter = createLimiter(opts.maxConcurrency);
  return {
    limiter,
    limiterFor: createKeyedLimiter(limiter, opts.maxConcurrency, opts.perModel),
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
 * Build a `Resilience` bundle straight from an `upstream` config block, so the
 * server and every strategy fallback wire the SAME per-model budgets — a
 * fallback that silently dropped them would disable the keyed gating on any
 * non-server call path.
 */
export function resilienceForUpstream(upstream: {
  max_concurrency: number;
  per_model_concurrency?: Record<string, number>;
  per_model_concurrency_default?: number;
}): Resilience {
  return createResilience({
    maxConcurrency: upstream.max_concurrency,
    perModel: {
      defaultPerModel: upstream.per_model_concurrency_default,
      overrides: upstream.per_model_concurrency,
    },
  });
}

// Re-export so callers can construct the typed fast-fail error without reaching
// into errors.ts for breaker-specific concerns.
export { CircuitOpenError };
