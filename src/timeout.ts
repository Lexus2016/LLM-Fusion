import { UpstreamTimeoutError } from "./errors";

/**
 * Stage-level timeout machinery shared by every strategy that bounds an
 * individual upstream call (fusion panel/judge/synth, the smart router). Kept
 * separate from any one strategy so all of them race the same way and tests can
 * inject a deterministic timer.
 */

/** A scheduled stage timeout: a promise that resolves on expiry, plus a cancel. */
export interface StageTimeout {
  expired: Promise<void>;
  cancel(): void;
}

/** Schedules a stage timeout. Default uses real timers; tests inject their own. */
export type TimerFactory = (ms: number) => StageTimeout;

export const realTimer: TimerFactory = (ms) => {
  let handle: ReturnType<typeof setTimeout> | undefined;
  const expired = new Promise<void>((resolve) => {
    handle = setTimeout(resolve, ms);
  });
  return {
    expired,
    cancel() {
      if (handle !== undefined) clearTimeout(handle);
    },
  };
};

/**
 * Combine an optional client-abort signal with a stage's own abort signal, so a
 * client disconnect AND a stage timeout both cancel the in-flight upstream call.
 * Returns the stage signal alone when there is no client signal.
 */
export function combineSignals(client: AbortSignal | undefined, stage: AbortSignal): AbortSignal {
  return client ? AbortSignal.any([client, stage]) : stage;
}

/**
 * Race `work` against a stage timeout. On timeout `onTimeout` runs first — used
 * to abort the in-flight upstream call so its concurrency-limiter slot frees
 * promptly instead of lingering until the call settles on its own — and a typed
 * `UpstreamTimeoutError` is thrown so the proxy is always first to fail.
 */
export async function withTimeout<T>(
  work: Promise<T>,
  ms: number,
  timer: TimerFactory,
  label: string,
  onTimeout?: () => void,
): Promise<T> {
  const t = timer(ms);
  try {
    return await Promise.race([
      work,
      t.expired.then((): never => {
        onTimeout?.();
        throw new UpstreamTimeoutError(label);
      }),
    ]);
  } finally {
    t.cancel();
  }
}
