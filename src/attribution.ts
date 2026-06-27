import type { Logger } from "pino";
import { CircuitOpenError, UpstreamTimeoutError } from "./errors";

/**
 * Per-upstream-call error attribution (observability).
 *
 * Every upstream-call failure — a thrown network/timeout error, a non-OK status
 * the proxy treats as an availability failure (429 / 5xx), or a circuit-open
 * skip — is logged with a CONSISTENT structured shape so a `grep` of the proxy
 * log can answer "which model and which stage is slow or throttling, and how
 * slow":
 *
 *  - `stage`          — the pipeline stage that issued the call;
 *  - `upstream_model` — the REAL upstream model the call targeted;
 *  - `err_kind`       — coarse failure class (timeout / rate_limit / ...);
 *  - `status`         — the HTTP status when the upstream answered (omitted for
 *                       thrown errors and circuit-open skips);
 *  - `latency_ms`     — wall-clock time the call took (0 for a circuit-open skip,
 *                       where no call is made).
 *
 * Prompt CONTENT never reaches this module — only model names, status, timing.
 */

/** Which pipeline stage issued the upstream call that failed. */
export type UpstreamStage =
  | "panel"
  | "judge"
  | "synth"
  | "single"
  | "router"
  | "failover-member";

/** Coarse failure class for grep / aggregation. */
export type UpstreamFailureKind =
  | "error" // network / unknown thrown error
  | "timeout" // UpstreamTimeoutError (deadline hit)
  | "rate_limit" // HTTP 429
  | "server_error" // HTTP 5xx
  | "client_error" // HTTP 4xx (non-429)
  | "circuit_open"; // skipped: breaker open, no call made

export interface UpstreamFailure {
  stage: UpstreamStage;
  /** The real upstream model name the failed call targeted. */
  model: string;
  kind: UpstreamFailureKind;
  /** HTTP status when the upstream answered; omit for thrown errors / circuit-open. */
  status?: number;
  /** Call latency in ms; 0 for a circuit-open skip (no call was made). */
  latencyMs: number;
  /** Short failure detail (error message / status note). Never prompt content. */
  reason?: string;
}

/** Classify a thrown upstream error into a coarse failure kind. */
export function failureKindForError(err: unknown): UpstreamFailureKind {
  if (err instanceof UpstreamTimeoutError) return "timeout";
  if (err instanceof CircuitOpenError) return "circuit_open";
  return "error";
}

/** Classify a non-OK upstream HTTP status into a coarse failure kind. */
export function failureKindForStatus(status: number): UpstreamFailureKind {
  if (status === 429) return "rate_limit";
  if (status >= 500) return "server_error";
  return "client_error";
}

/**
 * True for statuses the proxy treats as an upstream AVAILABILITY failure: 429
 * (throttling) and 5xx (server). A 4xx other than 429 is a client/request error
 * — it is surfaced to the caller and never counted against the model's health.
 * Shared by the circuit breaker and the attribution logger so both agree.
 */
export function isAvailabilityFailureStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

/** Emit one structured `upstream call failed` warn line carrying the attribution fields. */
export function logUpstreamFailure(logger: Logger, f: UpstreamFailure): void {
  const fields: Record<string, unknown> = {
    stage: f.stage,
    upstream_model: f.model,
    err_kind: f.kind,
    latency_ms: f.latencyMs,
  };
  if (f.status !== undefined) fields.status = f.status;
  if (f.reason !== undefined) fields.reason = f.reason;
  logger.warn(fields, "upstream call failed");
}
