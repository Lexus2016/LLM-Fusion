import { FusionError, UpstreamNetworkError, UpstreamTimeoutError } from "../errors";

/**
 * Connector health model + response classification.
 *
 * A connector (one upstream account/endpoint) moves through four states:
 *  - `up`      — serving.
 *  - `cooling` — transient failure; auto-probes after a short cooldown. Soft
 *                reasons: rate_limit, server_error, network, timeout.
 *  - `down`    — hard failure that won't self-heal on a short timer; auto-probes
 *                only after a much longer recheck. Hard reasons: auth, payment,
 *                quota. This is the "billing ended / key dead / daily cap" case.
 *  - `off`     — operator-disabled via the panel; only manual enable revives it.
 *
 * Classification is deliberately conservative and provider-tunable (see
 * `ClassifyOptions`) so a transient 429 is never mistaken for account death.
 */

export type ConnectorState = "up" | "cooling" | "down" | "off";

export type ConnectorReason =
  // soft → cooling
  | "rate_limit"
  | "server_error"
  | "network"
  | "timeout"
  // hard → down
  | "auth"
  | "payment"
  | "quota"
  // operator
  | "manual";

/** Soft reasons cool down briefly and auto-probe; hard reasons stay down. */
export function isHardReason(reason: ConnectorReason): boolean {
  return reason === "auth" || reason === "payment" || reason === "quota";
}

export interface ClassifyOptions {
  /** How to treat a 403. Default `passthrough` (client/request error, connector
   *  stays healthy). Some gateways return 403 for no-credits/disabled — set
   *  `down` per-connector there. */
  treat403As: "passthrough" | "down";
  /** Opt-in, per-connector lowercased substrings that escalate a 429 body from a
   *  transient `rate_limit` (cooling) to `quota` (down). Empty = never escalate
   *  on body text (safe default; avoids false-DOWN of a transient rate limit). */
  quotaMarkers: string[];
}

export const DEFAULT_CLASSIFY_OPTIONS: ClassifyOptions = {
  treat403As: "passthrough",
  quotaMarkers: [],
};

/**
 * Rank of a failure for deciding which one to surface to the strategy above when
 * every connector fails. Higher = more recoverable = preferred to surface, so a
 * primary connector's transient 429 wins over a dead backup's 401 (which would
 * otherwise be handed to the client as a hard auth error). See design §11.
 */
export const SEVERITY: Record<ConnectorReason, number> = {
  rate_limit: 5,
  server_error: 4,
  network: 3,
  timeout: 3,
  payment: 2,
  auth: 2,
  quota: 2,
  manual: 0,
};

export type Classification =
  /** status < 400: connector healthy, return the result. */
  | { kind: "success" }
  /** 4xx the connector *answered* (400/422, or 403 when passthrough): the request
   *  is the problem, not the connector — return immediately, no advance. */
  | { kind: "request_error"; status: number }
  /** 404 / model-not-found: a `model_map`/routing issue on THIS connector; advance
   *  to the next connector without touching this one's health. */
  | { kind: "not_found"; status: number }
  /** A thrown error that is NOT a connector-health failure (capability /
   *  not-implemented / bad-request) — surface it immediately; another connector
   *  would throw the same, and cooling this one would be wrong. */
  | { kind: "surface" }
  /** An availability failure: mark the connector and advance. */
  | { kind: "failure"; reason: ConnectorReason; hard: boolean; severity: number };

function failure(reason: ConnectorReason): Classification {
  return { kind: "failure", reason, hard: isHardReason(reason), severity: SEVERITY[reason] };
}

/** Classify an upstream HTTP status (+ optional body text for 429 quota markers). */
export function classifyStatus(status: number, bodyText: string, opts: ClassifyOptions): Classification {
  if (status < 400) return { kind: "success" };
  if (status === 401) return failure("auth");
  if (status === 402) return failure("payment");
  if (status === 403) return opts.treat403As === "down" ? failure("auth") : { kind: "request_error", status };
  if (status === 404) return { kind: "not_found", status };
  if (status === 429) {
    return matchesQuota(bodyText, opts.quotaMarkers) ? failure("quota") : failure("rate_limit");
  }
  if (status >= 500) return failure("server_error");
  // Other 4xx (400, 422, …): the connector answered a bad request — surface it,
  // don't advance (another connector would reject it identically).
  return { kind: "request_error", status };
}

/**
 * Classify a thrown upstream error. Timeouts and network errors are soft
 * availability failures (cool the connector). Other `FusionError`s — capability /
 * not-implemented (e.g. native streaming) / bad-request — are NOT connector-health
 * problems and must be surfaced, not cooled (every connector would throw the same).
 * An unknown thrown value is treated as a network failure.
 */
export function classifyThrown(err: unknown): Classification {
  if (err instanceof UpstreamTimeoutError) return failure("timeout");
  if (err instanceof UpstreamNetworkError) return failure("network");
  if (err instanceof FusionError) return { kind: "surface" };
  return failure("network");
}

function matchesQuota(bodyText: string, markers: string[]): boolean {
  if (markers.length === 0 || bodyText.length === 0) return false;
  const hay = bodyText.toLowerCase();
  return markers.some((m) => hay.includes(m.toLowerCase()));
}

/** Best-effort text of a json/text upstream body, for quota-marker matching. */
export function bodyText(data: unknown): string {
  if (data == null) return "";
  if (typeof data === "string") return data;
  try {
    return JSON.stringify(data);
  } catch {
    return "";
  }
}
