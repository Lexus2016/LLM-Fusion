/**
 * Typed errors + OpenAI-style error-body mapping.
 *
 * Every error carries the HTTP status it maps to and an OpenAI `error.type`
 * string. `toErrorResponse` converts any thrown value into an OpenAI-compatible
 * error `Response`.
 */

export class FusionError extends Error {
  constructor(
    message: string,
    public readonly httpStatus: number,
    public readonly errorType: string,
  ) {
    super(message);
    this.name = new.target.name;
  }
}

/** 400 — malformed request body or unparseable input. */
export class BadRequestError extends FusionError {
  constructor(message: string) {
    super(message, 400, "invalid_request_error");
  }
}

/** 404 — unknown virtual model name. */
export class NotFoundError extends FusionError {
  constructor(message: string) {
    super(message, 404, "not_found_error");
  }
}

/** 401 — missing/invalid client token when one is configured. */
export class AuthError extends FusionError {
  constructor(message: string) {
    super(message, 401, "authentication_error");
  }
}

/** 400 — request cannot be served by the resolved target (e.g. image to non-vision model). */
export class CapabilityError extends FusionError {
  constructor(message: string) {
    super(message, 400, "capability_error");
  }
}

/** 501 — strategy not implemented in the current phase. */
export class NotImplementedError extends FusionError {
  constructor(message: string) {
    super(message, 501, "not_implemented");
  }
}

/**
 * 501 — the native `/api/chat` backend was selected for a streaming request but
 * native NDJSON streaming is not yet wired (deferred from the vision phase). The
 * native non-stream path and the OpenAI-compat streaming path are both live.
 */
export class NativeStreamingNotImplementedError extends FusionError {
  constructor(message: string) {
    super(message, 501, "native_streaming_not_implemented");
  }
}

/** 504 — upstream call exceeded the configured (sub-182 s) timeout. */
export class UpstreamTimeoutError extends FusionError {
  constructor(message: string) {
    super(message, 504, "upstream_timeout");
  }
}

/** 502 — upstream unreachable / network failure / discovery failure. */
export class UpstreamNetworkError extends FusionError {
  constructor(message: string) {
    super(message, 502, "upstream_unavailable");
  }
}

/** 502 — every member of a failover chain was attempted and failed. */
export class AllMembersFailedError extends FusionError {
  constructor(message: string) {
    super(message, 502, "upstream_unavailable");
  }
}

/**
 * 503 — the per-model circuit breaker is open (the model is fast-failing while
 * it cools down). In a failover chain this is raised only when *every* member
 * is open and therefore skipped.
 */
export class CircuitOpenError extends FusionError {
  constructor(message: string) {
    super(message, 503, "circuit_open");
  }
}

export function jsonError(status: number, message: string, type: string): Response {
  const body = JSON.stringify({ error: { message, type, code: null } });
  return new Response(body, { status, headers: { "content-type": "application/json" } });
}

export function toErrorResponse(err: unknown): Response {
  if (err instanceof FusionError) {
    return jsonError(err.httpStatus, err.message, err.errorType);
  }
  const message = err instanceof Error ? err.message : "internal server error";
  return jsonError(500, message, "internal_error");
}
