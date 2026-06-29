/**
 * Remove hop-by-hop and length/encoding headers that describe the *upstream*
 * body, not the body we are about to return. When the proxy mutates a response
 * (injecting `usage` into JSON or wrapping an SSE transform), the upstream
 * `content-length` / `content-encoding` / `transfer-encoding` become stale:
 * clients honoring them truncate or try to gunzip a non-gzip stream. Delete
 * them before returning a mutated body; the platform re-derives them.
 */
export function stripHopByHopHeaders(headers: Headers): void {
  headers.delete("content-length");
  headers.delete("content-encoding");
  headers.delete("transfer-encoding");
}

/**
 * True when an error is an abort/cancellation (fetch throws `AbortError`,
 * `DOMException` with name `AbortError`, or our own thrown abort). Used to keep
 * client disconnects from being recorded as upstream failures and tripping the
 * circuit breaker.
 */
export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}