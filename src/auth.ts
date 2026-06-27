import type { MiddlewareHandler } from "hono";

/**
 * Optional client-token auth.
 *
 * If a token is configured (via `server.auth_token_env` resolved to a non-empty
 * value), every request must carry `Authorization: Bearer <token>`; mismatches
 * get 401. If no token is configured, requests are allowed (localhost
 * single-user) — the startup warning is emitted by the entrypoint.
 *
 * The upstream Ollama key is injected server-side by the OllamaClient and is
 * never taken from the client header.
 */
export function createAuthMiddleware(getToken: () => string | undefined): MiddlewareHandler {
  return async (c, next) => {
    const token = getToken();
    if (!token) {
      await next();
      return;
    }
    const header = c.req.header("authorization") ?? "";
    const match = /^Bearer\s+(.+)$/i.exec(header);
    const provided = match?.[1];
    if (!provided || provided !== token) {
      return c.json(
        { error: { message: "invalid or missing client token", type: "authentication_error", code: null } },
        401,
      );
    }
    await next();
    return;
  };
}
