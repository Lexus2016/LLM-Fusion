import type { MiddlewareHandler } from "hono";
import { timingSafeEqual } from "node:crypto";

/**
 * Constant-time client-token comparison. The length guard returns early on a
 * size mismatch (token length may leak, the secret bytes do not) so
 * `timingSafeEqual` — which throws on unequal-length buffers — is only reached
 * with equal lengths. This removes the byte-by-byte timing side-channel of a
 * plain `===`/`!==` string compare.
 */
function tokensMatch(provided: string, token: string): boolean {
  const a = Buffer.from(provided);
  const b = Buffer.from(token);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

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
    let provided: string | undefined;
    const authHeader = c.req.header("authorization") ?? "";
    const bearerMatch = /^Bearer\s+(.+)$/i.exec(authHeader);
    if (bearerMatch) {
      provided = bearerMatch[1];
    } else {
      // Anthropic SDK / Claude Code sends the key in `x-api-key`.
      provided = c.req.header("x-api-key");
    }
    if (!provided || !tokensMatch(provided, token)) {
      return c.json(
        { error: { message: "invalid or missing client token", type: "authentication_error", code: null } },
        401,
      );
    }
    await next();
    return;
  };
}
