import type { MiddlewareHandler } from "hono";
import { createHash, timingSafeEqual } from "node:crypto";

/**
 * Constant-time client-token comparison. The length guard returns early on a
 * size mismatch (token length may leak, the secret bytes do not) so
 * `timingSafeEqual` — which throws on unequal-length buffers — is only reached
 * with equal lengths. This removes the byte-by-byte timing side-channel of a
 * plain `===`/`!==` string compare.
 */
function tokensMatch(provided: string, token: string): boolean {
  const hashA = createHash("sha256").update(provided).digest();
  const hashB = createHash("sha256").update(token).digest();
  return timingSafeEqual(hashA, hashB);
}

/**
 * Resolve the client token from the configured env var. Three states:
 *  - no `auth_token_env` configured → `undefined` (auth intentionally off);
 *  - configured and set → the value (which may be "" — the middleware
 *    hard-errors on that below);
 *  - configured but UNSET (e.g. a misnamed/typo'd var) → "" as well: fail
 *    CLOSED through the same "configured but empty" 500 path instead of
 *    silently disabling auth while the operator believes it is on.
 */
export function resolveAuthToken(
  envName: string | undefined,
  env: Record<string, string | undefined>,
): string | undefined {
  if (!envName) return undefined;
  return env[envName] ?? "";
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
    if (token === undefined) {
      // No token configured (localhost single-user) — the startup warning is
      // emitted by the entrypoint.
      await next();
      return;
    }
    if (token === "") {
      // A configured-but-empty token is a misconfiguration (e.g. an env var set
      // to "" in CI/Docker). Treat it as a hard error rather than silently
      // disabling auth — otherwise the proxy would run open with no startup sign.
      return c.json(
        { error: { message: "client auth token configured but empty", type: "configuration_error", code: null } },
        500,
      );
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
