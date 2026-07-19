import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import type { Context, MiddlewareHandler } from "hono";
import type { Logger } from "pino";
import { readFile, writeFile, rename, copyFile } from "node:fs/promises";
import { parseDocument, type Document } from "yaml";
import { z, type ZodError } from "zod";
import type { Config } from "../config";
import { ConfigSchema } from "../config";

/**
 * Inbound body cap for the admin surface (config edits are small, but an
 * unbounded body is a memory-exhaustion DoS on any non-loopback deployment).
 * Same generous ceiling as the JSON API: vision-sized payloads, 50 MB.
 */
export const MAX_BODY_BYTES = 50 * 1024 * 1024;

/** Strict loopback hostnames (NOT a `127.` prefix — `127.evil.com` must fail; and
 *  proper octets — `127.999.x.x` must fail). */
const LOOPBACK_HOST = new RegExp(
  `^(127\\.(25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.(25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.(25[0-5]|2[0-4]\\d|1?\\d?\\d)|localhost|::1|0:0:0:0:0:0:0:1)$`,
  "i",
);

/** Hostname of a `Host`/`Origin` `host[:port]`, IPv6 brackets stripped. Returns
 *  undefined for a malformed bracketed host (`[::1]junk`) so the caller fails
 *  closed instead of matching on a parser-differential prefix. */
function hostnameOf(hostHeader: string): string | undefined {
  if (hostHeader.startsWith("[")) {
    const end = hostHeader.indexOf("]");
    if (end === -1) return undefined; // no closing bracket
    const after = hostHeader.slice(end + 1);
    if (after !== "" && !after.startsWith(":")) return undefined; // junk after ]
    return hostHeader.slice(1, end);
  }
  const colon = hostHeader.indexOf(":");
  return colon === -1 ? hostHeader : hostHeader.slice(0, colon);
}

export interface AdminGuardOptions {
  /**
   * Whether client auth is enforced (a token resolves). When it is NOT — the
   * unauthenticated default AND the `FUSION_ALLOW_OPEN` deployment — the admin
   * surface is loopback-only: the `Host` must be a loopback name, and a
   * MISSING/malformed Host fails closed. This is the check that actually stops
   * DNS rebinding, and it also keeps an unauthenticated admin plane off the
   * network even under `FUSION_ALLOW_OPEN` (set a token for remote admin, or
   * have your reverse proxy rewrite Host to `localhost`). When auth IS enforced,
   * a non-loopback Host is allowed (the token is the real gate; a rebinding page
   * cannot read another origin's token).
   */
  authEnforced?: () => boolean;
}

/**
 * CSRF + DNS-rebinding guard for the whole `/admin/*` surface.
 *
 *  - **Host pinning (anti-rebinding).** When auth is off, the `Host` must be a
 *    loopback name; a missing/malformed Host fails CLOSED (403). This is the
 *    check that actually defeats DNS rebinding: a rebound `evil.tld → 127.0.0.1`
 *    page still sends `Host: evil.tld`, rejected even though Origin==Host. (An
 *    Origin==Host equality check alone does NOT stop rebinding — both headers
 *    carry the attacker's own name after the rebind.) Failing closed on a
 *    missing Host keeps the guard's "loopback-only unless authenticated"
 *    contract true against non-browser clients too.
 *  - **Origin==Host (anti-CSRF).** A present `Origin` must match `Host` — kills
 *    classic cross-origin writes (a cross-origin fetch carries a foreign Origin).
 *  - **content-type (anti-CSRF).** A mutating request carrying a body must be
 *    `application/json` — a browser `no-cors`/form write can only send a "simple"
 *    content-type; a JSON content-type would trigger a preflight the attacker
 *    page cannot pass. `transfer-encoding` and a nonzero/malformed `content-length`
 *    both count as a body (closes the chunked / bogus-length bypasses). Bodyless
 *    mutations (connector actions) are covered by the Host + Origin checks above.
 */
export function makeAdminApiGuard(opts: AdminGuardOptions = {}): MiddlewareHandler {
  const authEnforced = opts.authEnforced ?? (() => false);
  return async (c, next) => {
    const hostHeader = c.req.header("host");
    // Anti-rebinding: on the unauthenticated admin plane, ONLY a loopback Host is
    // admissible — a rebound attacker page always carries its own (non-loopback)
    // hostname, and a missing/malformed Host (non-browser client on an ALLOW_OPEN
    // deployment) must not slip through either. Skipped when auth is enforced.
    if (!authEnforced()) {
      const hn = hostHeader === undefined ? undefined : hostnameOf(hostHeader);
      if (hn === undefined || !LOOPBACK_HOST.test(hn)) {
        return c.json({ error: "admin API is loopback-only unless authenticated (set a token for remote admin)" }, 403);
      }
    }
    const origin = c.req.header("origin");
    if (origin !== undefined) {
      let originHost: string | undefined;
      try {
        originHost = new URL(origin).host;
      } catch {
        originHost = undefined; // unparseable, e.g. "Origin: null" (sandboxed frame)
      }
      if (originHost === undefined || originHost !== hostHeader) {
        return c.json({ error: "origin does not match the request host" }, 403);
      }
    }
    if (c.req.method === "POST" || c.req.method === "PUT" || c.req.method === "DELETE") {
      const contentType = c.req.header("content-type");
      const contentLength = c.req.header("content-length");
      const hasBody =
        contentType !== undefined ||
        c.req.header("transfer-encoding") !== undefined ||
        (contentLength !== undefined && Number(contentLength) !== 0); // NaN (bogus length) !== 0 → treated as a body
      if (hasBody && (contentType === undefined || !contentType.toLowerCase().startsWith("application/json"))) {
        return c.json({ error: "mutating admin requests with a body require content-type application/json" }, 415);
      }
    }
    await next();
  };
}

/**
 * No-YAML config editor for the panel. Every write edits the on-disk config
 * through the `yaml` Document API (so the operator's COMMENTS are preserved on
 * untouched nodes), validates the WHOLE resulting config with the same zod schema
 * the server boots from, writes a timestamped backup, then replaces the file
 * atomically (temp + rename). The existing config watcher then hot-reloads models;
 * `providers:` changes are re-applied to the live router by the boot wiring.
 *
 * Nothing invalid is ever written: a bad edit returns a friendly error and the
 * file is left untouched.
 */

export interface ConfigEditorDeps {
  getConfig: () => Config;
  configPath: string;
  auth: MiddlewareHandler;
  logger: Logger;
  /** Whether an api-key env var currently resolves (no secret values exposed). */
  envHas: (name: string) => boolean;
  /** Whether client auth is enforced — controls Host pinning (see makeAdminApiGuard). */
  authEnforced?: () => boolean;
  /**
   * Ask a provider group for its live model catalog (for the model picker).
   * Optional: when absent, the picker falls back to free-typed model ids.
   */
  listProviderModels?: (groupId: string, opts: { signal?: AbortSignal }) => Promise<string[]>;
}

function friendly(err: ZodError): string {
  const first = err.issues[0];
  if (!first) return "invalid configuration";
  const path = first.path.join(".") || "config";
  return `${path}: ${first.message}`;
}

/** Placeholder shown in place of every extra_headers value. */
const REDACTED = "•••";

/**
 * Same shape as the configured providers, but with every `extra_headers` VALUE
 * masked (keys preserved — the panel needs to see which headers are set, never
 * what they contain). The write path must undo this masking — see
 * restoreExtraHeaders below.
 */
function redactExtraHeaders(providers: Config["providers"]): Config["providers"] {
  if (!providers) return providers;
  return Object.fromEntries(
    Object.entries(providers).map(([id, g]) => [
      id,
      {
        ...g,
        accounts: g.accounts.map((a) => ({
          ...a,
          extra_headers: a.extra_headers
            ? Object.fromEntries(Object.keys(a.extra_headers).map((k) => [k, REDACTED]))
            : a.extra_headers,
        })),
      },
    ]),
  );
}

/** Loose shape used only to locate accounts/extra_headers inside an incoming provider body. */
const ProviderBodySchema = z
  .object({
    accounts: z
      .array(
        z
          .object({
            id: z.string(),
            extra_headers: z.record(z.string(), z.string()).optional(),
          })
          .passthrough(),
      )
      .optional(),
  })
  .passthrough();

/**
 * The panel edits a REDACTED view of the provider, so a naive write would
 * persist "•••" over the real header values (or drop them entirely — the form
 * has no extra_headers field). Before writing, restore every account's real
 * values from the live config: a missing map means "untouched" (the form can't
 * edit headers), and any value still equal to the placeholder means
 * "unchanged". Genuinely new/changed values pass through. A placeholder with
 * NO real value to restore (stale session, hand-rolled client, new provider)
 * is DROPPED — the mask must never become a live header value. Returns the
 * body unchanged when it doesn't look like a provider object (schema
 * validation downstream reports the real problem).
 */
function restoreExtraHeaders(body: unknown, existing: NonNullable<Config["providers"]>[string] | undefined): unknown {
  const parsed = ProviderBodySchema.safeParse(body);
  if (!parsed.success || !parsed.data.accounts) return body;
  const existingById = new Map((existing?.accounts ?? []).map((a) => [a.id, a]));
  for (const acc of parsed.data.accounts) {
    const prev = existingById.get(acc.id);
    if (!acc.extra_headers) {
      if (prev?.extra_headers) acc.extra_headers = prev.extra_headers;
      continue;
    }
    for (const [k, v] of Object.entries(acc.extra_headers)) {
      if (v !== REDACTED) continue;
      const real = prev?.extra_headers?.[k];
      if (real !== undefined) acc.extra_headers[k] = real;
      else delete acc.extra_headers[k]; // never persist the mask itself
    }
  }
  return parsed.data;
}

export function createConfigEditorApp(deps: ConfigEditorDeps): Hono {
  const app = new Hono();

  app.use("/admin/*", makeAdminApiGuard({ authEnforced: deps.authEnforced }));
  app.use(
    "/admin/*",
    bodyLimit({
      maxSize: MAX_BODY_BYTES,
      onError: (c) => c.json({ error: "request body too large" }, 413),
    }),
  );

  // Short-lived cache of provider model catalogs. The catalog is stable across a
  // panel session, so this spares the upstream a round-trip every form open while
  // still refreshing within a minute. Keyed by group id.
  const modelCache = new Map<string, { at: number; models: string[] }>();
  const MODEL_CACHE_TTL_MS = 60_000;

  // Current editable config (structured; no secret values — env-var NAMES only,
  // and extra_headers values masked: some providers auth via custom headers).
  app.get("/admin/config", deps.auth, (c) => {
    const cfg = deps.getConfig();
    const envKnown: Record<string, boolean> = {};
    if (cfg.providers) {
      for (const g of Object.values(cfg.providers)) {
        for (const a of g.accounts) envKnown[a.api_key_env] = deps.envHas(a.api_key_env);
      }
    }
    if (cfg.upstream.api_key_env) envKnown[cfg.upstream.api_key_env] = deps.envHas(cfg.upstream.api_key_env);
    return c.json({
      providers: redactExtraHeaders(cfg.providers) ?? null,
      upstreamLegacy: { base_url: cfg.upstream.base_url ?? null, api_key_env: cfg.upstream.api_key_env ?? null },
      models: cfg.models,
      defaults: cfg.defaults,
      envKnown,
    });
  });

  async function applyEdit(
    mutate: (doc: Document.Parsed) => void,
  ): Promise<{ ok: true } | { ok: false; error: string }> {
    let text: string;
    try {
      text = await readFile(deps.configPath, "utf8");
    } catch (e) {
      return { ok: false, error: `cannot read config: ${e instanceof Error ? e.message : String(e)}` };
    }
    const doc = parseDocument(text);
    mutate(doc);
    const parsed = ConfigSchema.safeParse(doc.toJSON());
    if (!parsed.success) return { ok: false, error: friendly(parsed.error) };
    try {
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      await copyFile(deps.configPath, `${deps.configPath}.bak-${stamp}`);
      const tmp = `${deps.configPath}.tmp-${process.pid}`;
      await writeFile(tmp, doc.toString(), "utf8");
      await rename(tmp, deps.configPath);
    } catch (e) {
      return { ok: false, error: `write failed: ${e instanceof Error ? e.message : String(e)}` };
    }
    return { ok: true };
  }

  async function readBody(c: Context): Promise<unknown | undefined> {
    try {
      return await c.req.json();
    } catch {
      return undefined;
    }
  }

  app.put("/admin/config/models/:name", deps.auth, async (c) => {
    const name = c.req.param("name");
    const body = await readBody(c);
    if (body === undefined) return c.json({ error: "request body must be valid JSON" }, 400);
    const res = await applyEdit((doc) => doc.setIn(["models", name], body));
    if (!res.ok) return c.json({ error: res.error }, 400);
    deps.logger.info({ model: name }, "config: model saved via panel");
    return c.json({ ok: true });
  });

  app.delete("/admin/config/models/:name", deps.auth, async (c) => {
    const name = c.req.param("name");
    const res = await applyEdit((doc) => {
      doc.deleteIn(["models", name]);
    });
    if (!res.ok) return c.json({ error: res.error }, 400);
    deps.logger.info({ model: name }, "config: model deleted via panel");
    return c.json({ ok: true });
  });

  app.put("/admin/config/providers/:id", deps.auth, async (c) => {
    const id = c.req.param("id");
    const body = await readBody(c);
    if (body === undefined) return c.json({ error: "request body must be valid JSON" }, 400);
    // The panel edited a redacted view — put the real extra_headers values back
    // before they hit the disk (see restoreExtraHeaders).
    const restored = restoreExtraHeaders(body, deps.getConfig().providers?.[id]);
    const res = await applyEdit((doc) => doc.setIn(["providers", id], restored));
    if (!res.ok) return c.json({ error: res.error }, 400);
    deps.logger.info({ provider: id }, "config: provider saved via panel");
    return c.json({ ok: true });
  });

  app.delete("/admin/config/providers/:id", deps.auth, async (c) => {
    const id = c.req.param("id");
    const res = await applyEdit((doc) => {
      doc.deleteIn(["providers", id]);
    });
    if (!res.ok) return c.json({ error: res.error }, 400);
    deps.logger.info({ provider: id }, "config: provider deleted via panel");
    return c.json({ ok: true });
  });

  // Live model catalog for a provider group — powers the no-typo model picker.
  // Returns { models: [...] } (possibly cached). On any upstream failure it still
  // returns 200 with an empty list + a note, so the form degrades to free-text
  // rather than blocking the operator.
  app.get("/admin/config/providers/:id/models", deps.auth, async (c) => {
    const id = c.req.param("id");
    if (!deps.listProviderModels) {
      return c.json({ models: [], note: "live model discovery is not available" });
    }
    const cached = modelCache.get(id);
    if (cached && Date.now() - cached.at < MODEL_CACHE_TTL_MS) {
      return c.json({ models: cached.models, cached: true });
    }
    try {
      const models = await deps.listProviderModels(id, { signal: AbortSignal.timeout(10_000) });
      modelCache.set(id, { at: Date.now(), models });
      return c.json({ models });
    } catch (e) {
      const note = e instanceof Error ? e.message : String(e);
      deps.logger.warn({ provider: id, err: note }, "config: live model discovery failed");
      return c.json({ models: [], note });
    }
  });

  return app;
}
