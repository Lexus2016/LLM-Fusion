import { Hono } from "hono";
import type { Context, MiddlewareHandler } from "hono";
import type { Logger } from "pino";
import { readFile, writeFile, rename, copyFile } from "node:fs/promises";
import { parseDocument, type Document } from "yaml";
import type { ZodError } from "zod";
import type { Config } from "../config";
import { ConfigSchema } from "../config";

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
}

function friendly(err: ZodError): string {
  const first = err.issues[0];
  if (!first) return "invalid configuration";
  const path = first.path.join(".") || "config";
  return `${path}: ${first.message}`;
}

export function createConfigEditorApp(deps: ConfigEditorDeps): Hono {
  const app = new Hono();

  // Current editable config (structured; no secret values — only env-var names).
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
      providers: cfg.providers ?? null,
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
    const res = await applyEdit((doc) => doc.setIn(["providers", id], body));
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

  return app;
}
