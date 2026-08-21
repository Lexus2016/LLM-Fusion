import { describe, it, expect } from "vitest";
import { mkdtemp, writeFile, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { MiddlewareHandler } from "hono";
import pino from "pino";
import { createConfigEditorApp, createSerializer, BACKUP_RETENTION } from "../src/panel/config_editor";
import { loadConfigFile } from "../src/config";

const logger = pino({ level: "silent" });
const openAuth: MiddlewareHandler = async (_c, next) => {
  await next();
};

const BASE = `# my config — keep this comment
upstream:
  api_mode: openai
providers:
  ollama-cloud:
    type: ollama
    base_url: https://ollama.com
    accounts:
      - id: acc-1   # first account — keep this too
        api_key_env: OLLAMA_API_KEY
models:
  fast-glm:
    strategy: single
    provider: ollama-cloud
    target: glm-5.2
`;

async function setup() {
  const dir = await mkdtemp(join(tmpdir(), "fusion-cfg-"));
  const path = join(dir, "fusion.yaml");
  await writeFile(path, BASE, "utf8");
  const cfg = await loadConfigFile(path);
  const app = createConfigEditorApp({
    getConfig: () => cfg,
    configPath: path,
    auth: openAuth,
    logger,
    envHas: () => true,
    authEnforced: () => true, // these tests exercise the editor, not the transport Host guard
  });
  return { app, path, dir };
}

function put(app: Awaited<ReturnType<typeof setup>>["app"], p: string, body: unknown) {
  return app.request(p, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}

describe("config editor", () => {
  it("GET /admin/config returns providers + models + env presence", async () => {
    const { app } = await setup();
    const res = await app.request("/admin/config");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Object.keys(body.providers)).toContain("ollama-cloud");
    expect(Object.keys(body.models)).toContain("fast-glm");
    expect(body.envKnown.OLLAMA_API_KEY).toBe(true);
  });

  it("creates a valid model and hot-persists it", async () => {
    const { app, path } = await setup();
    const res = await put(app, "/admin/config/models/fast-kimi", { strategy: "single", target: "kimi-k2.7-code" });
    expect(res.status).toBe(200);
    const cfg = await loadConfigFile(path);
    expect(Object.keys(cfg.models)).toContain("fast-kimi");
  });

  it("415 on a mutating request with a non-JSON body", async () => {
    const { app } = await setup();
    const res = await app.request("/admin/config/models/fast-kimi", {
      method: "PUT",
      headers: { "content-type": "text/plain" },
      body: "strategy=single",
    });
    expect(res.status).toBe(415);
  });

  it("GET /admin/config masks extra_headers values (keys preserved)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "fusion-cfg-redact-"));
    const path = join(dir, "fusion.yaml");
    await writeFile(
      path,
      `upstream:
  base_url: https://mock.test
  api_key_env: X
providers:
  openrouter:
    type: openai-compat
    base_url: https://openrouter.ai/api/v1
    accounts:
      - id: or-1
        api_key_env: OPENROUTER_API_KEY
        extra_headers:
          authorization: Key super-secret-value
          x-title: My App
models:
  m:
    strategy: single
    provider: openrouter
    target: x
`,
      "utf8",
    );
    const cfg = await loadConfigFile(path);
    const app = createConfigEditorApp({
      getConfig: () => cfg,
      configPath: path,
      auth: openAuth,
      logger,
      envHas: () => true,
      authEnforced: () => true, // exercises the editor, not the transport Host guard
    });
    const res = await app.request("/admin/config");
    expect(res.status).toBe(200);
    const body = await res.json();
    const account = body.providers.openrouter.accounts[0];
    expect(account.extra_headers).toEqual({ authorization: "•••", "x-title": "•••" });
    expect(JSON.stringify(body)).not.toContain("super-secret-value");
    // Write path is untouched: the on-disk file still holds the real value.
    expect(await readFile(path, "utf8")).toContain("super-secret-value");
  });

  it("PUT of the redacted provider view restores the real extra_headers values", async () => {
    const dir = await mkdtemp(join(tmpdir(), "fusion-cfg-roundtrip-"));
    const path = join(dir, "fusion.yaml");
    await writeFile(
      path,
      `upstream:
  base_url: https://mock.test
  api_key_env: X
providers:
  openrouter:
    type: openai-compat
    base_url: https://openrouter.ai/api/v1
    accounts:
      - id: or-1
        api_key_env: OPENROUTER_API_KEY
        treat_403_as: down
        model_map:
          x: real-x
        extra_headers:
          authorization: Key super-secret-value
      - id: or-2
        api_key_env: OPENROUTER_API_KEY_2
        extra_headers:
          x-title: other-secret
models:
  m:
    strategy: single
    provider: openrouter
    target: x
`,
      "utf8",
    );
    const cfg = await loadConfigFile(path);
    const app = createConfigEditorApp({
      getConfig: () => cfg,
      configPath: path,
      auth: openAuth,
      logger,
      envHas: () => true,
      authEnforced: () => true, // exercises the editor, not the transport Host guard
    });
    // What the panel does: GET the (redacted) provider, edit an unrelated field,
    // PUT the whole group back — with one account's extra_headers dropped
    // entirely (the form has no such field). Also add a brand-new account whose
    // only header value is the placeholder (nothing real to restore).
    const view = (await (await app.request("/admin/config")).json()).providers.openrouter;
    view.base_url = "https://openrouter.ai/api/v2";
    delete view.accounts[1].extra_headers;
    view.accounts.push({ id: "or-3", api_key_env: "K3", extra_headers: { "x-stale": "•••" } });
    const res = await put(app, "/admin/config/providers/openrouter", view);
    expect(res.status).toBe(200);
    const text = await readFile(path, "utf8");
    expect(text).toContain("super-secret-value"); // masked value restored
    expect(text).toContain("other-secret"); // dropped map restored
    expect(text).toContain("api/v2"); // the actual edit landed
    expect(text).toContain("real-x"); // account-level passthrough (model_map) survives
    expect(text).toContain("treat_403_as"); // account-level passthrough survives
    expect(text).not.toContain("•••"); // no placeholder ever persisted
    expect(text).not.toContain("x-stale"); // unrestorable placeholder dropped, not written
  });

  it("rejects an invalid model and leaves the file untouched", async () => {
    const { app, path } = await setup();
    // fusion without judge/synth is invalid
    const res = await put(app, "/admin/config/models/broken", { strategy: "fusion", panel: ["a"] });
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBeTruthy();
    const text = await readFile(path, "utf8");
    expect(text).not.toContain("broken"); // file unchanged
  });

  it("preserves the operator's comments on an edit", async () => {
    const { app, path } = await setup();
    await put(app, "/admin/config/models/fast-kimi", { strategy: "single", target: "kimi-k2.7-code" });
    const text = await readFile(path, "utf8");
    expect(text).toContain("keep this comment");
    expect(text).toContain("keep this too");
  });

  it("writes a timestamped backup before each edit", async () => {
    const { app, dir } = await setup();
    await put(app, "/admin/config/models/fast-kimi", { strategy: "single", target: "kimi-k2.7-code" });
    const files = await readdir(dir);
    expect(files.some((f) => f.startsWith("fusion.yaml.bak-"))).toBe(true);
  });

  it("deletes a model", async () => {
    const { app, path } = await setup();
    const res = await app.request("/admin/config/models/fast-glm", { method: "DELETE" });
    expect(res.status).toBe(200);
    const cfg = await loadConfigFile(path);
    expect(Object.keys(cfg.models)).not.toContain("fast-glm");
  });

  it("creates a provider group and rejects a duplicate account id", async () => {
    const { app, path } = await setup();
    const ok = await put(app, "/admin/config/providers/openrouter", {
      type: "openai-compat",
      base_url: "https://openrouter.ai/api/v1",
      accounts: [{ id: "or-1", api_key_env: "OPENROUTER_API_KEY" }],
    });
    expect(ok.status).toBe(200);
    const cfg = await loadConfigFile(path);
    expect(Object.keys(cfg.providers ?? {})).toContain("openrouter");

    // duplicate account id across providers -> rejected, file unchanged
    const dup = await put(app, "/admin/config/providers/another", {
      type: "ollama",
      base_url: "https://x.com",
      accounts: [{ id: "acc-1", api_key_env: "K" }],
    });
    expect(dup.status).toBe(400);
    expect((await dup.json()).error).toMatch(/duplicate account id/);
  });
});

describe("config editor — global settings & restart (Task 2)", () => {
  it("GET /admin/config surfaces server / upstream / defaults / pricing / overrides", async () => {
    const { app } = await setup();
    const body = await (await app.request("/admin/config")).json();
    // server + full upstream (not just base_url/api_key_env) are now editable.
    expect(body.server.port).toBe(8080);
    expect(body.server.bind).toBe("127.0.0.1");
    expect(body.upstream.api_mode).toBe("openai");
    expect(body.upstream.request_timeout_s).toBe(170); // schema default surfaced
    expect(body.defaults.min_panel_success).toBe(1);
    expect(body.pricing).toEqual({}); // absent pricing -> {} (never undefined for the form)
    expect(body.overrides).toEqual({});
  });

  it("PUT /admin/config/settings/server changes the port and persists", async () => {
    const { app, path } = await setup();
    const res = await put(app, "/admin/config/settings/server", { bind: "127.0.0.1", port: 9090 });
    expect(res.status).toBe(200);
    const cfg = await loadConfigFile(path);
    expect(cfg.server.port).toBe(9090);
  });

  it("PUT /admin/config/settings/defaults changes min_panel_success and persists", async () => {
    const { app, path } = await setup();
    const res = await put(app, "/admin/config/settings/defaults", { min_panel_success: 2, judge_timeout_s: 45 });
    expect(res.status).toBe(200);
    const cfg = await loadConfigFile(path);
    expect(cfg.defaults.min_panel_success).toBe(2);
    expect(cfg.defaults.judge_timeout_s).toBe(45);
  });

  it("rejects an out-of-range port and leaves the file untouched", async () => {
    const { app, path } = await setup();
    const res = await put(app, "/admin/config/settings/server", { bind: "127.0.0.1", port: 70000 });
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBeTruthy();
    expect(await readFile(path, "utf8")).not.toContain("70000"); // never landed
  });

  it("404s a section that is not in the settings allowlist", async () => {
    const { app } = await setup();
    // `models` has its own per-item route; the generic settings route must refuse it.
    const res = await put(app, "/admin/config/settings/models", { anything: true });
    expect(res.status).toBe(404);
  });

  it("preserves operator comments when saving a settings section", async () => {
    const { app, path } = await setup();
    await put(app, "/admin/config/settings/server", { bind: "127.0.0.1", port: 9091 });
    expect(await readFile(path, "utf8")).toContain("keep this comment");
  });

  it("POST /admin/restart returns 501 when no restart handler is wired", async () => {
    const { app } = await setup(); // setup() wires no requestRestart
    const res = await app.request("/admin/restart", { method: "POST" });
    expect(res.status).toBe(501);
  });

  it("POST /admin/restart invokes the handler and reports restarting when wired", async () => {
    const dir = await mkdtemp(join(tmpdir(), "fusion-cfg-restart-"));
    const path = join(dir, "fusion.yaml");
    await writeFile(path, BASE, "utf8");
    const cfg = await loadConfigFile(path);
    let restarted = 0;
    const app = createConfigEditorApp({
      getConfig: () => cfg,
      configPath: path,
      auth: openAuth,
      logger,
      envHas: () => true,
      authEnforced: () => true,
      requestRestart: () => {
        restarted++;
      },
    });
    const res = await app.request("/admin/restart", { method: "POST" });
    expect(res.status).toBe(200);
    expect((await res.json()).restarting).toBe(true);
    expect(restarted).toBe(1);
  });
});

describe("config editor — concurrent saves & backup retention", () => {
  it("two concurrent saves both land; neither is lost and the file stays valid", async () => {
    const { app, path } = await setup();
    // Fired without awaiting the first: the two handlers interleave at every
    // await inside applyEdit. Unserialized, both read the same base text and the
    // later rename discards the earlier model.
    const [a, b] = await Promise.all([
      put(app, "/admin/config/models/model-a", { strategy: "single", target: "t-a" }),
      put(app, "/admin/config/models/model-b", { strategy: "single", target: "t-b" }),
    ]);
    expect([a.status, b.status]).toEqual([200, 200]);
    const text = await readFile(path, "utf8");
    expect(text).toContain("keep this comment"); // still comment-preserving
    const cfg = await loadConfigFile(path); // parses AND validates
    expect(Object.keys(cfg.models)).toContain("model-a");
    expect(Object.keys(cfg.models)).toContain("model-b");
    expect(Object.keys(cfg.models)).toContain("fast-glm"); // pre-existing model untouched
    // No temp file left behind, and no two writers shared one temp path.
    expect((await readdir(dirname(path))).some((f) => f.includes(".tmp-"))).toBe(false);
  });

  it("eight concurrent saves all land", async () => {
    const { app, path } = await setup();
    const names = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"];
    const results = await Promise.all(
      names.map((n) => put(app, `/admin/config/models/${n}`, { strategy: "single", target: `t-${n}` })),
    );
    expect(results.map((r) => r.status)).toEqual(names.map(() => 200));
    const cfg = await loadConfigFile(path);
    for (const n of names) expect(Object.keys(cfg.models)).toContain(n);
  });

  it("prunes backups to the newest BACKUP_RETENTION after a save", async () => {
    const { app, path, dir } = await setup();
    // 15 pre-existing backups, oldest first (ISO stamps sort chronologically).
    const stale = Array.from({ length: 15 }, (_, i) => `fusion.yaml.bak-2000-01-01T00-00-${String(i).padStart(2, "0")}-000Z`);
    for (const f of stale) await writeFile(join(dir, f), "stale\n", "utf8");

    const res = await put(app, "/admin/config/models/fast-kimi", { strategy: "single", target: "kimi-k2.7-code" });
    expect(res.status).toBe(200);

    const backups = (await readdir(dir)).filter((f) => f.startsWith("fusion.yaml.bak-")).sort();
    expect(backups.length).toBe(BACKUP_RETENTION);
    // The 6 oldest are gone, the newest stale ones and the fresh backup remain.
    expect(backups).not.toContain(stale[0]);
    expect(backups).not.toContain(stale[5]);
    expect(backups).toContain(stale[6]);
    expect(backups).toContain(stale[14]);
    expect(backups.at(-1)).not.toBe(stale[14]); // the save's own backup is the newest
    // Pruning never touches the config itself.
    expect(await readFile(path, "utf8")).toContain("fast-kimi");
  });

  // The pruner must only ever delete files it wrote itself. Operators keep
  // hand-named backups next to ours (`fusion.yaml.bak-legacy-…`); matching on the
  // bare `.bak-` prefix made them deletion candidates, and because `'l' > '2'` a
  // lexicographic sort ranked them as the NEWEST — so they were retained while
  // genuinely newer ISO-stamped backups were deleted first.
  const HAND_NAMED = [
    "fusion.yaml.bak-legacy-20260715-164854",
    "fusion.yaml.bak-manual-keep-forever",
    ...Array.from({ length: 13 }, (_, i) => `fusion.yaml.bak-manual-${String(i).padStart(2, "0")}`),
  ];
  const isoBackups = (files: string[]) =>
    files.filter((f) => /^fusion\.yaml\.bak-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z$/.test(f)).sort();

  it("never prunes hand-named backups, however many there are", async () => {
    const { app, dir } = await setup();
    // 15 hand-made backups — well past BACKUP_RETENTION on their own.
    for (const f of HAND_NAMED) await writeFile(join(dir, f), "operator's own\n", "utf8");

    const res = await put(app, "/admin/config/models/fast-kimi", { strategy: "single", target: "kimi-k2.7-code" });
    expect(res.status).toBe(200);

    const files = await readdir(dir);
    // Every single one survives — none of them is ours to delete.
    for (const f of HAND_NAMED) expect(files).toContain(f);
    // ...and the save's own backup was written and (being alone) not pruned either.
    expect(isoBackups(files).length).toBe(1);
  });

  it("prunes only the ISO-stamped backups when both kinds are present", async () => {
    const { app, path, dir } = await setup();
    const keepers = ["fusion.yaml.bak-legacy-20260715-164854", "fusion.yaml.bak-manual-keep-forever"];
    for (const f of keepers) await writeFile(join(dir, f), "operator's own\n", "utf8");
    // 12 real backups, oldest first.
    const stale = Array.from(
      { length: 12 },
      (_, i) => `fusion.yaml.bak-2000-01-01T00-00-${String(i).padStart(2, "0")}-000Z`,
    );
    for (const f of stale) await writeFile(join(dir, f), "stale\n", "utf8");

    const res = await put(app, "/admin/config/models/fast-kimi", { strategy: "single", target: "kimi-k2.7-code" });
    expect(res.status).toBe(200);

    const files = await readdir(dir);
    // Hand-named files neither deleted nor counted toward retention.
    for (const f of keepers) expect(files).toContain(f);
    const iso = isoBackups(files);
    expect(iso.length).toBe(BACKUP_RETENTION);
    // 12 stale + 1 fresh = 13 ours; the 3 oldest ISO ones went, the rest stayed.
    expect(iso).not.toContain(stale[0]);
    expect(iso).not.toContain(stale[2]);
    expect(iso).toContain(stale[3]);
    expect(iso).toContain(stale[11]);
    expect(iso.at(-1)).not.toBe(stale[11]); // the save's own backup is the newest
    expect(await readFile(path, "utf8")).toContain("fast-kimi");
  });

  it("a rejected edit neither blocks nor corrupts the next edit", async () => {
    const { app, path } = await setup();
    // Invalid (fusion without judge/synth) racing a valid save.
    const [bad, good] = await Promise.all([
      put(app, "/admin/config/models/broken", { strategy: "fusion", panel: ["a"] }),
      put(app, "/admin/config/models/fine", { strategy: "single", target: "t" }),
    ]);
    expect(bad.status).toBe(400);
    expect(good.status).toBe(200);
    // And the queue still works afterwards.
    const after = await put(app, "/admin/config/models/later", { strategy: "single", target: "t2" });
    expect(after.status).toBe(200);
    const cfg = await loadConfigFile(path);
    expect(Object.keys(cfg.models)).toContain("fine");
    expect(Object.keys(cfg.models)).toContain("later");
    expect(Object.keys(cfg.models)).not.toContain("broken");
    expect(await readFile(path, "utf8")).toContain("keep this comment");
  });

  it("createSerializer runs jobs one at a time and survives a rejection", async () => {
    const serialize = createSerializer();
    let active = 0;
    let maxActive = 0;
    const order: string[] = [];
    const job = (name: string, fail = false) =>
      serialize(async () => {
        active++;
        maxActive = Math.max(maxActive, active);
        await new Promise((r) => setTimeout(r, 5));
        order.push(name);
        active--;
        if (fail) throw new Error(`boom ${name}`);
        return name;
      });

    const first = job("a");
    const rejected = job("b", true);
    const third = job("c");
    await expect(rejected).rejects.toThrow("boom b");
    expect(await first).toBe("a");
    expect(await third).toBe("c"); // the rejection did not poison the chain
    expect(maxActive).toBe(1);
    expect(order).toEqual(["a", "b", "c"]);
  });
});
