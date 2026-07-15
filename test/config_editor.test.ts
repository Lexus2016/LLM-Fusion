import { describe, it, expect } from "vitest";
import { mkdtemp, writeFile, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MiddlewareHandler } from "hono";
import pino from "pino";
import { createConfigEditorApp } from "../src/panel/config_editor";
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
