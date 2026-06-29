import { describe, it, expect } from "vitest";
import { createApp } from "../src/server";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { mockFetch, jsonResponse } from "./helpers";

const logger = createLogger({ level: "silent" });
const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  models: { "fast-glm": { strategy: "single", target: "glm-5.2" } },
});

function app(token?: string) {
  const client = new OllamaClient({
    baseUrl: "https://mock.test",
    apiKey: "k",
    fetchFn: mockFetch([
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () => jsonResponse({ choices: [{ message: { content: "ok" } }] }),
      },
    ]),
  });
  const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
  return createApp({ getConfig: () => config, client, capabilities, getAuthToken: () => token, logger });
}

function post(a: ReturnType<typeof app>, headers: Record<string, string>) {
  return a.request("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify({ model: "fast-glm", messages: [] }),
  });
}

function getModels(a: ReturnType<typeof app>, headers: Record<string, string>) {
  return a.request("/v1/models", { method: "GET", headers });
}

describe("auth", () => {
  it("rejects a missing bearer when a token is configured", async () => {
    const res = await post(app("secret"), {});
    expect(res.status).toBe(401);
  });

  it("rejects a wrong bearer", async () => {
    const res = await post(app("secret"), { authorization: "Bearer wrong" });
    expect(res.status).toBe(401);
  });

  it("rejects an equal-length wrong bearer (constant-time path, no throw)", async () => {
    // "secres" and "secret" are the same byte length: exercises the
    // timingSafeEqual branch that throws on unequal-length buffers.
    const res = await post(app("secret"), { authorization: "Bearer secres" });
    expect(res.status).toBe(401);
  });

  it("accepts the correct bearer", async () => {
    const res = await post(app("secret"), { authorization: "Bearer secret" });
    expect(res.status).toBe(200);
  });

  it("allows all requests when no token is configured", async () => {
    const res = await post(app(undefined), {});
    expect(res.status).toBe(200);
  });

  it("rejects with 500 when the configured token is empty (no silent auth bypass)", async () => {
    // An env var set to "" is a misconfiguration; the proxy must not run open.
    const res = await post(app(""), {});
    expect(res.status).toBe(500);
  });

  it("protects /v1/models behind auth (rejects missing bearer)", async () => {
    const res = await getModels(app("secret"), {});
    expect(res.status).toBe(401);
  });

  it("allows /v1/models with the correct bearer", async () => {
    const res = await getModels(app("secret"), { authorization: "Bearer secret" });
    expect(res.status).toBe(200);
  });
});
