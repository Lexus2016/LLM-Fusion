import { describe, it, expect } from "vitest";
import { createApp } from "../src/server";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { mockFetch, jsonResponse, sseResponse } from "./helpers";
import type { MockRoute } from "./helpers";

const logger = createLogger({ level: "silent" });

const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  server: { bind: "127.0.0.1", port: 8080 },
  models: {
    "fast-glm": { strategy: "single", target: "glm-5.2" },
    "fusion-1": { strategy: "fusion", panel: ["a", "b"], judge: "a", synth: "b" },
  },
});

function defaultRoutes(): MockRoute[] {
  return [
    {
      match: (u) => u.endsWith("/v1/chat/completions"),
      respond: () => jsonResponse({ id: "x", choices: [{ message: { content: "ok" } }] }),
    },
    {
      match: (u) => u.endsWith("/api/show"),
      respond: () => jsonResponse({ capabilities: ["completion"], model_info: {} }),
    },
  ];
}

function makeApp(routes: MockRoute[] = defaultRoutes(), authToken?: string) {
  const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn: mockFetch(routes) });
  const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
  return createApp({ getConfig: () => config, client, capabilities, getAuthToken: () => authToken, logger });
}

function postChat(app: ReturnType<typeof makeApp>, body: unknown) {
  return app.request("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("server", () => {
  it("GET /health returns ok", async () => {
    const res = await makeApp().request("/health");
    expect(res.status).toBe(200);
    expect(JSON.parse(await res.text())).toEqual({ status: "ok" });
  });

  it("GET /v1/models lists the configured virtual models", async () => {
    const res = await makeApp().request("/v1/models");
    expect(res.status).toBe(200);
    const body = JSON.parse(await res.text());
    expect(body.object).toBe("list");
    const ids = body.data.map((m: { id: string }) => m.id);
    expect(ids).toContain("fast-glm");
    expect(ids).toContain("fusion-1");
  });

  it("POST /v1/chat/completions on a single model returns the mocked completion", async () => {
    const res = await postChat(makeApp(), { model: "fast-glm", messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("ok");
  });

  it("unknown virtual model -> 404", async () => {
    const res = await postChat(makeApp(), { model: "nope", messages: [] });
    expect(res.status).toBe(404);
  });

  it("fusion model dispatches end-to-end (Phase 3) -> 200", async () => {
    // Every panel/judge/synth call hits the default mock; the judge body "ok"
    // is not valid JSON, so fusion uses the judge fallback and synth still
    // returns a completion. The 501 placeholder is gone now that fusion exists.
    const res = await postChat(makeApp(), { model: "fusion-1", messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("ok");
  });

  it("malformed body -> 400", async () => {
    const res = await postChat(makeApp(), { messages: [] }); // missing model
    expect(res.status).toBe(400);
  });

  it("GET /ready reports degraded (503) when upstream /api/show fails", async () => {
    const routes: MockRoute[] = [
      { match: (u) => u.endsWith("/api/show"), respond: () => jsonResponse({ error: "nope" }, 500) },
    ];
    const res = await makeApp(routes).request("/ready");
    expect(res.status).toBe(503);
  });

  it("GET /ready returns ok when upstream is reachable", async () => {
    const res = await makeApp().request("/ready");
    expect(res.status).toBe(200);
  });

  it("streams SSE end-to-end for stream:true", async () => {
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () => sseResponse([{ choices: [{ delta: { content: "x" } }] }]),
      },
    ];
    const res = await postChat(makeApp(routes), { model: "fast-glm", stream: true, messages: [] });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain('"content":"x"');
    expect(text).toContain("[DONE]");
  });
});
