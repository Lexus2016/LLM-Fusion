import { describe, it, expect } from "vitest";
import { pino } from "pino";
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

  it("GET /v1/models reports a fusion model's context_window as the MIN across all members", async () => {
    // A fusion request fans out to every panel member, so the merged virtual
    // model's usable window is the SMALLEST member's — never the largest.
    const ctxByModel: Record<string, number> = { big: 1_000_000, mid: 500_000, small: 250_000 };
    const fusionConfig = parseConfig({
      upstream: { base_url: "https://mock.test", api_key_env: "X" },
      models: { merged: { strategy: "fusion", panel: ["big", "mid", "small"], judge: "big", synth: "big" } },
    });
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/api/show"),
        respond: (_u, init) => {
          const model = JSON.parse(String(init?.body)).model;
          return jsonResponse({
            capabilities: ["completion"],
            model_info: { "general.context_length": ctxByModel[model] ?? 0 },
          });
        },
      },
    ];
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn: mockFetch(routes) });
    const capabilities = new CapabilityService({ client, getOverrides: () => fusionConfig.overrides, logger });
    const app = createApp({ getConfig: () => fusionConfig, client, capabilities, getAuthToken: () => undefined, logger });

    const res = await app.request("/v1/models");
    const body = JSON.parse(await res.text());
    const merged = body.data.find((m: { id: string }) => m.id === "merged");
    expect(merged.context_window).toBe(250_000); // min(1M, 500k, 250k), NOT 1M
  });

  it("GET /v1/models omits context_window when any member's context is unknown", async () => {
    const fusionConfig = parseConfig({
      upstream: { base_url: "https://mock.test", api_key_env: "X" },
      models: { merged: { strategy: "fusion", panel: ["known", "unknown"], judge: "known", synth: "known" } },
    });
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/api/show"),
        respond: (_u, init) => {
          const model = JSON.parse(String(init?.body)).model;
          // "unknown" returns no context_length -> its window is unknown.
          const model_info = model === "known" ? { "general.context_length": 250_000 } : {};
          return jsonResponse({ capabilities: ["completion"], model_info });
        },
      },
    ];
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn: mockFetch(routes) });
    const capabilities = new CapabilityService({ client, getOverrides: () => fusionConfig.overrides, logger });
    const app = createApp({ getConfig: () => fusionConfig, client, capabilities, getAuthToken: () => undefined, logger });

    const res = await app.request("/v1/models");
    const body = JSON.parse(await res.text());
    const merged = body.data.find((m: { id: string }) => m.id === "merged");
    expect(merged.context_window).toBeUndefined(); // never over-advertise an unknown bound
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

  it("does not forward stale upstream content-length / content-encoding on a rewritten body", async () => {
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () =>
          new Response(
            JSON.stringify({
              choices: [{ message: { role: "assistant", content: "x" } }],
              usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }),
            {
              status: 200,
              headers: {
                "content-type": "application/json",
                "content-length": "999", // stale — the body will be rewritten for usage injection
                "content-encoding": "gzip", // stale
              },
            },
          ),
      },
    ];
    const res = await postChat(makeApp(routes), { model: "fast-glm", stream: false, messages: [] });
    expect(res.headers.get("content-length")).toBeNull();
    expect(res.headers.get("content-encoding")).toBeNull();
    expect(res.headers.get("content-type")).toContain("application/json");
  });

  it("logs `request usage` even when the upstream stream errors mid-stream (H-2)", async () => {
    const lines: Array<Record<string, unknown>> = [];
    const capLogger = pino({ level: "info", base: undefined }, { write(s: string) { lines.push(JSON.parse(s)); } });
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () =>
          new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"x"}}]}\n\n'));
                controller.error(new Error("upstream dropped mid-stream"));
              },
            }),
            { status: 200, headers: { "content-type": "text/event-stream" } },
          ),
      },
    ];
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn: mockFetch(routes) });
    const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger: capLogger });
    const app = createApp({ getConfig: () => config, client, capabilities, getAuthToken: () => undefined, logger: capLogger });

    const res = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "fast-glm", stream: true, messages: [] }),
    });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const body = res.body;
    if (body) {
      const reader = body.getReader();
      try {
        for (;;) {
          const { done } = await reader.read();
          if (done) break;
        }
      } catch {
        /* expected: the upstream stream errored mid-way */
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 10)); // let the pipeTo rejection settle the log
    expect(lines.find((l) => l.msg === "request usage")).toBeDefined();
  });

  it("fused stream preserves chunk order: promoted content, then usage, then [DONE] (5b)", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: async (input, init) => {
        const url =
          typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
        if (url.endsWith("/api/show")) return jsonResponse({ capabilities: ["completion"], model_info: {} });
        const body = JSON.parse(String(init?.body));
        if (body.stream === true) {
          // synth streams reasoning-only deltas -> the promotion transform turns them into content
          return sseResponse([
            { choices: [{ delta: { reasoning: "alpha " } }] },
            { choices: [{ delta: { reasoning: "beta" } }] },
          ]);
        }
        if (body.response_format) {
          return jsonResponse({ choices: [{ message: { content: JSON.stringify({ consensus: "ok" }) } }] });
        }
        return jsonResponse({ choices: [{ message: { content: "panel-ans" } }] });
      },
    });
    const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
    const app = createApp({ getConfig: () => config, client, capabilities, getAuthToken: () => undefined, logger });
    const res = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "fusion-1", stream: true, messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    // reasoning was promoted to content (both transforms ran)
    expect(text).toContain('"content":"alpha "');
    expect(text).toContain('"content":"beta"');
    // ordering invariant: all content -> the single usage chunk -> [DONE]
    const idxContent = text.lastIndexOf('"content":"beta"');
    const idxUsage = text.indexOf("fusion-usage");
    const idxDone = text.indexOf("[DONE]");
    expect(idxUsage).toBeGreaterThan(idxContent);
    expect(idxDone).toBeGreaterThan(idxUsage);
    expect(text.match(/\[DONE\]/g)?.length).toBe(1); // exactly one terminator
    expect(text.match(/fusion-usage/g)?.length).toBe(1); // exactly one usage chunk
  });

  it("gracefully completes the stream with usage and [DONE] even when the upstream stream fails mid-way", async () => {
    const errorStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"part1"}}]}\n\n'));
        setTimeout(() => {
          controller.error(new Error("upstream connection broken"));
        }, 10);
      },
    });
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: async () =>
        new Response(errorStream, {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        }),
    });
    const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
    const app = createApp({ getConfig: () => config, client, capabilities, getAuthToken: () => undefined, logger });
    const res = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "fast-glm", stream: true, messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("part1");
    expect(text).toContain("fusion-usage");
    expect(text).toContain("[DONE]");
    expect(text.match(/\[DONE\]/g)?.length).toBe(1);
    expect(text.match(/fusion-usage/g)?.length).toBe(1);
  });
});

