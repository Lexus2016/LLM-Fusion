import { describe, it, expect } from "vitest";
import { OllamaClient } from "../src/upstream/ollama";
import { singleStrategy } from "../src/strategies/single";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { CapabilityService } from "../src/capabilities";
import { mockFetch, jsonResponse, sseResponse } from "./helpers";
import type { ChatCompletionRequest, StrategyContext, UpstreamClient } from "../src/types";

const logger = createLogger({ level: "silent" });
const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  models: { "fast-glm": { strategy: "single", target: "glm-5.2" } },
});

function ctxWith(client: UpstreamClient, request: ChatCompletionRequest): StrategyContext {
  const capabilities = new CapabilityService({
    client,
    getOverrides: () => config.overrides,
    logger,
  });
  const entry = config.models["fast-glm"];
  if (!entry) throw new Error("test config missing fast-glm");
  return { request, config, client, capabilities, logger, modelConfig: entry };
}

describe("single strategy", () => {
  it("returns the upstream JSON for a non-stream request", async () => {
    const completion = {
      id: "x",
      object: "chat.completion",
      choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
    };
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        { match: (u) => u.endsWith("/v1/chat/completions"), respond: () => jsonResponse(completion) },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", messages: [{ role: "user", content: "hi" }] }),
    );
    expect(res.status).toBe(200);
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("hi");
  });

  it("rewrites the virtual model name to the resolved upstream target", async () => {
    let sentModel: unknown;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const parsed = JSON.parse(String(init?.body));
            sentModel = parsed.model;
            return jsonResponse({ ok: true });
          },
        },
      ]),
    });
    await singleStrategy.execute(ctxWith(client, { model: "fast-glm", messages: [] }));
    expect(sentModel).toBe("glm-5.2");
  });

  it("pipes SSE chunks through for a stream request", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () =>
            sseResponse([
              { choices: [{ delta: { content: "a" } }] },
              { choices: [{ delta: { content: "b" } }] },
            ]),
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, messages: [] }),
    );
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain('"content":"a"');
    expect(text).toContain('"content":"b"');
    expect(text).toContain("[DONE]");
  });
});
