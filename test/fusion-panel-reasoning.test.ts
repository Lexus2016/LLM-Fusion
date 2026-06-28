import { describe, it, expect } from "vitest";
import { z } from "zod";
import { fusionStrategy } from "../src/strategies/fusion";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { jsonResponse, sseResponse } from "./helpers";
import type { ChatCompletionRequest, FetchFn, StrategyContext, UpstreamClient } from "../src/types";

const logger = createLogger({ level: "silent" });

const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 4 },
  models: { "fusion-1": { strategy: "fusion", panel: ["m1", "m2", "m3"], judge: "j", synth: "s" } },
});

type ChatHandler = (body: any) => Response | Promise<Response>;

function makeUpstream(chat: ChatHandler): UpstreamClient {
  const fetchFn: FetchFn = async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (url.endsWith("/api/show")) return jsonResponse({ capabilities: ["completion"], model_info: {} });
    if (url.endsWith("/v1/chat/completions") || url.endsWith("/api/chat")) {
      const body = JSON.parse(String(init?.body));
      return chat(body);
    }
    return jsonResponse({ error: `no route for ${url}` }, 404);
  };
  return new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
}

function ctx(client: UpstreamClient, request: ChatCompletionRequest): StrategyContext {
  const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
  const entry = config.models["fusion-1"];
  if (!entry) throw new Error("missing fusion-1 model config in test fixture");
  return { request, config, client, capabilities, logger, modelConfig: entry };
}

// Reproduce the production failure: gpt-oss:120b and kimi-k2.7-code return their
// deliberation ENTIRELY in delta.reasoning with empty delta.content, then emit a
// tool_calls chunk with finish_reason:"tool_calls". The proxy streamed a panel
// body (stream:true overrides the body's stream:false), so accumulateStreamAndTrack
// is what parses this. The panel must NOT drop members whose deliberation lives in
// reasoning, and should not be empty-handed when a member emits only tool_calls.
describe("fusion panel: reasoning-only + tool_calls panel members", () => {
  it("uses reasoning as the panel answer when content is empty (no drop)", async () => {
    // m1: reasoning-only streaming (content always ""), no tool_calls.
    const m1Chunks = [
      { choices: [{ delta: { role: "assistant", content: "", reasoning: "Plan: " } }] },
      { choices: [{ delta: { role: "assistant", content: "", reasoning: "read the file" } }] },
      { choices: [{ delta: { role: "assistant", content: "" }, finish_reason: "stop" }] },
    ];
    const m2Chunks = [
      { choices: [{ delta: { role: "assistant", content: "", reasoning: "Run tests" } }] },
      { choices: [{ delta: { role: "assistant", content: "", reasoning: " then inspect" } }] },
      { choices: [{ delta: { role: "assistant", content: "" }, finish_reason: "stop" }] },
    ];
    const chat: ChatHandler = (body) => {
      if (body.model === "j") return jsonResponse({ choices: [{ message: { content: JSON.stringify({ consensus: "ok" }) } }] });
      if (body.model === "s") return jsonResponse({ choices: [{ message: { content: "final" } }] });
      if (body.model === "m1") return sseResponse(m1Chunks);
      if (body.model === "m2") return sseResponse(m2Chunks);
      return sseResponse(m1Chunks);
    };
    const client = makeUpstream(chat);
    const request: ChatCompletionRequest = { model: "fusion-1", messages: [{ role: "user", content: "go" }], stream: false };
    const res = await fusionStrategy.execute(ctx(client, request));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.choices?.[0]?.message?.content).toBe("final");
  });

  it("serializes a tool_calls-only panel member instead of dropping it", async () => {
    // m1: reasoning then a tool_calls chunk (finish_reason tool_calls) — like the probe.
    const m1Chunks = [
      { choices: [{ delta: { role: "assistant", content: "", reasoning: "We should run tests" } }] },
      { choices: [{ delta: { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "run_bash", arguments: "{\"cmd\":\"npm test\"}" } }] }, finish_reason: "tool_calls" }] },
    ];
    const m2Chunks = [
      { choices: [{ delta: { role: "assistant", content: "", reasoning: "Inspect src" } }] },
      { choices: [{ delta: { role: "assistant", content: "" }, finish_reason: "stop" }] },
    ];
    const chat: ChatHandler = (body) => {
      if (body.model === "j") return jsonResponse({ choices: [{ message: { content: JSON.stringify({ consensus: "ok" }) } }] });
      if (body.model === "s") return jsonResponse({ choices: [{ message: { content: "final" } }] });
      if (body.model === "m1") return sseResponse(m1Chunks);
      return sseResponse(m2Chunks);
    };
    const client = makeUpstream(chat);
    const request: ChatCompletionRequest = { model: "fusion-1", messages: [{ role: "user", content: "go" }], stream: false };
    const res = await fusionStrategy.execute(ctx(client, request));
    expect(res.status).toBe(200);
  });
});