import { describe, it, expect } from "vitest";
import { createApp } from "../src/server";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { mockFetch, jsonResponse, sseResponse } from "./helpers";
import type { MockRoute } from "./helpers";
import {
  anthropicToOpenAiRequest,
  openAiToAnthropicResponse,
  type AnthropicRequest,
} from "../src/anthropic";

const logger = createLogger({ level: "silent" });

const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  server: { bind: "127.0.0.1", port: 8080 },
  models: {
    "anthropic-fast": { strategy: "single", target: "glm-5.2" },
    "anthropic-fusion": { strategy: "fusion", panel: ["a", "b"], judge: "a", synth: "b" },
  },
});

function defaultRoutes(): MockRoute[] {
  return [
    {
      match: (u) => u.endsWith("/v1/chat/completions"),
      respond: () =>
        jsonResponse({
          id: "up-1",
          choices: [{ message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
        }),
    },
    {
      match: (u) => u.endsWith("/api/show"),
      respond: () => jsonResponse({ capabilities: ["completion"], model_info: {} }),
    },
  ];
}

function makeApp(routes: MockRoute[] = defaultRoutes(), authToken?: string) {
  const client = new OllamaClient({
    baseUrl: "https://mock.test",
    apiKey: "k",
    fetchFn: mockFetch(routes),
  });
  const capabilities = new CapabilityService({
    client,
    getOverrides: () => config.overrides,
    logger,
  });
  return createApp({
    getConfig: () => config,
    client,
    capabilities,
    getAuthToken: () => authToken,
    logger,
  });
}

function postMessages(app: ReturnType<typeof makeApp>, body: unknown, headers?: Record<string, string>) {
  return app.request("/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

describe("anthropic abort propagation", () => {
  it("wires the client abort signal into upstream calls (/v1/messages)", async () => {
    let upstreamSignal: AbortSignal | undefined;
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: (_u, init) => {
          upstreamSignal = init?.signal ?? undefined;
          return jsonResponse({
            id: "up-1",
            choices: [{ message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          });
        },
      },
      { match: (u) => u.endsWith("/api/show"), respond: () => jsonResponse({ capabilities: ["completion"], model_info: {} }) },
    ];
    const app = makeApp(routes);
    const controller = new AbortController();
    const res = await app.request("/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "anthropic-fast", max_tokens: 10, messages: [{ role: "user", content: "hi" }] }),
      signal: controller.signal,
    });
    await res.text();
    // Without the wiring ctx.signal is undefined and single forwards undefined upstream.
    expect(upstreamSignal).toBeDefined();
    expect(upstreamSignal?.aborted).toBe(false);
    // Aborting the CLIENT must abort the captured UPSTREAM signal -> proves propagation.
    controller.abort();
    expect(upstreamSignal?.aborted).toBe(true);
  });
});

describe("anthropic error shape", () => {
  it("returns Anthropic-shaped errors for invalid JSON body", async () => {
    const app = makeApp();
    const res = await app.request("/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "not-json",
    });
    expect(res.status).toBe(400);
    const body = JSON.parse(await res.text());
    expect(body).toEqual({ type: "error", error: { type: "invalid_request_error", message: "request body must be valid JSON" } });
  });
});

describe("anthropic translation", () => {
  it("maps a simple user + system prompt to OpenAI messages", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [{ role: "user", content: "hi" }],
      system: "be brief",
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.model).toBe("anthropic-fast");
    expect(openAi.messages).toEqual([
      { role: "system", content: "be brief" },
      { role: "user", content: "hi" },
    ]);
  });

  it("maps a system role message inside messages to an OpenAI system message", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        { role: "system", content: "be brief" },
        { role: "user", content: "hi" },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      { role: "system", content: "be brief" },
      { role: "user", content: "hi" },
    ]);
  });

  it("maps assistant tool_use to OpenAI tool_calls", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        { role: "user", content: "run it" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "okay" },
            { type: "tool_use", id: "tu-1", name: "bash", input: { command: "echo hi" } },
          ],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "tu-1", content: "hi" }],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      { role: "user", content: "run it" },
      {
        role: "assistant",
        content: "okay",
        tool_calls: [
          {
            id: "tu-1",
            type: "function",
            function: { name: "bash", arguments: JSON.stringify({ command: "echo hi" }) },
          },
        ],
      },
      { role: "tool", content: "hi", tool_call_id: "tu-1" },
    ]);
  });

  it("maps Anthropic tools to OpenAI functions", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [{ role: "user", content: "hi" }],
      tools: [{ name: "read", description: "read a file", input_schema: { type: "object" } }],
      tool_choice: { type: "tool", name: "read" },
      max_tokens: 1024,
      temperature: 0.5,
      top_p: 0.9,
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.tools).toEqual([
      {
        type: "function",
        function: { name: "read", description: "read a file", parameters: { type: "object" } },
      },
    ]);
    expect(openAi.tool_choice).toEqual({ type: "function", function: { name: "read" } });
    expect(openAi.max_tokens).toBe(1024);
    expect(openAi.temperature).toBe(0.5);
    expect(openAi.top_p).toBe(0.9);
  });

  it("maps an OpenAI response with content to an Anthropic message", () => {
    const openAi = {
      id: "r-1",
      choices: [{ message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
    };
    const anthropic = openAiToAnthropicResponse(openAi, "anthropic-fast", {
      upstreamCalls: 1,
      promptTokens: 3,
      completionTokens: 2,
      totalTokens: 5,
      costUsd: null,
    });
    expect(anthropic).toMatchObject({
      id: "r-1",
      type: "message",
      role: "assistant",
      model: "anthropic-fast",
      content: [{ type: "text", text: "hello" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 3, output_tokens: 2 },
    });
  });

  it("maps OpenAI tool_calls to Anthropic tool_use blocks", () => {
    const openAi = {
      id: "r-2",
      choices: [
        {
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "tu-2",
                function: { name: "bash", arguments: JSON.stringify({ command: "ls" }) },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    };
    const anthropic = openAiToAnthropicResponse(openAi, "anthropic-fast", {
      upstreamCalls: 1,
      promptTokens: 4,
      completionTokens: 5,
      totalTokens: 9,
      costUsd: null,
    });
    expect(anthropic).toMatchObject({
      type: "message",
      stop_reason: "tool_use",
      content: [{ type: "tool_use", id: "tu-2", name: "bash", input: { command: "ls" } }],
      usage: { input_tokens: 4, output_tokens: 5 },
    });
  });

  it("maps a length-truncated tool_calls response to stop_reason:max_tokens, not tool_use", () => {
    // Regression: a Write/Edit tool call cut by max_tokens arrives with
    // finish_reason "length" and a tool_calls block whose arguments JSON is
    // missing its tail. Reporting stop_reason "tool_use" makes Claude Code
    // execute the truncated input; "max_tokens" lets it recover instead.
    const openAi = {
      id: "r-3",
      choices: [
        {
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "tu-3",
                function: { name: "write_file", arguments: JSON.stringify({ path: "a.html" }) },
              },
            ],
          },
          finish_reason: "length",
        },
      ],
    };
    const anthropic = openAiToAnthropicResponse(openAi, "anthropic-fast", {
      upstreamCalls: 1,
      promptTokens: 4,
      completionTokens: 5,
      totalTokens: 9,
      costUsd: null,
    });
    expect(anthropic).toMatchObject({ stop_reason: "max_tokens" });
  });

  it("accepts null assistant content and converts it to an empty string", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: null },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "" },
    ]);
  });

  it("ignores thinking and redacted_thinking blocks", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "step one" },
            { type: "redacted_thinking", data: "abcd" },
            { type: "text", text: "result" },
          ],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([{ role: "assistant", content: "result" }]);
  });

  it("accepts a tool_result with null content", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        { role: "user", content: "run it" },
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "tu-1", name: "bash", input: { command: "ls" } }],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "tu-1", content: null }],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      { role: "user", content: "run it" },
      { role: "assistant", tool_calls: [{ id: "tu-1", type: "function", function: { name: "bash", arguments: JSON.stringify({ command: "ls" }) } }] },
      { role: "tool", content: "", tool_call_id: "tu-1" },
    ]);
  });

  it("accepts a tool_result containing an image and maps it to a URL string", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tu-2",
              content: [
                { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
              ],
            },
          ],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      { role: "tool", content: "data:image/png;base64,iVBORw0KGgo=", tool_call_id: "tu-2" },
    ]);
  });

  it("falls back to empty content when an assistant message has only thinking blocks", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "assistant",
          content: [{ type: "thinking", thinking: "internal reasoning" }],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([{ role: "assistant", content: "" }]);
  });

  it("accepts thinking and redacted_thinking with missing or null content fields", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "thinking", signature: "sig1" },
            { type: "redacted_thinking", signature: "sig2" },
          ],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([{ role: "assistant", content: "" }]);
  });

  it("accepts a tool_use block with null and string inputs", () => {
    const req1: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "tu-1", name: "bash", input: null },
          ],
        },
      ],
    };
    const openAi1 = anthropicToOpenAiRequest(req1);
    expect(openAi1.messages).toEqual([
      {
        role: "assistant",
        tool_calls: [
          {
            id: "tu-1",
            type: "function",
            function: { name: "bash", arguments: "{}" },
          },
        ],
      },
    ]);

    const req2: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "tu-2", name: "bash", input: '{"arg":"val"}' },
          ],
        },
      ],
    };
    const openAi2 = anthropicToOpenAiRequest(req2);
    expect(openAi2.messages).toEqual([
      {
        role: "assistant",
        tool_calls: [
          {
            id: "tu-2",
            type: "function",
            function: { name: "bash", arguments: '{"arg":"val"}' },
          },
        ],
      },
    ]);
  });

  it("accepts a tool_result with embedded thinking blocks", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tu-1",
              content: [
                { type: "thinking", thinking: "thought text" },
                { type: "text", text: "result text" },
              ],
            },
          ],
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      { role: "tool", content: "thought text\nresult text", tool_call_id: "tu-1" },
    ]);
  });

  it("accepts a single content block object directly in content", () => {
    const req: AnthropicRequest = {
      model: "anthropic-fast",
      messages: [
        {
          role: "user",
          content: { type: "text", text: "hello" },
        },
      ],
    };
    const openAi = anthropicToOpenAiRequest(req);
    expect(openAi.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "hello" }],
      },
    ]);
  });
});

describe("anthropic route", () => {
  it("POST /v1/messages returns an Anthropic-shaped message", async () => {
    const res = await postMessages(makeApp(), {
      model: "anthropic-fast",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.type).toBe("message");
    expect(body.role).toBe("assistant");
    expect(body.content).toEqual([{ type: "text", text: "hello" }]);
    expect(body.stop_reason).toBe("end_turn");
    expect(res.headers.get("x-fusion-usage")).toContain('"calls":1');
  });

  it("POST /v1/messages streams Anthropic SSE events", async () => {
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () => sseResponse([{ choices: [{ delta: { content: "x" } }] }]),
      },
    ];
    const res = await postMessages(makeApp(routes), {
      model: "anthropic-fast",
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain('event: message_start');
    expect(text).toContain('"type":"text_delta"');
    expect(text).toContain('"text":"x"');
    expect(text).toContain('event: message_delta');
    expect(text).toContain('event: message_stop');
    expect(res.headers.get("x-fusion-usage")).toContain('"calls":1');
  });

  it("streamed tool_calls with finish_reason:stop still yield stop_reason:tool_use", async () => {
    // Regression: a deviant upstream that emits tool_calls but finish_reason "stop"
    // (or null) must not produce stop_reason:"end_turn" — Claude Code keys its agent
    // loop on stop_reason:"tool_use" and would otherwise never run the tool.
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () =>
          sseResponse([
            { choices: [{ delta: { tool_calls: [{ index: 0, id: "tu-1", function: { name: "bash", arguments: "{}" } }] } }] },
            { choices: [{ delta: {}, finish_reason: "stop" }] },
          ]),
      },
    ];
    const res = await postMessages(makeApp(routes), {
      model: "anthropic-fast",
      stream: true,
      messages: [{ role: "user", content: "run ls" }],
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain('"type":"tool_use"'); // tool block was emitted
    expect(text).toContain('"stop_reason":"tool_use"'); // and reflected despite finish:stop
  });

  it("streamed tool_calls cut by finish_reason:length yield stop_reason:max_tokens", async () => {
    // Regression: a big streamed tool call truncated by the token limit must NOT
    // be presented as a runnable tool_use — its input_json_delta JSON is missing
    // its tail. stop_reason:"max_tokens" tells the client the turn was cut.
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () =>
          sseResponse([
            { choices: [{ delta: { tool_calls: [{ index: 0, id: "tu-1", function: { name: "write_file", arguments: '{"path":"a.html","content":"<html>' } }] } }] },
            { choices: [{ delta: {}, finish_reason: "length" }] },
          ]),
      },
    ];
    const res = await postMessages(makeApp(routes), {
      model: "anthropic-fast",
      stream: true,
      messages: [{ role: "user", content: "write the file" }],
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain('"type":"tool_use"'); // the (partial) tool block still streams
    expect(text).toContain('"stop_reason":"max_tokens"'); // but the turn is honestly marked as cut
  });

  it("authenticates with x-api-key header", async () => {
    const app = makeApp(defaultRoutes(), "secret");
    const ok = await postMessages(
      app,
      { model: "anthropic-fast", messages: [{ role: "user", content: "hi" }] },
      { "x-api-key": "secret" },
    );
    expect(ok.status).toBe(200);

    const bad = await postMessages(
      app,
      { model: "anthropic-fast", messages: [{ role: "user", content: "hi" }] },
      { "x-api-key": "wrong" },
    );
    expect(bad.status).toBe(401);
  });

  it("rejects malformed Anthropic bodies with 400", async () => {
    const res = await postMessages(makeApp(), {});
    expect(res.status).toBe(400);
  });

  it("accepts a request with null content and thinking blocks", async () => {
    const res = await postMessages(makeApp(), {
      model: "anthropic-fast",
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "step" },
            { type: "text", text: "hello" },
          ],
        },
        { role: "user", content: null },
      ],
    });
    expect(res.status).toBe(200);
  });

  it("maps upstream tool_calls to a streamed Anthropic tool_use block", async () => {
    const routes: MockRoute[] = [
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: () =>
          sseResponse([
            {
              choices: [
                {
                  delta: {
                    tool_calls: [{ index: 0, id: "tu-3", function: { name: "bash" } }],
                  },
                },
              ],
            },
            {
              choices: [
                {
                  delta: {
                    tool_calls: [{ index: 0, function: { arguments: '{"co' } }],
                  },
                },
              ],
            },
            {
              choices: [
                {
                  delta: {
                    tool_calls: [{ index: 0, function: { arguments: 'mmand":"ls"}' } }],
                  },
                },
              ],
            },
          ]),
      },
    ];
    const res = await postMessages(makeApp(routes), {
      model: "anthropic-fast",
      stream: true,
      messages: [{ role: "user", content: "list files" }],
    });
    const text = await res.text();
    expect(text).toContain('"type":"tool_use"');
    expect(text).toContain('"name":"bash"');
    expect(text).toContain('"type":"input_json_delta"');
    const partials = [...text.matchAll(/"partial_json":("(?:\\.|[^"\\])*")/g)]
      .map((m) => m[1])
      .filter((s): s is string => s != null)
      .map((s) => JSON.parse(s) as string);
    const combined = partials.join("");
    expect(JSON.parse(combined)).toEqual({ command: "ls" });
  });

  // Regression: Claude Code (and the Anthropic API) can emit content blocks the
  // proxy does not natively translate — e.g. server_tool_use, web_search_tool_result,
  // document, container_upload, code_execution_tool_*. The schema MUST accept them
  // best-effort instead of rejecting the whole request with 400, which breaks the
  // agent loop mid-session as soon as one such block appears (root cause of
  // "invalid Anthropic messages request (messages.N.content: Invalid input)").
  it("accepts a request with an unknown content block type (best-effort)", async () => {
    const res = await postMessages(makeApp(), {
      model: "anthropic-fast",
      messages: [
        { role: "user", content: "search the web" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "let me search" },
            // A server-side tool block the proxy does not model.
            { type: "server_tool_use", id: "srv-1", name: "web_search", input: { query: "x" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "web_search_tool_result", tool_use_id: "srv-1", content: [{ type: "text", text: "result" }] },
          ],
        },
        { role: "user", content: "now answer" },
      ],
    });
    expect(res.status).toBe(200);
  });

  // Same regression as above, but routed through the FUSION strategy
  // (model "anthropic-fusion": panel ["a","b"], judge "a", synth "b"). The
  // schema-parse fix lives BEFORE dispatch, so it is route-independent; this
  // test proves the END-TO-END fusion flow (panel -> judge -> synth) completes
  // with 200 when the original Anthropic request carried unknown content blocks.
  // defaultRoutes() answers every /v1/chat/completions call with "hello"; the
  // judge stage gracefully degrades on non-JSON (parseJudgeAnalysis -> null ->
  // raw panel fallback), so the synth still produces a final answer.
  it("accepts an unknown content block type through the fusion route", async () => {
    const res = await postMessages(makeApp(), {
      model: "anthropic-fusion",
      messages: [
        { role: "user", content: "search the web" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "let me search" },
            { type: "server_tool_use", id: "srv-1", name: "web_search", input: { query: "x" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "web_search_tool_result", tool_use_id: "srv-1", content: [{ type: "text", text: "result" }] },
          ],
        },
        { role: "user", content: "now answer" },
      ],
    });
    expect(res.status).toBe(200);
  });
});
