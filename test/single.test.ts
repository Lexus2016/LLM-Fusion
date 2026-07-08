import { describe, it, expect } from "vitest";
import { OllamaClient } from "../src/upstream/ollama";
import { singleStrategy } from "../src/strategies/single";
import { detectIncompleteToolTurn } from "../src/strategies/tool_turn_guard";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { CapabilityService } from "../src/capabilities";
import { mockFetch, jsonResponse, sseResponse, sseThenError } from "./helpers";
import { createResilience } from "../src/concurrency";
import type { Resilience } from "../src/concurrency";
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

  it("honors a caller-provided AbortSignal and surfaces it as a typed timeout (M-1)", async () => {
    // A fetch that settles only when its signal aborts proves the caller's signal
    // reaches the request — so a fusion stage timeout can cancel the in-flight call
    // and free its concurrency slot instead of letting it linger.
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    });
    const ac = new AbortController();
    const pending = client.chatCompletions({ model: "m" }, { stream: false, signal: ac.signal });
    ac.abort();
    await expect(pending).rejects.toThrow(/cancelled by the caller/);
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

  it("promotes reasoning -> content when content is empty (thinking target, HIGH-1)", async () => {
    // A "thinking" model returns its answer in `reasoning` with empty `content`;
    // the single passthrough now normalizes it so content-only clients see the text.
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () =>
            jsonResponse({
              choices: [
                { index: 0, message: { role: "assistant", content: "", reasoning: "THE ANSWER" }, finish_reason: "stop" },
              ],
            }),
        },
      ]),
    });
    const res = await singleStrategy.execute(ctxWith(client, { model: "fast-glm", messages: [] }));
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("THE ANSWER");
  });

  it("strips <think> and </think> tags from promoted reasoning and content", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () =>
            jsonResponse({
              choices: [
                { index: 0, message: { role: "assistant", content: "hello</think>", reasoning: "" }, finish_reason: "stop" },
              ],
            }),
        },
      ]),
    });
    const res = await singleStrategy.execute(ctxWith(client, { model: "fast-glm", messages: [] }));
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("hello");
  });

  it("strips a complete inline <think>…</think> block from content (R1/QwQ inline reasoning)", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () =>
            jsonResponse({
              choices: [
                {
                  index: 0,
                  message: {
                    role: "assistant",
                    content: "Answer: <think>long private chain of reasoning that must not leak</think>42",
                    reasoning: "",
                  },
                  finish_reason: "stop",
                },
              ],
            }),
        },
      ]),
    });
    const res = await singleStrategy.execute(ctxWith(client, { model: "fast-glm", messages: [] }));
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("Answer: 42");
  });

  it("propagates the context abort signal to upstream (M-1 client disconnect)", async () => {
    // The hanging fetch only settles when ITS signal aborts; aborting the context
    // signal must reject the strategy call — proving ctx.signal reaches upstream.
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    });
    const ac = new AbortController();
    const ctx = ctxWith(client, { model: "fast-glm", messages: [] });
    const pending = singleStrategy.execute({ ...ctx, signal: ac.signal });
    ac.abort();
    await expect(pending).rejects.toThrow();
  });
});

describe("single strategy — circuit breaker availability semantics", () => {
  function statusClient(status: number): UpstreamClient {
    return new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        { match: (u) => u.endsWith("/v1/chat/completions"), respond: () => jsonResponse({ error: "x" }, status) },
      ]),
    });
  }

  function ctxRes(client: UpstreamClient, resilience: Resilience): StrategyContext {
    const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
    const entry = config.models["fast-glm"];
    if (!entry) throw new Error("missing fast-glm");
    return { request: { model: "fast-glm", messages: [] }, config, client, capabilities, logger, modelConfig: entry, resilience };
  }

  function res(failureThreshold: number): Resilience {
    return createResilience({ maxConcurrency: 4, now: () => 1_000_000, sleep: async () => {}, failureThreshold });
  }

  it("does NOT trip the breaker on repeated 4xx client errors", async () => {
    const resilience = res(2); // 2 availability failures would open it
    const client = statusClient(400);
    for (let i = 0; i < 5; i += 1) {
      const out = await singleStrategy.execute(ctxRes(client, resilience));
      expect(out.status).toBe(400); // passed through to the client
    }
    expect(resilience.breaker.getState("glm-5.2")).toBe("closed");
  });

  it("trips the breaker on repeated 5xx availability failures", async () => {
    const resilience = res(2);
    const client = statusClient(503);
    await singleStrategy.execute(ctxRes(client, resilience));
    expect(resilience.breaker.getState("glm-5.2")).toBe("closed"); // 1 < threshold
    await singleStrategy.execute(ctxRes(client, resilience));
    expect(resilience.breaker.getState("glm-5.2")).toBe("open"); // 2 >= threshold
  });

  it("trips the breaker on repeated 429 rate-limits", async () => {
    const resilience = res(2);
    const client = statusClient(429);
    await singleStrategy.execute(ctxRes(client, resilience));
    await singleStrategy.execute(ctxRes(client, resilience));
    expect(resilience.breaker.getState("glm-5.2")).toBe("open");
  });

  it("releases the half-open probe on a 4xx response so the model is not jammed until restart (HIGH)", async () => {
    // Open the breaker with availability failures, then probe with a 4xx.
    // Before the fix the 4xx neither recorded success nor failure, so the
    // half-open probe stuck and every subsequent call fast-failed as open.
    let now = 1_000_000;
    const resilience = createResilience({
      maxConcurrency: 4,
      failureThreshold: 2,
      cooldownMs: 30_000,
      now: () => now,
      sleep: async () => {},
    });
    const failClient = statusClient(503);
    const cFail = ctxRes(failClient, resilience);
    await singleStrategy.execute(cFail); // 1st 5xx
    await singleStrategy.execute(cFail); // 2nd 5xx -> open
    expect(resilience.breaker.getState("glm-5.2")).toBe("open");

    // Cooldown elapses -> half-open. The next call is the probe.
    now += 30_000;
    expect(resilience.breaker.getState("glm-5.2")).toBe("half-open");

    // Probe returns a 4xx (client/request error, NOT a health failure).
    const probeClient = statusClient(400);
    const out = await singleStrategy.execute(ctxRes(probeClient, resilience));
    expect(out.status).toBe(400);

    // The probe MUST be released: a fresh call is allowed again (not circuit-open).
    expect(resilience.breaker.getState("glm-5.2")).not.toBe("open");
    expect(resilience.breaker.canAttempt("glm-5.2")).toBe(true);
  });
});

describe("single strategy — request_overrides", () => {
  const overridesConfig = parseConfig({
    upstream: { base_url: "https://mock.test", api_key_env: "X" },
    models: {
      "fast-glm": {
        strategy: "single",
        target: "glm-5.2",
        request_overrides: { reasoning_effort: "none", model: "evil", messages: [], tools: "nope" },
      },
    },
  });

  function ctxOverrides(client: UpstreamClient, request: ChatCompletionRequest): StrategyContext {
    const capabilities = new CapabilityService({
      client,
      getOverrides: () => overridesConfig.overrides,
      logger,
    });
    const entry = overridesConfig.models["fast-glm"];
    if (!entry) throw new Error("test config missing fast-glm");
    return { request, config: overridesConfig, client, capabilities, logger, modelConfig: entry };
  }

  it("merges request_overrides into the upstream body but never the protected keys", async () => {
    let sent: Record<string, unknown> = {};
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            sent = JSON.parse(String(init?.body));
            return jsonResponse({ choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }] });
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxOverrides(client, { model: "fast-glm", messages: [{ role: "user", content: "hi" }] }),
    );
    expect(res.status).toBe(200);
    expect(sent.reasoning_effort).toBe("none"); // override applied
    expect(sent.model).toBe("glm-5.2"); // protected: resolved target, not "evil"
    expect(sent.messages).toEqual([{ role: "user", content: "hi" }]); // protected: client messages kept
    expect(sent.tools).toBeUndefined(); // protected: no tools smuggled in
  });
});

describe("single strategy — tool-turn completeness guard", () => {
  const TOOLS = [
    {
      type: "function",
      function: {
        name: "write",
        description: "Create a file",
        parameters: {
          type: "object",
          properties: { path: { type: "string" }, content: { type: "string" } },
          required: ["path", "content"],
        },
      },
    },
  ];

  it("detectIncompleteToolTurn: flags empty and intent-tail stops; passes tool_calls and real answers", () => {
    const stop = (msg: Record<string, unknown>) => ({
      choices: [{ finish_reason: "stop", message: { role: "assistant", ...msg } }],
    });
    expect(detectIncompleteToolTurn(stop({ content: "" }))).toBe("empty");
    expect(detectIncompleteToolTurn(stop({ content: "Let me write the complete HTML file now." }))).toBe("intent_tail");
    // reasoning-only narration (thinking model) is judged on its real text
    expect(detectIncompleteToolTurn(stop({ content: "", reasoning: "Now I'll write the file." }))).toBe("intent_tail");
    // a tool call IS the action -> complete
    expect(
      detectIncompleteToolTurn({
        choices: [{ finish_reason: "stop", message: { tool_calls: [{ id: "1", function: { name: "write" } }] } }],
      }),
    ).toBeNull();
    // a genuine completion summary -> complete (no false positive)
    expect(detectIncompleteToolTurn(stop({ content: "Done — the file has been created and verified." }))).toBeNull();
  });

  it("detectIncompleteToolTurn: judges length-cut turns (the large-file truncation failure mode)", () => {
    const len = (msg: Record<string, unknown>) => ({
      choices: [{ finish_reason: "length", message: { role: "assistant", ...msg } }],
    });
    // truncated tool-call arguments (unparseable JSON) -> not runnable -> retry
    expect(
      detectIncompleteToolTurn(len({ tool_calls: [{ id: "1", function: { name: "write", arguments: '{"path":"a.html","content":"<html>...' } }] })),
    ).toBe("broken_tool_call");
    // intact tool call at the cap -> runnable -> leave alone
    expect(
      detectIncompleteToolTurn(len({ tool_calls: [{ id: "1", function: { name: "write", arguments: '{"path":"a.html"}' } }] })),
    ).toBeNull();
    // no calls, everything burned in reasoning, no content -> nothing delivered -> retry
    expect(detectIncompleteToolTurn(len({ content: "", reasoning: "…enormous plan…" }))).toBe("empty");
    // honest length-cut PROSE is still worth delivering -> leave alone
    expect(detectIncompleteToolTurn(len({ content: "let me write a long explanation that got cut" }))).toBeNull();
  });

  it("recovers a length-truncated STREAMING tool call (fragmented broken args) into a complete one", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              // STREAMING recovery retry -> the model finally emits the tool call
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_2", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"short"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Output-cap truncation mid-arguments: args split across chunks, cut
            // before the JSON closes, terminal chunk says "length".
            return sseResponse([
              { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html>' } }] } }] },
              { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "<h1>Guide</h1><p>truncat" } }] } }] },
              { choices: [{ delta: {}, finish_reason: "length" }] },
            ]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "write the full guide" }] }),
    );
    const text = await res.text();
    expect(text).toContain('"content\\":\\"short\\"'); // the recovered COMPLETE call replaced the broken one
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).toContain("[DONE]");
  });

  it("leaves an INTACT length-capped streaming tool call alone (args parse -> runnable, no retry)", async () => {
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            return sseResponse([
              { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "c", type: "function", function: { name: "write", arguments: '{"path":"a"' } }] } }] },
              { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ',"content":"x"}' } }] } }] },
              { choices: [{ delta: {}, finish_reason: "length" }] },
            ]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    const text = await res.text();
    expect(calls).toBe(1); // no recovery retry fired
    expect(text).toContain('"finish_reason":"length"'); // original terminal chunk forwarded
  });

  it("recovers a MID-FLIGHT upstream cut (the Ollama Cloud 'terminated' failure) into a streamed retry", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              // STREAMING recovery retry succeeds with a complete (smaller) call
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"part 1"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Original turn: preamble + partial tool-call args, then the upstream
            // connection dies mid-generation (no terminal chunk ever arrives).
            return sseThenError(
              [
                { choices: [{ delta: { role: "assistant", content: "Створюю посібник — частина 1:" } }] },
                { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_x", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html>' } }] } }] },
              ],
              "terminated",
            );
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "напиши великий посібник" }] }),
    );
    const text = await res.text(); // must NOT throw — the guard converts the error into a recovered stream
    expect(text).toContain("Створюю посібник"); // live-forwarded preamble kept
    expect(text).toContain('"content\\":\\"part 1\\"'); // recovery emitted the complete call
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).toContain("[DONE]");
  });

  it("recovers a narrate-and-stop STREAMING turn into the announced tool call", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              // STREAMING recovery retry -> the model finally emits the tool call
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html></html>"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // first turn -> narrate-and-stop, no tool call
            return sseResponse([
              { choices: [{ delta: { role: "assistant", content: "Let me write the complete HTML file now." } }] },
              { choices: [{ delta: {}, finish_reason: "stop" }] },
            ]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "make guide.html" }] }),
    );
    const text = await res.text();
    expect(text).toContain('"name":"write"'); // the announced tool call was recovered
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).not.toContain('"finish_reason":"stop"'); // held-back narrate-and-stop terminal replaced, not spliced
    expect(text).toContain("[DONE]");
  });

  it("recovers a narrate-and-stop NON-STREAM turn into the announced tool call", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              return jsonResponse({
                choices: [
                  {
                    index: 0,
                    finish_reason: "tool_calls",
                    message: { role: "assistant", content: "", tool_calls: [{ id: "call_1", type: "function", function: { name: "write", arguments: "{}" } }] },
                  },
                ],
              });
            }
            return jsonResponse({
              choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "Let me write the file now." } }],
            });
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", tools: TOOLS, messages: [{ role: "user", content: "write it" }] }),
    );
    const parsed = JSON.parse(await res.text());
    expect(parsed.choices[0].message.tool_calls?.[0]?.function?.name).toBe("write");
  });

  it("does NOT retry a genuinely complete turn (no false positive, single upstream call)", async () => {
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            return jsonResponse({
              choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "Done — the file has been created and verified." } }],
            });
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    const parsed = JSON.parse(await res.text());
    expect(parsed.choices[0].message.content).toContain("has been created");
    expect(calls).toBe(1); // no recovery retry fired
  });

  it("emits exactly ONE [DONE] even when the upstream ends with [DONE] but no finish_reason chunk", async () => {
    // Post-release review finding: the guard used to forward the upstream [DONE]
    // and then append its own after the terminal-less recovery — double framing.
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "c", type: "function", function: { name: "write", arguments: "{}" } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Anomalous upstream: content chunks, then [DONE] with NO finish_reason chunk.
            return sseResponse([{ choices: [{ delta: { role: "assistant", content: "partial" } }] }]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    const text = await res.text();
    const doneCount = (text.match(/data: \[DONE\]/g) ?? []).length;
    expect(doneCount).toBe(1); // canonical framing: exactly one [DONE], appended by the guard
    expect(text.indexOf("[DONE]")).toBeGreaterThan(text.indexOf('"name":"write"')); // recovery BEFORE the single [DONE]
  });

  it("leaves tool-less requests as plain passthrough (guard inert even on narrate-and-stop)", async () => {
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            return jsonResponse({
              choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "Let me write the file now." } }],
            });
          },
        },
      ]),
    });
    // Same narrate-and-stop content but NO tools -> the guard must not run.
    const res = await singleStrategy.execute(ctxWith(client, { model: "fast-glm", messages: [] }));
    const parsed = JSON.parse(await res.text());
    expect(parsed.choices[0].message.content).toContain("Let me write");
    expect(calls).toBe(1);
  });
});
