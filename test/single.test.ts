import { describe, it, expect } from "vitest";
import { z } from "zod";
import { OllamaClient } from "../src/upstream/ollama";
import { singleStrategy } from "../src/strategies/single";
import { detectIncompleteToolTurn } from "../src/strategies/tool_turn_guard";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { CapabilityService } from "../src/capabilities";
import { mockFetch, jsonResponse, sseResponse, sseThenError, streamErrorImmediate } from "./helpers";
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

  it("buffers streaming tool_call fragments so a length-cut + recovery yields VALID index-0 JSON on the client (corruption regression)", async () => {
    // The confirmed silent-corruption bug: the guard forwarded truncated tool-call
    // arg fragments LIVE, then recovery re-emitted a fresh call restarting at
    // index:0. An index-keyed client (openai-python, Vercel AI SDK, OpenCode)
    // concatenated the truncated old args with the recovered args -> invalid JSON.
    // Option B buffers tool_call deltas, so the client only ever sees the clean
    // recovered call. This test accumulates arguments BY INDEX like a real client
    // and asserts JSON.parse SUCCEEDS.
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              // Recovery retry emits a COMPLETE call, restarting at index:0.
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_2", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"short"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Output-cap truncation: index-0 args split across two chunks, cut
            // before the JSON closes, terminal chunk says "length".
            return sseResponse([
              { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>Guide</h1><p>truncat' } }] } }] },
              { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "ed at the output cap" } }] } }] },
              { choices: [{ delta: {}, finish_reason: "length" }] },
            ]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "write the full guide" }] }),
    );
    if (!res.body) throw new Error("expected a stream body");

    // Minimal client-shaped chunk schema (no `as`, no `any`).
    const ClientChunk = z
      .object({
        choices: z
          .array(
            z.object({
              delta: z
                .object({
                  tool_calls: z
                    .array(
                      z.object({
                        index: z.number().optional(),
                        function: z.object({ arguments: z.string().optional() }).passthrough().optional(),
                      }).passthrough(),
                    )
                    .optional(),
                })
                .passthrough()
                .optional(),
            }).passthrough(),
          )
          .optional(),
      })
      .passthrough();

    // Reconstruct the client's per-index argument accumulation.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let raw = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      raw += decoder.decode(value, { stream: true });
    }
    const argsByIndex = new Map<number, string>();
    for (const line of raw.split("\n")) {
      const t = line.trimStart();
      if (!t.startsWith("data:")) continue;
      const payload = t.slice("data:".length).trim();
      if (payload === "[DONE]" || payload.length === 0) continue;
      let obj: unknown;
      try {
        obj = JSON.parse(payload);
      } catch {
        continue;
      }
      const parsed = ClientChunk.safeParse(obj);
      if (!parsed.success) continue;
      const calls = parsed.data.choices?.[0]?.delta?.tool_calls;
      if (!Array.isArray(calls)) continue;
      for (const c of calls) {
        const idx = typeof c.index === "number" ? c.index : 0;
        const prev = argsByIndex.get(idx) ?? "";
        argsByIndex.set(idx, prev + (typeof c.function?.arguments === "string" ? c.function.arguments : ""));
      }
    }

    const assembled = argsByIndex.get(0) ?? "";
    // The bug produced `{"path":..."truncat...ed at the output cap{"path":...}` —
    // JSON.parse throws. With buffering the client sees only the recovered call.
    expect(() => JSON.parse(assembled)).not.toThrow();
    expect(JSON.parse(assembled)).toEqual({ path: "guide.html", content: "short" });
    expect(raw).toContain("[DONE]");
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

  it("recovers a MID-FLIGHT upstream cut that happens BEFORE anything was forwarded", async () => {
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
            // Original turn: the upstream connection dies before the first token
            // (the Ollama Cloud "terminated" failure). Nothing reached the client,
            // so the recovery retry IS the whole answer — safe to splice in.
            return streamErrorImmediate("terminated");
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "напиши великий посібник" }] }),
    );
    const text = await res.text(); // must NOT throw — the guard converts the error into a recovered stream
    expect(text).toContain('"content\\":\\"part 1\\"'); // recovery emitted the complete call
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).toContain("[DONE]");
  });

  it("propagates a mid-flight cut as a stream ERROR after partial output was forwarded (no spliced duplicate)", async () => {
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            // Preamble + partial tool-call args, then the upstream dies
            // mid-generation (no terminal chunk ever arrives).
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
    // The preamble and the truncated tool-call fragments were already forwarded
    // live, so a recovery retry would splice a full replacement turn onto them —
    // duplicated prose, and the retry's tool call restarts at index:0 so the
    // client would concatenate the truncated old arguments with the new ones
    // into invalid JSON. The guard must error the stream honestly instead
    // (failover's committed-stream semantics) and never fire the retry.
    if (!res.body) throw new Error("expected a stream body");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let text = "";
    let streamErr: unknown = null;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
      }
    } catch (err) {
      streamErr = err;
    }
    if (!(streamErr instanceof Error)) throw new Error("expected the stream to error");
    expect(streamErr.message).toBe("terminated");
    expect(text).toContain("Створюю посібник"); // partial output was delivered before the failure
    expect(calls).toBe(1); // no recovery retry was attempted after partial delivery
  });

  it("recovers when the upstream ends CLEANLY mid-tool-arguments (no finish_reason chunk) — FINDING A", async () => {
    // The upstream streams a tool-call whose args are truncated, then closes the
    // SSE stream cleanly (just [DONE]) with NO finish_reason chunk. The buffered
    // call is unparseable; the guard must RECOVER a complete call instead of
    // emitting the truncated one to the client (which would drop the tool call
    // and stall the agent loop).
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            calls += 1;
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"recovered"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Truncated tool-call args split across two chunks, then a CLEAN close
            // ([DONE]) with no finish_reason chunk at all.
            return sseResponse([
              { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>Gui' } }] } }] },
              { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "de</h1><p>truncat" } }] } }] },
            ]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "write the full guide" }] }),
    );
    const text = await res.text();
    expect(calls).toBe(2); // recovery retry fired (original + retry)
    expect(text).toContain('"content\\":\\"recovered\\"'); // the recovered COMPLETE call
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).not.toContain("truncat"); // the broken buffered fragment never reached the client
    expect(text).toContain("[DONE]");
  });

  it("recovers a mid-flight cut that happens after ONLY buffered tool-call fragments (nothing client-visible) — FINDING B", async () => {
    // The upstream emits a truncated tool-call fragment (BUFFERED, never forwarded)
    // then dies mid-flight with no content ever reaching the client. Because option
    // B withholds tool fragments, the client is uncommitted — the guard must RECOVER
    // a clean call, not error the stream.
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            calls += 1;
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"recovered"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Only a truncated tool-call fragment, then the upstream terminates.
            return sseThenError(
              [
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_x", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html>' } }] } }] },
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
    const text = await res.text(); // must NOT throw — the cut was recoverable
    expect(calls).toBe(2); // original + recovery retry
    expect(text).toContain('"content\\":\\"recovered\\"'); // clean recovered call delivered
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).not.toContain("<html>"); // the buffered truncated fragment never reached the client
    expect(text).toContain("[DONE]");
  });

  it("delivers the buffered tool call when the upstream errors AFTER the terminal chunk was held (post-terminal cut) — FINDING C", async () => {
    // The terminal (finish_reason) chunk arrives and is held, THEN the upstream
    // errors before [DONE]. The buffered COMPLETE tool call must still reach the
    // client — otherwise it sees a terminal chunk with no tool call (actionless).
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            return sseThenError(
              [
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_c", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"done"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ],
              "terminated",
            );
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "make guide.html" }] }),
    );
    const text = await res.text(); // must NOT throw — terminal was already held, deliver it
    expect(calls).toBe(1); // no recovery — the buffered call was complete
    expect(text).toContain('"content\\":\\"done\\"'); // buffered tool call delivered
    expect(text).toContain('"finish_reason":"tool_calls"'); // held terminal chunk delivered
    expect(text).toContain("[DONE]");
  });

  it("RECOVERS a BROKEN terminal turn when the upstream errors AFTER the terminal chunk was held (post-terminal cut) — FINDING D", async () => {
    // The terminal finish_reason:"length" chunk arrives (buffered args are TRUNCATED),
    // THEN the connection errors before [DONE]. The turn is a normal finish that only
    // lost its trailing [DONE], so the shared terminal reconciliation must RECOVER the
    // broken call — NOT forward a dead/actionless terminal chunk.
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            calls += 1;
            const body = String(init?.body ?? "");
            if (body.includes("Emit the tool call NOW")) {
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"recovered"}' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            }
            // Truncated tool-call args + a terminal finish_reason:"length" chunk, THEN
            // the upstream errors before the trailing [DONE].
            return sseThenError(
              [
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_b", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>trunc' } }] } }] },
                { choices: [{ delta: {}, finish_reason: "length" }] },
              ],
              "terminated",
            );
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "write the full guide" }] }),
    );
    const text = await res.text(); // must NOT throw — the held terminal is reconciled
    expect(calls).toBe(2); // recovery retry fired (original + retry)
    expect(text).toContain('"content\\":\\"recovered\\"'); // the recovered COMPLETE call
    expect(text).toContain('"finish_reason":"tool_calls"'); // recovery's terminal, not the dead "length"
    expect(text).not.toContain('"finish_reason":"length"'); // the broken terminal was NOT forwarded
    expect(text).not.toContain("trunc"); // the truncated buffered fragment never reached the client
    expect(text).toContain("[DONE]");
  });

  it("forwards the content of a MIXED content+tool_calls chunk (tool_calls stripped) and keeps the recovery decision consistent — FINDING E", async () => {
    // A SINGLE delta chunk carries BOTH content ("partial answer ") AND a truncated
    // tool call, then the stream ends CLEANLY (no finish_reason chunk). The content
    // must reach the client (it is recorded in `content` state); if it were only
    // buffered, `nothingReachedClient()` would wrongly report the client committed —
    // losing the text while still declining recovery. The truncated tool fragment
    // must NOT be forwarded raw. Because the content genuinely reached the client,
    // the guard correctly declines a splice-recovery (which would duplicate the
    // prose) and closes honestly — the text is delivered exactly once, not lost.
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            // One mixed chunk (content + truncated tool-call args), then a clean [DONE]
            // with no finish_reason chunk at all.
            return sseResponse([
              { choices: [{ delta: { role: "assistant", content: "partial answer ", tool_calls: [{ index: 0, id: "call_m", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>tr' } }] } }] },
            ]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "answer then write" }] }),
    );
    const text = await res.text();
    // (a) the content part of the mixed chunk reached the client...
    expect(text).toContain("partial answer");
    // ...exactly once (not duplicated by a spurious recovery splice)...
    expect((text.match(/partial answer/g) ?? []).length).toBe(1);
    // ...and the truncated tool-call fragment was NEVER forwarded raw.
    expect(text).not.toContain("<html><h1>tr");
    // (b) recovery decision is consistent with what the client actually saw: content
    //     is committed, so the guard closes honestly WITHOUT a recovery retry.
    expect(calls).toBe(1);
    expect((text.match(/data: \[DONE\]/g) ?? []).length).toBe(1);
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

  it("emits exactly ONE [DONE] and NO recovery when the upstream ends with [DONE] but no finish_reason chunk AFTER forwarded content", async () => {
    // Post-release review finding: the guard used to forward the upstream [DONE]
    // and then append its own after the terminal-less recovery — double framing.
    // H6 follow-up: with partial content already forwarded, the recovery itself
    // was the splice — it re-delivered the whole answer a second time. Now the
    // guard just closes with its own single [DONE].
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
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
    expect(calls).toBe(1); // no recovery retry — "partial" already reached the client
    expect((text.match(/partial/g) ?? []).length).toBe(1); // delivered once, not duplicated
  });

  it("recovers when the upstream ends with [DONE] but no finish_reason chunk BEFORE anything was forwarded", async () => {
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
            // Empty upstream stream: [DONE] with no chunks at all. Nothing reached
            // the client, so the recovery retry is the whole answer — safe to run.
            return sseResponse([]);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    const text = await res.text();
    expect(text).toContain('"name":"write"'); // the recovered tool call
    const doneCount = (text.match(/data: \[DONE\]/g) ?? []).length;
    expect(doneCount).toBe(1);
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
