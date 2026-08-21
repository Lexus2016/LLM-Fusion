import { describe, it, expect } from "vitest";
import { z } from "zod";
import { OllamaClient } from "../src/upstream/ollama";
import { singleStrategy } from "../src/strategies/single";
import { detectIncompleteToolTurn, makeToolTurnGuardStream } from "../src/strategies/tool_turn_guard";
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

  it("detectIncompleteToolTurn: scalar and array tool arguments are not runnable", () => {
    // `arguments` is a JSON OBJECT string by protocol. A bare JSON.parse also accepts
    // `5`, `null`, `true` and `[1,2]` — and this surface used to, while the Anthropic
    // surface (which always required an object) answered `stop_reason: "max_tokens"`
    // with an empty `input` on the SAME upstream bytes. A Claude Code loop reads that
    // as truncation and retries forever against a deterministic upstream.
    const len = (args: string) => ({
      choices: [
        { finish_reason: "length", message: { role: "assistant", tool_calls: [{ id: "1", function: { name: "write", arguments: args } }] } },
      ],
    });
    for (const args of ["5", "null", "true", '"a"', "[1,2]"]) {
      expect(detectIncompleteToolTurn(len(args))).toBe("broken_tool_call");
    }
    // The object case is unchanged.
    expect(detectIncompleteToolTurn(len('{"path":"a.html"}'))).toBeNull();
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

  it("assembledCallsRunnable: a CUT stream never salvages scalar/array tool arguments (it recovers)", async () => {
    // The mid-flight-cut SALVAGE path judges the ASSEMBLED arguments. A bare
    // JSON.parse accepts `5`, `null` and `[1,2]` as "runnable", so the guard would
    // hand the client a complete-looking call whose input no tool can execute —
    // and, unlike the length-cut path, there is no terminal chunk left to tell the
    // client the turn was truncated. Only a JSON OBJECT is runnable input.
    const runCut = async (args: string) => {
      let calls = 0;
      const client = new OllamaClient({
        baseUrl: "https://mock.test",
        apiKey: "k",
        fetchFn: mockFetch([
          {
            match: (u) => u.endsWith("/v1/chat/completions"),
            respond: (_u, init) => {
              calls += 1;
              if (String(init?.body ?? "").includes("Emit the tool call NOW")) {
                return sseResponse([
                  { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: '{"path":"a.html","content":"recovered"}' } }] } }] },
                  { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
                ]);
              }
              // A WHOLE call (name + these arguments) is buffered, then the upstream
              // dies with no finish_reason chunk — the salvage decision point.
              return sseThenError(
                [{ choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_x", type: "function", function: { name: "write", arguments: args } }] } }] }],
                "terminated",
              );
            },
          },
        ]),
      });
      const res = await singleStrategy.execute(
        ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
      );
      // Drain FIRST: the recovery request is issued lazily while the stream is
      // read, so `calls` is only final once the body is consumed.
      const text = await res.text();
      return { calls, text };
    };

    for (const args of ["5", "null", "[1,2]"]) {
      const { calls, text } = await runCut(args);
      expect(calls).toBe(2); // original + recovery: the scalar/array call was NOT salvaged
      expect(text).toContain('"content\\":\\"recovered\\"');
      expect(text).not.toContain(`"arguments":${JSON.stringify(args)}`);
    }
    // Control: a real JSON OBJECT survives the cut and IS salvaged, no retry.
    const ok = await runCut('{"a":1}');
    expect(ok.calls).toBe(1);
    expect(ok.text).toContain('"arguments":"{\\"a\\":1}"');
    expect(ok.text).toContain("[DONE]");
  });

  it("assembledCallsEmittable: a CLEAN tool_calls finish recovers scalar/array arguments but emits empty ones", async () => {
    // The clean-finish check is deliberately LOOSER than the salvage one (empty
    // arguments are a legitimate no-arg tool), but it must still refuse a
    // non-empty argument string that is not a JSON object: `null` / `[1,2]` under
    // finish_reason:"tool_calls" are invisible to detectIncompleteToolTurn (which
    // only inspects broken args for "length"), so this predicate is the only thing
    // standing between the client and an unexecutable call.
    const runClean = async (args: string) => {
      let calls = 0;
      const client = new OllamaClient({
        baseUrl: "https://mock.test",
        apiKey: "k",
        fetchFn: mockFetch([
          {
            match: (u) => u.endsWith("/v1/chat/completions"),
            respond: (_u, init) => {
              calls += 1;
              if (String(init?.body ?? "").includes("Emit the tool call NOW")) {
                return sseResponse([
                  { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: '{"path":"a.html","content":"recovered"}' } }] } }] },
                  { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
                ]);
              }
              // A CLEAN terminal finish carrying the assembled call.
              return sseResponse([
                { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write", arguments: args } }] } }] },
                { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
              ]);
            },
          },
        ]),
      });
      const res = await singleStrategy.execute(
        ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
      );
      // Drain FIRST: the recovery request is issued lazily while the stream is
      // read, so `calls` is only final once the body is consumed.
      const text = await res.text();
      return { calls, text };
    };

    for (const args of ["null", "[1,2]"]) {
      const { calls, text } = await runClean(args);
      expect(calls).toBe(2); // the turn went to recovery instead of being emitted as-is
      expect(text).toContain('"content\\":\\"recovered\\"');
      expect(text).not.toContain(`"arguments":${JSON.stringify(args)}`);
    }
    // Control: a JSON object is emitted untouched.
    const obj = await runClean("{}");
    expect(obj.calls).toBe(1);
    expect(obj.text).toContain('"arguments":"{}"');
    // Control: a no-arg tool on a clean finish sends `arguments: ""` — emittable by
    // design. Tightening this predicate to reject empty args would send every
    // no-arg call to a pointless billed retry.
    const noArg = await runClean("");
    expect(noArg.calls).toBe(1);
    expect(noArg.text).toContain('"arguments":""');
    expect(noArg.text).toContain('"finish_reason":"tool_calls"');
  });

  it("fail-open terminal honesty: a DROPPED broken call rewrites ONLY a tool_calls terminal to \"length\" (stop/narrate terminals survive, space-less `data:` framing included)", async () => {
    // Fail-open path: the streaming recovery never reached the client (the retry
    // stream carries no data chunks at all), so the guard delivers the ORIGINAL
    // held terminal. When the buffered call was DROPPED as unrunnable the turn
    // then contains nothing to execute, and a surviving `finish_reason:"tool_calls"`
    // announces a call the client cannot find — it must be rewritten to the cut it
    // actually was. The other three cases pin how NARROW that rewrite is: it fires
    // only on `tool_calls` (2: a dropped call under `stop` keeps `stop` — inventing
    // a token cap that was never hit would be a second lie), never without a
    // dropped call (3), and it must survive a `data:{...}` frame with no space
    // after the colon (4), which is what `terminalLine` holds verbatim whenever the
    // terminal chunk carried no tool_call fragments of its own.
    //
    // NOT covered, because the state does not exist: a fail-open with an EMITTABLE
    // buffered call (`dropped === false` while `assembledCalls !== undefined`).
    // `incomplete` can only be non-null with calls present when
    // detectIncompleteToolTurn hits its `fin === "length" && toolCallArgsBroken`
    // branch — and "args broken" is the exact negation of "emittable", so the two
    // cannot hold at once (every other finish_reason returns null there and falls
    // through to the `!assembledCallsEmittable` fallback, which sets `dropped`).
    const rawSse = (lines: string[]): Response =>
      new Response(lines.join("") + "data: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    const runFailOpen = async (original: unknown[], frame?: (chunks: unknown[]) => Response) => {
      let calls = 0;
      const client = new OllamaClient({
        baseUrl: "https://mock.test",
        apiKey: "k",
        fetchFn: mockFetch([
          {
            match: (u) => u.endsWith("/v1/chat/completions"),
            respond: (_u, init) => {
              calls += 1;
              // The recovery retry answers with an EMPTY stream ([DONE] only): zero
              // data chunks forwarded -> streamRetryToolTurn returns false -> fail open.
              if (String(init?.body ?? "").includes("Emit the tool call NOW")) return sseResponse([]);
              return (frame ?? sseResponse)(original);
            },
          },
        ]),
      });
      const res = await singleStrategy.execute(
        ctxWith(client, { model: "fast-glm", stream: true, tools: TOOLS, messages: [{ role: "user", content: "make guide.html" }] }),
      );
      const text = await res.text();
      return { calls, text };
    };

    // 1) Broken (unrunnable) buffered call under a `tool_calls` finish + failed recovery.
    const broken = await runFailOpen([
      { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_b", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>trunc' } }] } }] },
      { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
    ]);
    expect(broken.calls).toBe(2); // original + the (useless) recovery retry
    expect(broken.text).toContain('"finish_reason":"length"'); // rewritten: the turn was cut, not acted on
    expect(broken.text).not.toContain('"finish_reason":"tool_calls"'); // never announce a call that is not in the payload
    expect(broken.text).not.toContain("tool_calls"); // the unrunnable call was dropped, not emitted
    expect(broken.text).not.toContain("trunc"); // ...and its buffered fragment never reached the client
    expect(broken.text).toContain("[DONE]");

    // 2) Control — narrate-and-stop: no buffered call at all (assembledCalls ===
    // undefined), so nothing was dropped and the ORIGINAL finish_reason must survive.
    const narrate = await runFailOpen([
      { choices: [{ delta: { role: "assistant", content: "Let me write the complete HTML file now." } }] },
      { choices: [{ delta: {}, finish_reason: "stop" }] },
    ]);
    expect(narrate.calls).toBe(2); // recovery was attempted here too, and also failed
    expect(narrate.text).toContain("Let me write the complete HTML file now."); // prose delivered live
    expect(narrate.text).toContain('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'); // terminal untouched
    expect(narrate.text).not.toContain('"finish_reason":"length"'); // the rewrite must not fire here
    expect(narrate.text).toContain("[DONE]");

    // 3) A DROPPED broken call under a `stop` finish (prose + a truncated call:
    // detectIncompleteToolTurn returns null for stop-with-calls, so the
    // `!assembledCallsEmittable` fallback classifies it). The call is still dropped,
    // but `stop` already describes the turn honestly — rewriting it to "length"
    // would fabricate a token cap that was never hit.
    const stopDrop = await runFailOpen([
      { choices: [{ delta: { role: "assistant", content: "Let me write the complete HTML file now." } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_s", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>trunc' } }] } }] },
      { choices: [{ delta: {}, finish_reason: "stop" }] },
    ]);
    expect(stopDrop.calls).toBe(2); // recovery attempted (broken_tool_call) and failed
    expect(stopDrop.text).toContain('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'); // untouched
    expect(stopDrop.text).not.toContain('"finish_reason":"length"'); // no invented token cap
    expect(stopDrop.text).not.toContain("tool_calls"); // the broken call was still dropped
    expect(stopDrop.text).not.toContain("trunc");

    // 4) Same dropped-call rewrite, but the upstream frames its chunks as
    // `data:{...}` with NO space. That line is held VERBATIM as the terminal (the
    // chunk carries no tool_call fragments, so nothing re-serializes it), so a
    // prefix check stricter than handleLine's own would silently skip the rewrite.
    const noSpace = await runFailOpen(
      [
        { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_n", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>trunc' } }] } }] },
        { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
      ],
      (chunks) => rawSse(chunks.map((c) => `data:${JSON.stringify(c)}\n\n`)),
    );
    expect(noSpace.calls).toBe(2);
    expect(noSpace.text).toContain('"finish_reason":"length"'); // rewritten despite the space-less frame
    expect(noSpace.text).not.toContain('"finish_reason":"tool_calls"');
    expect(noSpace.text).not.toContain("tool_calls");
    expect(noSpace.text).toContain("[DONE]");
  });
});

describe("tool-turn guard — upstream backpressure", () => {
  it("stops pulling upstream once the client stops reading", async () => {
    // The guard used to drain the whole upstream inside ReadableStream.start(), which
    // runs to completion whether or not anyone reads and whose enqueue() never blocks.
    // A client that stalls (slow terminal, paused agent) therefore parked an entire
    // generation in the stream queue: ~195 B per SSE line, so a 32k-token answer is
    // ~6 MiB of wire and several times that in heap — per stalled connection, times
    // upstream.max_concurrency.
    const N = 5000;
    let pulled = 0;
    const enc = new TextEncoder();
    const upstream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pulled >= N) {
          controller.close();
          return;
        }
        pulled++;
        const chunk = { choices: [{ delta: { content: "x".repeat(180) } }] };
        controller.enqueue(enc.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      },
    });

    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn: mockFetch([]) });
    const request: ChatCompletionRequest = { model: "fast-glm", messages: [{ role: "user", content: "hi" }], stream: true };
    const guarded = makeToolTurnGuardStream(ctxWith(client, request), undefined, "glm-5.2", { ...request }, upstream);

    const reader = guarded.getReader();
    await reader.read();
    await reader.read();
    // Give any eager drain loop every chance to run to completion.
    for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0));

    // A handful of chunks in flight, not the whole generation.
    expect(pulled).toBeLessThan(20);
    expect(pulled).toBeGreaterThan(0);

    await reader.cancel();
  });

  it("releases the upstream generation and fires NO recovery when the client cancels while pull() is parked", async () => {
    // Same parked state as above, but now the client walks away. Two things must
    // happen: the upstream reader is cancelled (otherwise a cloud generation keeps
    // burning tokens with nobody at the other end), and NO recovery request is
    // issued — a departed client is not a broken turn.
    let recoveries = 0;
    let upstreamCancelled = false;
    const enc = new TextEncoder();
    // Keepalive comments are forwarded verbatim (they are not `data:` chunks), so
    // they fill the queue and park pull() while leaving `content` empty — i.e. the
    // guard would still judge a recovery "safe" here if it wrongly ran one.
    const upstream = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(enc.encode(": keepalive\n\n"));
      },
      cancel() {
        upstreamCancelled = true;
      },
    });

    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            recoveries += 1;
            return sseResponse([{ choices: [{ delta: {}, finish_reason: "tool_calls" }] }]);
          },
        },
      ]),
    });
    const request: ChatCompletionRequest = { model: "fast-glm", messages: [{ role: "user", content: "hi" }], stream: true };
    const guarded = makeToolTurnGuardStream(ctxWith(client, request), undefined, "glm-5.2", { ...request }, upstream);

    const reader = guarded.getReader();
    await reader.read();
    await reader.read();
    for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0)); // let pull() park on the full queue
    await reader.cancel();
    for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0)); // and let anything it scheduled run

    expect(upstreamCancelled).toBe(true); // the generation is released, not left running
    expect(recoveries).toBe(0);
  });
});

describe("tool-turn guard — client disconnect mid-pull", () => {
  /** The guard's only upstream route is its recovery retry — counted, never expected here. */
  const recoveryCountingClient = () => {
    let recoveries = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            recoveries += 1;
            return sseResponse([
              { choices: [{ delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_r", type: "function", function: { name: "write", arguments: "{}" } }] } }] },
              { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
            ]);
          },
        },
      ]),
    });
    return { client, recoveries: () => recoveries };
  };

  /**
   * A hand-driven upstream: it delivers only when the test says so. highWaterMark 0
   * means its `pull` fires exactly when the guard issues `reader.read()` on an empty
   * queue — i.e. `state.pulls` is the proof that pull() is parked inside that read
   * (without it a push/cancel pair can race past a loop that never started).
   */
  const drivenUpstream = () => {
    const enc = new TextEncoder();
    const state = { cancelled: false, pulls: 0, push: (_line: string): void => {} };
    const stream = new ReadableStream<Uint8Array>(
      {
        start(controller) {
          state.push = (line) => {
            try {
              controller.enqueue(enc.encode(line));
            } catch {
              /* already cancelled — a late upstream chunk has nowhere to go */
            }
          };
        },
        pull() {
          state.pulls += 1;
        },
        cancel() {
          state.cancelled = true;
        },
      },
      { highWaterMark: 0 },
    );
    return { stream, state };
  };

  const settle = async (): Promise<void> => {
    for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
  };

  const request: ChatCompletionRequest = { model: "fast-glm", messages: [{ role: "user", content: "hi" }], stream: true };

  it("cancel while a read is IN FLIGHT ends the turn silently (no recovery, no unhandled rejection)", async () => {
    // reader.cancel() resolves the pull loop's pending reader.read() with
    // {done: true}, so the loop breaks and lands in the normal-finish path looking
    // exactly like an upstream that finished. Without the `ended` latch, the guard
    // would reconcile the turn for a client that is already gone: a full upstream
    // recovery generation nobody will read, then a throw on controller.enqueue.
    const { client, recoveries } = recoveryCountingClient();
    const { stream, state } = drivenUpstream();
    const guarded = makeToolTurnGuardStream(ctxWith(client, request), undefined, "glm-5.2", { ...request }, stream);

    const rejections: unknown[] = [];
    const onRejection = (err: unknown): void => {
      rejections.push(err);
    };
    process.on("unhandledRejection", onRejection);
    try {
      const reader = guarded.getReader();
      const pending = reader.read();
      await settle();
      expect(state.pulls).toBe(1); // pull() is parked inside await reader.read()
      // A truncated tool-call fragment: BUFFERED, never forwarded, so `content`
      // stays empty and the guard still considers recovery safe — exactly the
      // state in which a wrongly-resumed finish path bills a whole generation.
      // (one "\n": the trailing blank separator line is forwarded verbatim by
      // handleLine, and delivering it would resolve the client's read early —
      // holding it back keeps pull() demonstrably parked inside reader.read().)
      state.push('data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"write","arguments":"{\\"path\\":\\"a"}}]}}]}\n');
      await settle();
      expect(state.pulls).toBe(2); // fragment consumed, parked on the NEXT read
      await reader.cancel(); // the client leaves while that read is in flight
      state.push('data: {"choices":[{"delta":{"content":"too late"}}]}\n'); // upstream chunk after the departure
      await settle();

      expect(await pending).toEqual({ done: true, value: undefined });
      expect(state.cancelled).toBe(true);
      expect(recoveries()).toBe(0); // a departed client is never worth a billed retry
      expect(rejections).toEqual([]);
    } finally {
      process.off("unhandledRejection", onRejection);
    }
  });

  it("a cancel landing between the upstream read and handleLine does not bill a recovery", async () => {
    // The narrow window: the consumer dies WHILE await reader.read() is in flight,
    // so the loop resumes holding a real chunk for a controller that is already
    // gone. Handing that line to handleLine first throws inside enqueue, and the
    // catch reads that throw as an upstream cut — billing a recovery for a client
    // that already left. The client-gone check therefore sits at the TOP of the
    // loop, ahead of handleLine.
    const { client, recoveries } = recoveryCountingClient();
    const { stream, state } = drivenUpstream();
    const guarded = makeToolTurnGuardStream(ctxWith(client, request), undefined, "glm-5.2", { ...request }, stream);

    const reader = guarded.getReader();
    const pending = reader.read();
    await settle();
    expect(state.pulls).toBe(1); // pull() is parked inside await reader.read()
    // Same tick, no await in between: the enqueue fulfils that pending read (its
    // continuation is only queued as a microtask) and the cancel runs the guard's
    // cancel hook synchronously, BEFORE the loop resumes with the chunk in hand.
    state.push(": keepalive\n\n"); // a comment line — handleLine would forward it verbatim
    const cancelled = reader.cancel();
    await cancelled;
    await settle();

    expect(await pending).toEqual({ done: true, value: undefined });
    expect(state.cancelled).toBe(true);
    expect(recoveries()).toBe(0);
  });

  it("stops reading upstream and fires NO recovery when a downstream sink errors mid-stream", async () => {
    // The client-side death that is not an explicit cancel: the guard's bytes are
    // piped onward and the destination blows up mid-turn. pipeTo cancels the
    // source, so the guard sees the same departure as an explicit cancel — it must
    // stop pulling the upstream and must not treat the departure as a broken turn.
    // (Per the streams spec a consumer can only ever CLOSE the guard's stream, so
    // the sibling `desiredSize === null` check covers the errored-controller case
    // rather than this one.)
    const { client, recoveries } = recoveryCountingClient();
    const { stream, state } = drivenUpstream();
    const guarded = makeToolTurnGuardStream(ctxWith(client, request), undefined, "glm-5.2", { ...request }, stream);

    const piped = guarded.pipeTo(
      new WritableStream<Uint8Array>({
        write() {
          throw new Error("client socket gone");
        },
      }),
    );
    let pipeErr: unknown = null;
    piped.catch((err) => {
      pipeErr = err;
    });

    await settle();
    expect(state.pulls).toBe(1); // pull() is parked inside await reader.read()
    // ONE line, so the guard's single enqueue goes straight to the pipe's pending
    // read (desiredSize untouched) and the loop parks on the NEXT upstream read
    // rather than on backpressure — the state where a departure is misread.
    state.push(": keepalive\n"); // first byte reaches the sink -> the sink throws
    await settle();
    state.push('data: {"choices":[{"delta":{"content":"still generating"}}]}\n\n'); // upstream keeps talking
    await settle();

    if (!(pipeErr instanceof Error)) throw new Error("expected the pipe to reject");
    expect(pipeErr.message).toBe("client socket gone");
    expect(state.cancelled).toBe(true); // the guard released the upstream
    expect(recoveries()).toBe(0);
  });
});
