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

    // 5) A dropped broken call under `finish_reason:"content_filter"`. Same shape as
    // case 3 but with the OTHER non-tool terminal the OpenAI API defines, and it is
    // the case that pins the predicate rather than the outcome: `!== "tool_calls"`
    // and `=== "stop"` agree on cases 1-4, so a mutant that narrows the guard to
    // `=== "stop"` survived the whole suite. Here they disagree — the mutant would
    // rewrite `content_filter` to `"length"`, erasing the one signal that tells the
    // client the turn was refused rather than truncated, and turning a refusal into
    // an Anthropic `max_tokens` that invites an auto-continuation loop.
    const filtered = await runFailOpen([
      { choices: [{ delta: { role: "assistant", content: "I cannot help with that." } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_f", type: "function", function: { name: "write", arguments: '{"path":"guide.html","content":"<html><h1>trunc' } }] } }] },
      { choices: [{ delta: {}, finish_reason: "content_filter" }] },
    ]);
    // 1, not 2: `content_filter` is on NO_RETRY_FINISH_REASONS — the refusal is not
    // re-prompted. The drop-and-preserve behaviour this case pins is unaffected.
    expect(filtered.calls).toBe(1);
    expect(filtered.text).toContain('data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}'); // untouched
    expect(filtered.text).not.toContain('"finish_reason":"length"');
    expect(filtered.text).not.toContain("tool_calls"); // the broken call was still dropped
    expect(filtered.text).not.toContain("trunc");
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

// ---------------------------------------------------------------------------
// ADVERSARIAL REVIEW (R6-T2) — each `it` below encodes the behaviour the guard
// SHOULD have and currently does not. They are expected to FAIL until fixed.
// ---------------------------------------------------------------------------
describe("tool-turn guard — adversarial review regressions (D1-D6)", () => {
  const TOOLS = [{ type: "function", function: { name: "write", parameters: { type: "object" } } }];

  const rawSse = (frames: string[]): Response =>
    new Response(frames.join("") + "data: [DONE]\n\n", {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });

  /**
   * Drive one guarded streaming turn with EXTRA request fields (e.g. `n`);
   * `recovery` builds the retry response.
   */
  const runGuardedRequest = async (
    extra: Record<string, unknown>,
    original: () => Response,
    recovery: () => Response = () => sseResponse([]),
  ): Promise<{ calls: number; text: string }> => {
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: (_u, init) => {
            calls += 1;
            if (String(init?.body ?? "").includes("Emit the tool call NOW")) return recovery();
            return original();
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, {
        model: "fast-glm",
        stream: true,
        tools: TOOLS,
        messages: [{ role: "user", content: "make guide.html" }],
        ...extra,
      }),
    );
    const text = await res.text();
    return { calls, text };
  };

  /** Drive one guarded streaming turn; `recovery` builds the retry response. */
  const runGuarded = (
    original: () => Response,
    recovery: () => Response = () => sseResponse([]),
  ): Promise<{ calls: number; text: string }> => runGuardedRequest({}, original, recovery);

  /** Every `finish_reason` string present in the guard's output, in order. */
  const finishReasonsOf = (text: string): string[] => {
    const out: string[] = [];
    for (const m of text.matchAll(/"finish_reason":\s*"([^"]*)"/g)) {
      if (m[1] !== undefined) out.push(m[1]);
    }
    return out;
  };

  const brokenCallDelta = (choiceIndex: number, id: string) => ({
    index: choiceIndex,
    delta: {
      tool_calls: [
        {
          index: 0,
          id,
          type: "function",
          function: { name: "write", arguments: '{"path":"guide.html","content":"<html>trunc' },
        },
      ],
    },
  });

  it("D1: an n>1 stream latches the guard OFF — no half-applied terminal rewrite", async () => {
    // `n` is passthrough (ChatCompletionRequestSchema is .passthrough(), and single.ts
    // forwards the request body), so a multi-choice answer is reachable.
    // markTerminalLengthCut inspects and mutates `choices[0]` only, so it used to
    // rewrite choice 0 to "length" while choice 1 kept `finish_reason:"tool_calls"` —
    // one chunk telling the client both that a call is coming and that it is not.
    // The fix is the multiChoice latch: the guard becomes a pure passthrough, so the
    // terminal stays exactly as the upstream sent it, self-consistent.
    const terminal = {
      choices: [
        { index: 0, delta: {}, finish_reason: "tool_calls" },
        { index: 1, delta: {}, finish_reason: "tool_calls" },
      ],
    };
    const { calls, text } = await runGuarded(() =>
      rawSse([
        `data: ${JSON.stringify({ choices: [brokenCallDelta(0, "c0"), brokenCallDelta(1, "c1")] })}\n\n`,
        `data: ${JSON.stringify(terminal)}\n\n`,
      ]),
    );
    expect(calls).toBe(1); // no recovery retry: the guard did not judge this turn
    expect(text).toContain(JSON.stringify(terminal)); // byte-identical terminal
    expect(text).not.toContain('"finish_reason":"length"'); // nothing half-rewritten
    expect(finishReasonsOf(text)).toEqual(["tool_calls", "tool_calls"]);
  });

  it("D2: an n>1 stream latches OFF, so a choices[1] delta is no longer swallowed by choices[0]", async () => {
    // handleLine judged the whole CHUNK by `choices[0]`: a chunk whose first choice
    // carried tool-call fragments took the option-B suppression path and was dropped
    // wholesale — silently destroying whatever `choices[1]` carried in the same chunk.
    // (And stripToolCallsFromChunk deletes `delta.tool_calls` from `choices[0]` only,
    // so the mirror case leaked a truncated fragment instead.) The latch makes the
    // guard forward the chunk VERBATIM, which is the only correct answer for a stream
    // its single-choice state machine cannot model.
    const chunk = {
      choices: [brokenCallDelta(0, "c0"), { index: 1, delta: { content: "choice-1 answer" } }],
    };
    const { calls, text } = await runGuarded(() =>
      rawSse([
        `data: ${JSON.stringify(chunk)}\n\n`,
        `data: ${JSON.stringify({
          choices: [
            { index: 0, delta: {}, finish_reason: "tool_calls" },
            { index: 1, delta: {}, finish_reason: "stop" },
          ],
        })}\n\n`,
      ]),
    );
    expect(text).toContain("choice-1 answer"); // was dropped with the whole chunk
    expect(calls).toBe(1);
    expect(text).toContain(JSON.stringify(chunk)); // verbatim, fragment still at index 1
    expect(finishReasonsOf(text)).toEqual(["tool_calls", "stop"]); // nothing rewritten
  });

  it("D2b: the multiChoice latch does NOT mis-fire on a single-choice stream that omits `index`", async () => {
    // `index` is optional in practice (Ollama and llama.cpp omit it). Latching on
    // `choices[0].index !== 0` without a typeof check would read `undefined !== 0` as
    // true and disable the guard for every such upstream. Here the guard must stay ON:
    // buffer the broken call, withhold it, and run recovery.
    const { calls, text } = await runGuarded(
      () =>
        rawSse([
          `data: ${JSON.stringify({
            choices: [
              {
                delta: {
                  tool_calls: [
                    { index: 0, id: "c", type: "function", function: { name: "write", arguments: '{"path":"a","content":"trunc' } },
                  ],
                },
              },
            ],
          })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
        ]),
      () =>
        sseResponse([
          {
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    { index: 0, id: "r", type: "function", function: { name: "write", arguments: '{"path":"a.html","content":"ok"}' } },
                  ],
                },
              },
            ],
          },
          { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
        ]),
    );
    expect(calls).toBe(2); // the guard fired recovery, so it was NOT latched off
    expect(text).not.toContain("trunc"); // the broken fragment never reached the client
    expect(text).toContain('"content\\":\\"ok');
  });

  /** The original turn every D3 case recovers from: one broken call + a tool_calls terminal. */
  const brokenOriginal = () =>
    sseResponse([
      { choices: [brokenCallDelta(0, "c")] },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
    ]);

  it("R1a: a SEQUENTIAL n>1 stream (one choice per chunk) latches OFF before withholding anything", async () => {
    // The regression the D1/D2 tests missed: a real OpenAI-compatible n>1 stream sends
    // ONE choice per chunk, so `choices.length > 1` never fires. Chunk 1 (index 0)
    // looked single-choice, option B withheld its tool fragments; chunk 2 (index 1)
    // latched the guard into passthrough; then finishNormally's multiChoice early-out
    // skipped emitAssembledToolCalls and the withheld call was LOST. The latch has to
    // come from the REQUEST (`n`), before any chunk is seen.
    const { calls, text } = await runGuardedRequest(
      { n: 2 },
      () =>
        rawSse([
          `data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "c0", type: "function", function: { name: "write", arguments: '{"path":"a.html"}' } }] } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ index: 1, delta: { content: "choice-1 answer" } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ index: 1, delta: {}, finish_reason: "stop" }] })}\n\n`,
        ]),
    );
    expect(calls).toBe(1);
    expect(text).toContain('"path\\":\\"a.html'); // choice 0's call survived
    expect(text).toContain("choice-1 answer");
    expect(finishReasonsOf(text)).toEqual(["tool_calls", "stop"]);
  });

  it("R1b: choice 0's terminal arriving FIRST does not discard the later choice-1 chunks", async () => {
    // The worse ordering: `terminalLine !== null` is checked ahead of the latch, so
    // once choice 0 finished, every subsequent choice-1 chunk was silently swallowed
    // by the held-terminal path.
    const { calls, text } = await runGuardedRequest(
      { n: 2 },
      () =>
        rawSse([
          `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "choice-0 answer" } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ index: 1, delta: { content: "choice-1 answer" } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ index: 1, delta: {}, finish_reason: "stop" }] })}\n\n`,
        ]),
    );
    expect(calls).toBe(1);
    expect(text).toContain("choice-0 answer");
    expect(text).toContain("choice-1 answer");
  });

  it("R1c: the first-chunk detector still catches an upstream that returns n>1 unasked, including a STRING index", async () => {
    // Secondary safety net: the request said nothing about `n`, but the upstream
    // answered with a second choice anyway. `index` is also accepted as a numeric
    // STRING here — some gateways serialise it that way, and a `typeof === "number"`
    // test alone would let the guard keep judging a stream it cannot model.
    // The stream carries a BROKEN call on purpose: that is what discriminates a
    // latched-off guard (pure passthrough, one upstream call, fragment forwarded
    // verbatim) from a live one (fragment withheld, recovery fired, 2 calls).
    const { calls, text } = await runGuarded(() =>
      rawSse([
        `data: ${JSON.stringify({ choices: [{ index: "1", delta: { tool_calls: [{ index: 0, id: "c", type: "function", function: { name: "write", arguments: '{"path":"a","content":"trunc' } }] } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [{ index: "1", delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
      ]),
    );
    expect(calls).toBe(1); // latched off: no recovery attempt
    expect(text).toContain("trunc"); // passthrough is verbatim, warts and all
    expect(finishReasonsOf(text)).toEqual(["tool_calls"]); // no terminal rewrite either
  });

  it("R1d: a LATE latch hands back what option B was holding instead of dropping it", async () => {
    // Unasked n>1 in its worst ordering: choice 0 completes (tool fragments withheld,
    // terminal held) before choice 1 shows up at all. The latch cannot fire any
    // earlier, so it has to FLUSH — assembled call first, then the held terminal —
    // before passthrough resumes, and the detector has to run ahead of the
    // held-terminal early-return or the choice-1 chunks never reach it.
    const { calls, text } = await runGuarded(() =>
      rawSse([
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "c0", type: "function", function: { name: "write", arguments: '{"path":"a.html"}' } }] } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [{ index: 1, delta: { content: "choice-1 answer" } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [{ index: 1, delta: {}, finish_reason: "stop" }] })}\n\n`,
      ]),
    );
    expect(calls).toBe(1); // no recovery: the turn was runnable, just multi-choice
    expect(text).toContain('"path\\":\\"a.html'); // withheld call flushed, not lost
    expect(text).toContain("choice-1 answer"); // not swallowed by the held terminal
    expect(finishReasonsOf(text)).toEqual(["tool_calls", "stop"]);
  });

  it("R1e: an n>1 stream that omits `index` ENTIRELY is only survivable via the request latch", async () => {
    // The case that makes the construction-time latch load-bearing rather than
    // belt-and-braces. R1a/R1b are killed by either mechanism (the late latch flushes
    // what it held, so the outcome is right by luck of the wire format); here there is
    // no `index` on any chunk and `choices.length` is always 1, so NOTHING in the
    // response distinguishes two sequential choices from one — Ollama and llama.cpp
    // both omit `index`, which is precisely why the guard tolerates its absence. With
    // the guard live, choice 0's terminal is held and every later line is swallowed:
    // choice 1 vanishes. `n` from the request is the only signal that exists.
    // Both spellings of `n`: some gateways re-serialise query/form params as strings,
    // and a `typeof === "number"` read would silently disarm the latch for those.
    for (const n of [2, "2"]) {
      const { calls, text } = await runGuardedRequest({ n }, () =>
        rawSse([
          `data: ${JSON.stringify({ choices: [{ delta: { content: "answer A" } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ delta: { content: "answer B" } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`,
        ]),
      );
      expect(calls).toBe(1);
      expect(text).toContain("answer A");
      expect(text).toContain("answer B");
      expect(finishReasonsOf(text)).toEqual(["stop", "stop"]);
    }
  });

  it("R1f: the request latch does NOT mis-fire on a single-choice stream (n absent, n:1, no `index`)", async () => {
    // The other half of R1e. A latch read from the request is only safe if it stays
    // off for every ordinary turn — `n` absent, `n: 1`, and chunks with no `index`
    // field at all (Ollama, llama.cpp). If it mis-fired the guard would be disabled
    // proxy-wide and every broken tool call would ship untouched.
    for (const extra of [{}, { n: 1 }]) {
      const { calls, text } = await runGuardedRequest(extra, () =>
        rawSse([
          `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: "c", type: "function", function: { name: "write", arguments: '{"path":"a","content":"trunc' } }] } }] })}\n\n`,
          `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
        ]),
      );
      expect(calls).toBe(2); // guard live: recovery attempted
      expect(text).not.toContain("trunc"); // fragment withheld, not passed through
      expect(finishReasonsOf(text)).toEqual(["length"]);
    }
  });

  it("R2: a recovery that sends ONLY a bare terminal is NOT recovery — the guard fails open", async () => {
    // A retry answering with nothing but `finish_reason:"tool_calls"` delivers an EMPTY
    // tool turn: no call to run, no prose. Counting it as a replacement suppressed the
    // held original terminal AND skipped the fail-open rewrite, handing the client
    // exactly the actionless tool_calls terminal this guard exists to eliminate.
    // Classification now happens BEFORE the enqueue, so a bare terminal is neither
    // forwarded nor counted, and no double-terminal is possible.
    for (const fin of ["tool_calls", "stop"]) {
      const { calls, text } = await runGuarded(brokenOriginal, () =>
        sseResponse([{ choices: [{ index: 0, delta: {}, finish_reason: fin }] }]),
      );
      expect(calls).toBe(2);
      expect(finishReasonsOf(text)).toEqual(["length"]); // fail open, terminal made honest
      expect(text).not.toContain("trunc");
    }
  });

  it("R3: a whitespace-only tool name is not dispatchable", () => {
    const call = (name: string) => ({
      choices: [
        {
          finish_reason: "tool_calls",
          message: { role: "assistant", content: null, tool_calls: [{ id: "c", type: "function", function: { name, arguments: "{}" } }] },
        },
      ],
    });
    expect(detectIncompleteToolTurn(call("   "))).toBe("broken_tool_call");
    expect(detectIncompleteToolTurn(call("\t\n"))).toBe("broken_tool_call");
    expect(detectIncompleteToolTurn(call("write"))).toBeNull();
  });

  it("R3b: a whitespace-only name is dropped on the fail-open path too, not shipped as runnable", async () => {
    // Parity with R3 one layer down. `assembledCallsEmittable` decides whether the
    // buffered call is handed to the client when recovery fails; a bare truthiness
    // test on the name passes `"   "` and ships an undispatchable call under a
    // `tool_calls` terminal — exactly the actionless turn the guard exists to stop.
    const { text } = await runGuarded(() =>
      rawSse([
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "c", type: "function", function: { name: "   ", arguments: '{"path":"a"}' } }] } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
      ]),
    );
    expect(finishReasonsOf(text)).toEqual(["length"]); // dropped, so the terminal is corrected
    expect(text).not.toContain('"path":"a"');
  });

  it("R3c: a synthesised terminal calls a whitespace-named recovery call unrunnable, not `tool_calls`", async () => {
    // Third and last name check: `assembledCallsRunnable` picks the finish_reason when
    // a retry produces calls but never terminates. Announcing `tool_calls` for a call
    // named "  " tells the client to dispatch a tool that cannot be looked up.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseResponse([
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "r", type: "function", function: { name: "  ", arguments: '{"path":"a"}' } }] } }] },
      ]),
    );
    expect(calls).toBe(2);
    expect(finishReasonsOf(text)).toEqual(["length"]);
  });

  it("R4a: a partially-valid parallel call SET is retried as a whole, matching the stream path", async () => {
    // One runnable `read` plus a nameless tail. The set is retried rather than shipped,
    // because a client that rejects the malformed entry rejects the whole assistant
    // message — and the STREAM path has always behaved this way (assembledCallsEmittable
    // requires EVERY call to be emittable). The valid call is not lost if the retry
    // fails: retryToolTurn returns null and the ORIGINAL response ships untouched.
    const partial = {
      id: "x",
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              { id: "c1", type: "function", function: { name: "read", arguments: '{"path":"a"}' } },
              { id: "c2", type: "function", function: { arguments: '{"path":"b"}' } },
            ],
          },
        },
      ],
    };
    expect(detectIncompleteToolTurn(partial)).toBe("broken_tool_call");

    // ...and when the retry fails, the original set survives verbatim.
    let calls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            calls += 1;
            return jsonResponse(partial);
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    const body = await res.text();
    expect(calls).toBe(2);
    expect(body).toContain('"name":"read"'); // the valid call was not lost
  });

  it("R4b: a content_filter turn is never re-prompted, on either path", async () => {
    // "Emit the tool call NOW" against a turn the upstream REFUSED re-runs the refusal:
    // it burns a call, and at best returns the same refusal. The broken call is still
    // dropped and `content_filter` still survives on the stream path — the client is
    // told, honestly, that the turn was filtered and carries nothing to execute.
    const filteredCalls = [{ id: "c", type: "function", function: { name: "write", arguments: '{"path":"a","content":"trunc' } }];

    // Stream path: no retry, call dropped, terminal preserved.
    const { calls, text } = await runGuarded(() =>
      sseResponse([
        { choices: [{ index: 0, delta: { role: "assistant", content: "I cannot help with that." } }] },
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, ...filteredCalls[0] }] } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: "content_filter" }] },
      ]),
    );
    expect(calls).toBe(1); // recovery NOT attempted
    expect(finishReasonsOf(text)).toEqual(["content_filter"]);
    expect(text).not.toContain("trunc");

    // Non-stream path: no retry either.
    let nsCalls = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            nsCalls += 1;
            return jsonResponse({
              id: "x",
              choices: [{ index: 0, finish_reason: "content_filter", message: { role: "assistant", content: "I cannot help with that.", tool_calls: filteredCalls } }],
            });
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    await res.text();
    expect(nsCalls).toBe(1);
  });

  it("D3a: an error-only recovery chunk is NOT recovery — the original terminal still ships", async () => {
    // streamRetryToolTurn used to count ANY non-empty `data:` payload as a forwarded
    // chunk. A 200 stream carrying only `data: {"error":...}` (Ollama / vLLM / OpenRouter
    // all signal mid-stream failure this way, so the status>=400 check never sees it)
    // therefore reported success, the held original terminal was suppressed, and the
    // client got an error object plus [DONE] with NO finish_reason at all — downstream,
    // an Anthropic `message_delta` with `stop_reason: null`.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseResponse([{ error: { message: "upstream overloaded", type: "server_error" } }]),
    );
    expect(calls).toBe(2);
    // Fail-open: the held original terminal is delivered, rewritten because the buffered
    // call was dropped as unrunnable.
    expect(finishReasonsOf(text)).toEqual(["length"]);
    expect(text).not.toContain("trunc");
  });

  it("D3b: a role-only recovery chunk is NOT recovery — the original terminal still ships", async () => {
    // Same root cause as D3a but far more likely: nearly every upstream opens with a
    // role-only delta, so a recovery stream that died right after its first chunk hit
    // `forwarded > 0` and suppressed the original terminal — replacing a broken turn
    // with an empty, terminal-less one.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseResponse([{ choices: [{ index: 0, delta: { role: "assistant" } }] }]),
    );
    expect(calls).toBe(2);
    expect(finishReasonsOf(text)).toEqual(["length"]);
  });

  it("D3c: recovery that forwards content then DIES gets a synthesised \"length\" terminal", async () => {
    // Real content reached the client, so failing open would splice two answers
    // together — the retry's turn stands. But it never sent a terminal, so one is
    // synthesised. The stream demonstrably broke mid-flight, which is what "length"
    // means: cut before the model was done.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseThenError([{ choices: [{ index: 0, delta: { content: "writing the file" } }] }]),
    );
    expect(calls).toBe(2);
    expect(text).toContain("writing the file");
    expect(finishReasonsOf(text)).toEqual(["length"]);
  });

  it("D3d: recovery that forwards content and ends CLEANLY with no terminal gets \"stop\"", async () => {
    // The ambiguous case, and the only one with no strictly honest answer: a clean EOF
    // could be a sloppy upstream that never emits finish_reason, or an intermediary that
    // closed tidily on a truncated turn. "stop" is chosen because nothing reported a cut;
    // "length" here would make every turn from such an upstream look truncated and drive
    // endless client-side auto-continuation.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseResponse([{ choices: [{ index: 0, delta: { content: "done, file written" } }] }]),
    );
    expect(calls).toBe(2);
    expect(text).toContain("done, file written");
    expect(finishReasonsOf(text)).toEqual(["stop"]);
  });

  it("D3e: recovery that forwards a COMPLETE tool call but no terminal gets \"tool_calls\"", async () => {
    // The client holds a name and JSON-object arguments — an executable call. That IS
    // why the turn ended; "length" would be the lie here, and would stop the client
    // from dispatching a call it can run.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseThenError([
        {
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  { index: 0, id: "r1", type: "function", function: { name: "write", arguments: '{"path":"a.html",' } },
                ],
              },
            },
          ],
        },
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"content":"hi"}' } }] } }] },
      ]),
    );
    expect(calls).toBe(2);
    expect(finishReasonsOf(text)).toEqual(["tool_calls"]);
  });

  it("D3f: recovery that forwards a TRUNCATED tool call and no terminal gets \"length\", not \"tool_calls\"", async () => {
    // The retry's own fragments do not assemble into valid JSON. Telling the client
    // "tool_calls" would order it to execute garbage; "length" maps to Anthropic
    // max_tokens and prompts a clean re-ask.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseThenError([{ choices: [brokenCallDelta(0, "r1")] }]),
    );
    expect(calls).toBe(2);
    expect(finishReasonsOf(text)).toEqual(["length"]);
  });

  it("D3g: a recovery that sends its OWN terminal gets exactly one, un-synthesised", async () => {
    // The control case: sawFinishReason must suppress both the synthesised terminal and
    // the held original one. Two terminals in one turn would be a worse bug than none.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseResponse([
        { choices: [{ index: 0, delta: { content: "no tool needed" } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
      ]),
    );
    expect(calls).toBe(2);
    expect(finishReasonsOf(text)).toEqual(["stop"]);
  });

  it("D4a: the NON-STREAM twin recovers a truncated call under finish_reason:\"tool_calls\"", async () => {
    // The stream path always caught this via `assembledCallsEmittable` (see the
    // "assembledCallsEmittable: a CLEAN tool_calls finish..." test). The non-stream path
    // has only detectIncompleteToolTurn, whose broken-args branch used to be gated on
    // finish_reason === "length" — so /v1 non-stream shipped an unparseable call
    // verbatim with `finish_reason:"tool_calls"`, no retry billed, no honesty fix.
    // The gate is now `hasCalls` under any finish_reason, so the two paths agree.
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
              id: "x",
              choices: [
                {
                  index: 0,
                  finish_reason: "tool_calls",
                  message: {
                    role: "assistant",
                    content: null,
                    tool_calls: [
                      { id: "c", type: "function", function: { name: "write", arguments: '{"path":"a","content":"trunc' } },
                    ],
                  },
                },
              ],
            });
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    await res.text();
    expect(calls).toBe(2); // a recovery retry must be attempted, as on the stream path
  });

  it("D4b: the NON-STREAM twin recovers a NAMELESS call", async () => {
    // toolCallArgsBroken inspects `arguments` only, never `function.name`. The stream
    // path's assembledCallsEmittable has always required a name; the non-stream path did
    // not, so a call with nothing to dispatch was delivered as runnable — and it also
    // makes the Anthropic non-stream converter's schema fail (`name: z.string()` is
    // required), which falls back to returning the raw OpenAI body on /v1/messages.
    // `toolCallNameMissing` now closes that gap for both paths.
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
              id: "x",
              choices: [
                {
                  index: 0,
                  finish_reason: "length",
                  message: {
                    role: "assistant",
                    content: null,
                    tool_calls: [{ id: "c", type: "function", function: { arguments: '{"path":"a"}' } }],
                  },
                },
              ],
            });
          },
        },
      ]),
    });
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", tools: TOOLS, messages: [{ role: "user", content: "x" }] }),
    );
    await res.text();
    expect(calls).toBe(2);
  });

  it("D5: markTerminalLengthCut parses a terminal EXACTLY as handleLine does (trimStart + payload trim)", async () => {
    // handleLine accepts `line.trimStart()` and then `payload.trim()`s before parsing;
    // markTerminalLengthCut used to do neither. Any line handleLine accepted but the
    // rewrite could not re-parse silently kept `finish_reason:"tool_calls"` for a call
    // the guard had just dropped. Two divergences, pinned separately:
    //
    //  (a) LEADING WHITESPACE before `data:`. Not spec-legal SSE (the field name is
    //      everything before the first colon, so " data" is not "data") — but handleLine
    //      deliberately accepts it, and a half-accepting guard is the bug.
    //  (b) A payload prefix that String.trim strips and JSON.parse REJECTS. JSON's
    //      whitespace set is only space/tab/CR/LF; trim's is the full Unicode WhiteSpace
    //      set plus U+FEFF. So `data:\u00A0{...}` parses in handleLine and threw here.
    //      This is why `trimStart()` alone is not parity.
    for (const frame of [
      (c: unknown) => ` data:${JSON.stringify(c)}\n\n`,
      (c: unknown) => `data:\u00A0${JSON.stringify(c)}\n\n`,
    ]) {
      const { text } = await runGuarded(() =>
        rawSse([
          frame({ choices: [brokenCallDelta(0, "c")] }),
          frame({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] }),
        ]),
      );
      expect(text).toContain('"finish_reason":"length"');
      expect(text).not.toContain('"finish_reason":"tool_calls"');
      expect(text).not.toContain("trunc");
    }
  });

  it("D3h: a recovery's OWN terminal is forwarded and is the turn's only one — no synthesis, no held original", async () => {
    // Reworked. The original D3h asserted that a terminal-ONLY retry counts as a
    // replacement turn; R2 showed that was pinning a workaround (it shipped an empty
    // tool turn), so that half now lives in R2 with the opposite expectation. What
    // survives is the real invariant: once the retry has sent something substantive,
    // its terminal goes out verbatim and nothing else follows it. `length` is chosen
    // deliberately — a synthesised terminal for a retry that produced prose and no
    // tool calls would say "stop", so dropping the retry's own terminal is visible.
    const { calls, text } = await runGuarded(brokenOriginal, () =>
      sseResponse([
        { choices: [{ index: 0, delta: { content: "partial answer" } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: "length" }] },
      ]),
    );
    expect(calls).toBe(2);
    expect(text).toContain("partial answer");
    expect(finishReasonsOf(text)).toEqual(["length"]);
  });

  it("D6: a failed recovery splices NOTHING but data: lines into the original turn", async () => {
    // forwardLine used to enqueue every line of the retry stream verbatim. The
    // cosmetic half was the blank separator trailing the swallowed `[DONE]` — a stray
    // empty SSE line ahead of the fail-open terminal. The non-cosmetic half is the
    // `event:` field: SSE `event:` is STICKY, naming the type of the NEXT dispatched
    // event, and on the fail-open path that next event is the guard's OWN terminal.
    // A retry that died after `event: error` therefore re-labelled the original
    // terminal as an error event; `id:` similarly rewrote the client's Last-Event-ID
    // from a stream that is not the one it is reading.
    const noisyRetry = () =>
      new Response(": keepalive\n\nevent: error\nid: 42\ndata: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    const { calls, text } = await runGuarded(
      () =>
        sseResponse([
          { choices: [brokenCallDelta(0, "c")] },
          { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
        ]),
      noisyRetry,
    );
    expect(calls).toBe(2);
    expect(text).not.toContain("event:"); // would re-type the guard's own terminal
    expect(text).not.toContain("id: 42"); // not this stream's event id
    expect(text).not.toContain("keepalive"); // the guard emits its own pings
    expect(text.startsWith("\n\n")).toBe(false); // no leaked separator
    expect(finishReasonsOf(text)).toEqual(["length"]); // fail-open, terminal rewritten
  });
});
