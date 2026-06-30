import { describe, it, expect } from "vitest";
import { OllamaClient } from "../src/upstream/ollama";
import { singleStrategy } from "../src/strategies/single";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { CapabilityService } from "../src/capabilities";
import { mockFetch, jsonResponse, sseResponse } from "./helpers";
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
