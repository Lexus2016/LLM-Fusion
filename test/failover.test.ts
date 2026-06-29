import { describe, it, expect } from "vitest";
import { z } from "zod";
import { failoverStrategy } from "../src/strategies/failover";
import { createApp } from "../src/server";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { createResilience } from "../src/concurrency";
import type { FailoverPolicy, Resilience } from "../src/concurrency";
import { jsonResponse, sseResponse, sseThenError, streamErrorImmediate } from "./helpers";
import type { ChatCompletionRequest, FetchFn, StrategyContext, UpstreamClient } from "../src/types";

const logger = createLogger({ level: "silent" });
const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  models: { ha: { strategy: "failover", chain: ["m1", "m2"] } },
});

const BodySchema = z.object({ model: z.string() }).passthrough();

/** A `/v1/chat/completions` mock that dispatches per requested model + call index. */
function chatMock(handler: (model: string, callIndex: number) => Response): {
  fetchFn: FetchFn;
  counts: Record<string, number>;
} {
  const counts: Record<string, number> = {};
  const fetchFn: FetchFn = async (input, init) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (!url.endsWith("/v1/chat/completions")) {
      return new Response(JSON.stringify({ error: `no route for ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }
    const parsed = BodySchema.safeParse(JSON.parse(String(init?.body)));
    const model = parsed.success ? parsed.data.model : "";
    const idx = counts[model] ?? 0;
    counts[model] = idx + 1;
    return handler(model, idx);
  };
  return { fetchFn, counts };
}

function makeResilience(
  policy?: Partial<FailoverPolicy>,
  breaker?: { failureThreshold?: number },
): { resilience: Resilience; sleeps: number[] } {
  const sleeps: number[] = [];
  const resilience = createResilience({
    maxConcurrency: 4,
    now: () => 1_000_000, // frozen clock: cooldown never elapses mid-test
    sleep: async (ms) => {
      sleeps.push(ms);
    },
    backoff: { baseMs: 10, factor: 2, maxMs: 1000, jitter: 0, rng: () => 0 },
    policy,
    failureThreshold: breaker?.failureThreshold,
  });
  return { resilience, sleeps };
}

function ctxWith(
  client: UpstreamClient,
  request: ChatCompletionRequest,
  resilience: Resilience,
  signal?: AbortSignal,
): StrategyContext {
  const capabilities = new CapabilityService({
    client,
    getOverrides: () => config.overrides,
    logger,
  });
  const entry = config.models["ha"];
  if (!entry) throw new Error("test config missing failover model 'ha'");
  return { request, config, client, capabilities, logger, modelConfig: entry, resilience, signal };
}

function clientWith(fetchFn: FetchFn): OllamaClient {
  return new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
}

async function readAll(res: Response): Promise<string> {
  return res.text();
}

describe("failover strategy", () => {
  it("advances to the next member on 5xx and returns its success", async () => {
    const { fetchFn, counts } = chatMock((model) =>
      model === "m1"
        ? jsonResponse({ error: "boom" }, 500)
        : jsonResponse({ id: "x", choices: [{ message: { content: "from-m2" } }] }),
    );
    const { resilience } = makeResilience({ maxServerRetries: 0 });
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", messages: [] }, resilience),
    );
    expect(res.status).toBe(200);
    const body = JSON.parse(await readAll(res));
    expect(body.choices[0].message.content).toBe("from-m2");
    expect(counts["m1"]).toBe(1);
    expect(counts["m2"]).toBe(1);
  });

  it("on 429 retries the SAME member with backoff, never advancing, and succeeds later", async () => {
    const { fetchFn, counts } = chatMock((model, idx) => {
      if (model === "m1") {
        return idx < 2
          ? jsonResponse({ error: "rate limited" }, 429)
          : jsonResponse({ id: "x", choices: [{ message: { content: "from-m1" } }] });
      }
      return jsonResponse({ id: "y", choices: [{ message: { content: "from-m2" } }] });
    });
    const { resilience, sleeps } = makeResilience();
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", messages: [] }, resilience),
    );
    expect(res.status).toBe(200);
    const body = JSON.parse(await readAll(res));
    expect(body.choices[0].message.content).toBe("from-m1");
    expect(counts["m1"]).toBe(3); // 429, 429, 200
    expect(counts["m2"] ?? 0).toBe(0); // 429 must NOT advance the chain
    expect(sleeps).toHaveLength(2); // backed off before each retry
    expect(sleeps[0]).toBeLessThan(sleeps[1] ?? 0); // increasing delays
  });

  it("surfaces a non-429 4xx immediately without advancing", async () => {
    const { fetchFn, counts } = chatMock((model) =>
      model === "m1"
        ? jsonResponse({ error: { message: "bad request" } }, 400)
        : jsonResponse({ id: "y", choices: [{ message: { content: "from-m2" } }] }),
    );
    const { resilience } = makeResilience();
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", messages: [] }, resilience),
    );
    expect(res.status).toBe(400);
    expect(counts["m1"]).toBe(1);
    expect(counts["m2"] ?? 0).toBe(0);
  });

  it("maps an all-members-failed chain to 502", async () => {
    const { fetchFn, counts } = chatMock(() => jsonResponse({ error: "boom" }, 500));
    const { resilience } = makeResilience({ maxServerRetries: 0 });
    await expect(
      failoverStrategy.execute(ctxWith(clientWith(fetchFn), { model: "ha", messages: [] }, resilience)),
    ).rejects.toMatchObject({ httpStatus: 502 });
    expect(counts["m1"]).toBe(1);
    expect(counts["m2"]).toBe(1);
  });

  it("skips circuit-open members; returns a healthy later member", async () => {
    const { fetchFn, counts } = chatMock((model) =>
      jsonResponse({ id: "x", choices: [{ message: { content: `from-${model}` } }] }),
    );
    const { resilience } = makeResilience(undefined, { failureThreshold: 2 });
    resilience.breaker.recordFailure("m1");
    resilience.breaker.recordFailure("m1"); // m1 now open
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", messages: [] }, resilience),
    );
    expect(res.status).toBe(200);
    const body = JSON.parse(await readAll(res));
    expect(body.choices[0].message.content).toBe("from-m2");
    expect(counts["m1"] ?? 0).toBe(0); // never called — breaker open
    expect(counts["m2"]).toBe(1);
  });

  it("returns 503 when every member is circuit-open", async () => {
    const { fetchFn, counts } = chatMock(() => jsonResponse({ ok: true }));
    const { resilience } = makeResilience(undefined, { failureThreshold: 1 });
    resilience.breaker.recordFailure("m1");
    resilience.breaker.recordFailure("m2");
    await expect(
      failoverStrategy.execute(ctxWith(clientWith(fetchFn), { model: "ha", messages: [] }, resilience)),
    ).rejects.toMatchObject({ httpStatus: 503 });
    expect(counts["m1"] ?? 0).toBe(0);
    expect(counts["m2"] ?? 0).toBe(0);
  });

  it("streaming: an error BEFORE the first chunk advances and streams the next member", async () => {
    const { fetchFn, counts } = chatMock((model) =>
      model === "m1"
        ? streamErrorImmediate()
        : sseResponse([
            { choices: [{ delta: { content: "x" } }] },
            { choices: [{ delta: { content: "y" } }] },
          ]),
    );
    const { resilience } = makeResilience({ maxServerRetries: 0 });
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", stream: true, messages: [] }, resilience),
    );
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await readAll(res);
    expect(text).toContain('"content":"x"');
    expect(text).toContain('"content":"y"');
    expect(text).toContain("[DONE]");
    expect(counts["m1"]).toBe(1);
    expect(counts["m2"]).toBe(1);
  });

  it("streaming: an error AFTER the first chunk surfaces as a stream error and does NOT switch members", async () => {
    const { fetchFn, counts } = chatMock((model) =>
      model === "m1"
        ? sseThenError([{ choices: [{ delta: { content: "a" } }] }])
        : sseResponse([{ choices: [{ delta: { content: "z" } }] }]),
    );
    const { resilience } = makeResilience();
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", stream: true, messages: [] }, resilience),
    );
    const body = res.body;
    if (!body) throw new Error("expected a streaming body");
    const reader = body.getReader();
    const decoder = new TextDecoder();

    const first = await reader.read();
    expect(first.done).toBe(false);
    expect(decoder.decode(first.value)).toContain('"content":"a"'); // first token forwarded

    // The next read must reject (mid-stream error surfaced), not silently re-route.
    await expect(reader.read()).rejects.toThrow();

    expect(counts["m1"]).toBe(1);
    expect(counts["m2"] ?? 0).toBe(0); // member NOT switched after first byte
  });

  // L-5: an SSE keep-alive comment (`:` line) or blank line is NOT a content
  // commitment. Failover may still advance until a real `data:` line arrives.
  function sseChunks(chunks: string[], endWithError?: string): Response {
    const encoder = new TextEncoder();
    let i = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        const chunk = chunks[i];
        if (chunk !== undefined) {
          controller.enqueue(encoder.encode(chunk));
          i += 1;
          return;
        }
        if (endWithError !== undefined) controller.error(new Error(endWithError));
        else controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
  }

  it("streaming: a leading keep-alive comment is not a commitment; a pre-content error advances", async () => {
    const { fetchFn, counts } = chatMock((model) =>
      model === "m1"
        ? sseChunks([": keep-alive\n\n"], "stalled before content") // comment, then error
        : sseResponse([{ choices: [{ delta: { content: "y" } }] }]),
    );
    const { resilience } = makeResilience({ maxServerRetries: 0 });
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", stream: true, messages: [] }, resilience),
    );
    const text = await readAll(res);
    expect(text).toContain('"content":"y"'); // served by m2
    expect(counts["m1"]).toBe(1);
    expect(counts["m2"]).toBe(1); // advanced past the keep-alive-only m1
  });

  it("streaming: keep-alive comment then real content commits to the SAME member (comment preserved)", async () => {
    const contentChunk = `data: ${JSON.stringify({ choices: [{ delta: { content: "x" } }] })}\n\n`;
    const { fetchFn, counts } = chatMock((model) =>
      model === "m1"
        ? sseChunks([": keep-alive\n\n", contentChunk, "data: [DONE]\n\n"])
        : sseResponse([{ choices: [{ delta: { content: "z" } }] }]),
    );
    const { resilience } = makeResilience({ maxServerRetries: 0 });
    const res = await failoverStrategy.execute(
      ctxWith(clientWith(fetchFn), { model: "ha", stream: true, messages: [] }, resilience),
    );
    const text = await readAll(res);
    expect(text).toContain(": keep-alive"); // the keep-alive bytes are preserved
    expect(text).toContain('"content":"x"'); // m1's real content
    expect(text).toContain("[DONE]");
    expect(counts["m1"]).toBe(1);
    expect(counts["m2"] ?? 0).toBe(0); // committed to m1, never advanced
  });
});

describe("failover wired through the server", () => {
  const appConfig = parseConfig({
    upstream: { base_url: "https://mock.test", api_key_env: "X" },
    models: { ha: { strategy: "failover", chain: ["m1", "m2"] } },
  });

  it("POST /v1/chat/completions advances past a 5xx member and returns 200", async () => {
    const { fetchFn } = chatMock((model) =>
      model === "m1"
        ? jsonResponse({ error: "boom" }, 500)
        : jsonResponse({ id: "x", choices: [{ message: { content: "served" } }] }),
    );
    const client = clientWith(fetchFn);
    const capabilities = new CapabilityService({
      client,
      getOverrides: () => appConfig.overrides,
      logger,
    });
    const { resilience } = makeResilience({ maxServerRetries: 0 });
    const app = createApp({
      getConfig: () => appConfig,
      client,
      capabilities,
      getAuthToken: () => undefined,
      logger,
      resilience,
    });
    const res = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "ha", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(200);
    const body = JSON.parse(await res.text());
    expect(body.choices[0].message.content).toBe("served");
  });

  it("streaming: a client disconnect during the pre-content peek does NOT trip the breaker", async () => {
    const controller = new AbortController();
    // A stream that errors on the first read, after we abort the client signal —
    // simulates a client disconnect surfacing as an AbortError during the peek.
    function abortingStream(): Response {
      const stream = new ReadableStream<Uint8Array>({
        start(c) {
          // Abort the client signal, then error the upstream read.
          controller.abort();
          c.error(new DOMException("aborted", "AbortError"));
        },
      });
      return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
    }
    const { fetchFn, counts } = chatMock(() => abortingStream());
    const { resilience } = makeResilience({ maxServerRetries: 0 }, { failureThreshold: 1 });

    await expect(
      failoverStrategy.execute(
        ctxWith(clientWith(fetchFn), { model: "ha", stream: true, messages: [] }, resilience, controller.signal),
      ),
    ).rejects.toThrow();

    // The member was attempted (peek ran) but the disconnect did NOT count as a failure.
    expect(counts["m1"]).toBe(1);
    expect(resilience.breaker.getState("m1")).toBe("closed");
  });
});
