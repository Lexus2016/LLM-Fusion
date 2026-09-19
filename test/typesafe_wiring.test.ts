import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { z } from "zod";
import { singleStrategy } from "../src/strategies/single";
import { smartStrategy, __resetRouterCacheForTesting } from "../src/strategies/smart";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { mockFetch, jsonResponse } from "./helpers";
import type { ChatCompletionRequest, StrategyContext, UpstreamClient } from "../src/types";

const logger = createLogger({ level: "silent" });
const TS_URL = "https://api.typesafe.ai/v1/systemone";

/** Both features on. Each test still controls the key via the environment. */
const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  typesafe: {
    enabled: true,
    timeout_s: 5,
    tool_turn_guard: { enabled: true, threshold: 0.75 },
    router: { enabled: true, confidence_min: 0.5, stuck_threshold: 0.7 },
  },
  models: {
    "fast-glm": { strategy: "single", target: "glm-5.2" },
    "smart-1": {
      strategy: "smart",
      router: "rt",
      default: "simple",
      simple: { target: "deepseek" },
      fusion: { panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
    },
  },
});

/** Everything off — the shipped default. Proves no third party is ever reached. */
const offConfig = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X" },
  models: { "fast-glm": { strategy: "single", target: "glm-5.2" } },
});

function ctxWith(
  client: UpstreamClient,
  request: ChatCompletionRequest,
  model: string,
  cfg = config,
): StrategyContext {
  const capabilities = new CapabilityService({ client, getOverrides: () => cfg.overrides, logger });
  const entry = cfg.models[model];
  if (!entry) throw new Error(`test config missing '${model}'`);
  return { request, config: cfg, client, capabilities, logger, modelConfig: entry };
}

/** One TypeSafe noul answer per id. */
function nouls(values: Record<string, number>): Response {
  return jsonResponse({
    model: "jev-1.13.0",
    answers: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, { type: "noul", noul: v }])),
    usage: { input_tokens: 100, output_tokens: 0 },
  });
}

function routeAnswer(choice: string, confidence: number, stuck: number): Response {
  return jsonResponse({
    model: "jev-1.13.0",
    answers: {
      route: {
        type: "choice",
        choice,
        probabilities: { simple: choice === "simple" ? 0.9 : 0.1, fusion: choice === "fusion" ? 0.9 : 0.1 },
        confidence,
      },
      stuck: { type: "noul", noul: stuck },
    },
    usage: { input_tokens: 100, output_tokens: 0 },
  });
}

/** Counts TypeSafe calls so "never called" is provable, not assumed. */
function stubTypeSafe(respond: (init?: RequestInit) => Response): { calls: () => number } {
  let calls = 0;
  vi.stubGlobal(
    "fetch",
    mockFetch([
      {
        match: (url) => url === TS_URL,
        respond: (_url, init) => {
          calls += 1;
          return respond(init);
        },
      },
    ]),
  );
  return { calls: () => calls };
}

// ---------------------------------------------------------------------------
// Tool-turn guard — the semantic backstop for the EN/UA/RU phrase list
// ---------------------------------------------------------------------------

const TOOLS = [{ type: "function", function: { name: "write_file", parameters: { type: "object" } } }];

/** A turn that narrates its next action in a language the phrase list has never seen. */
const JAPANESE_NARRATION = "これからファイルを作成します。";

function narrateThenRecover(): {
  client: UpstreamClient;
  bodies: () => unknown[];
} {
  const bodies: unknown[] = [];
  let call = 0;
  const client = new OllamaClient({
    baseUrl: "https://mock.test",
    apiKey: "k",
    fetchFn: mockFetch([
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: (_u, init) => {
          bodies.push(JSON.parse(String(init?.body)));
          call += 1;
          // First call narrates; the recovery retry emits the tool call.
          return call === 1
            ? jsonResponse({ choices: [{ finish_reason: "stop", message: { content: JAPANESE_NARRATION } }] })
            : jsonResponse({
                choices: [
                  {
                    finish_reason: "tool_calls",
                    message: {
                      content: "",
                      tool_calls: [{ id: "c1", type: "function", function: { name: "write_file", arguments: "{}" } }],
                    },
                  },
                ],
              });
        },
      },
      { match: (u) => u.endsWith("/api/show"), respond: () => jsonResponse({ capabilities: ["completion"] }) },
    ]),
  });
  return { client, bodies: () => bodies };
}

describe("typesafe wiring — tool-turn guard", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("recovers a narrate-and-stop written in a language the phrase list does not cover", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => nouls({ narrates_next_action: 0.93 }));
    const { client, bodies } = narrateThenRecover();

    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", messages: [{ role: "user", content: "make it" }], tools: TOOLS }, "fast-glm"),
    );

    expect(res.status).toBe(200);
    expect(ts.calls()).toBe(1);
    const body = z.object({ choices: z.array(z.unknown()) }).parse(JSON.parse(await res.text()));
    const parsed = z
      .object({ choices: z.array(z.object({ message: z.object({ tool_calls: z.unknown().optional() }) })) })
      .parse(body);
    // The recovery ran and its tool call is what the client receives.
    expect(parsed.choices[0]?.message.tool_calls).toBeDefined();
    expect(bodies()).toHaveLength(2);
  });

  it("leaves the turn alone when the judgment lands below the threshold", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => nouls({ narrates_next_action: 0.2 }));
    const { client, bodies } = narrateThenRecover();

    await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", messages: [{ role: "user", content: "make it" }], tools: TOOLS }, "fast-glm"),
    );

    expect(ts.calls()).toBe(1);
    expect(bodies()).toHaveLength(1); // asked, answered "no", no retry
  });

  it("never asks when the phrase list already caught the turn", async () => {
    // The cheap path must win outright: an English narration is free to detect,
    // and paying for a judgment to confirm it would be the whole point missed.
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => nouls({ narrates_next_action: 0.99 }));
    let call = 0;
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([
        {
          match: (u) => u.endsWith("/v1/chat/completions"),
          respond: () => {
            call += 1;
            return call === 1
              ? jsonResponse({ choices: [{ finish_reason: "stop", message: { content: "Let me write the file" } }] })
              : jsonResponse({
                  choices: [
                    {
                      finish_reason: "tool_calls",
                      message: {
                        tool_calls: [{ id: "c1", type: "function", function: { name: "write_file", arguments: "{}" } }],
                      },
                    },
                  ],
                });
          },
        },
        { match: (u) => u.endsWith("/api/show"), respond: () => jsonResponse({ capabilities: ["completion"] }) },
      ]),
    });

    await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", messages: [{ role: "user", content: "go" }], tools: TOOLS }, "fast-glm"),
    );
    expect(ts.calls()).toBe(0);
    expect(call).toBe(2); // recovered, without consulting TypeSafe
  });

  it("never asks when a tool call was emitted", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => nouls({ narrates_next_action: 0.99 }));
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
                  finish_reason: "tool_calls",
                  message: {
                    content: "I will write the file",
                    tool_calls: [{ id: "c1", type: "function", function: { name: "write_file", arguments: "{}" } }],
                  },
                },
              ],
            }),
        },
        { match: (u) => u.endsWith("/api/show"), respond: () => jsonResponse({ capabilities: ["completion"] }) },
      ]),
    });
    await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", messages: [{ role: "user", content: "go" }], tools: TOOLS }, "fast-glm"),
    );
    expect(ts.calls()).toBe(0);
  });

  it("makes no third-party call at all on a default install", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key"); // key present, config off — still nothing
    const ts = stubTypeSafe(() => nouls({ narrates_next_action: 0.99 }));
    const { client } = narrateThenRecover();
    await singleStrategy.execute(
      ctxWith(
        client,
        { model: "fast-glm", messages: [{ role: "user", content: "go" }], tools: TOOLS },
        "fast-glm",
        offConfig,
      ),
    );
    expect(ts.calls()).toBe(0);
  });

  it("falls back to the phrase list when the judge is unreachable", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => jsonResponse({ error: "rate limited" }, 429));
    const { client, bodies } = narrateThenRecover();
    const res = await singleStrategy.execute(
      ctxWith(client, { model: "fast-glm", messages: [{ role: "user", content: "go" }], tools: TOOLS }, "fast-glm"),
    );
    expect(res.status).toBe(200);
    expect(ts.calls()).toBe(1);
    expect(bodies()).toHaveLength(1); // no retry: exactly the pre-feature behaviour
  });
});

// ---------------------------------------------------------------------------
// Smart router — a typed choice instead of a prose prompt and a JSON parse
// ---------------------------------------------------------------------------

function smartUpstream(): { client: UpstreamClient; models: () => string[] } {
  const models: string[] = [];
  const client = new OllamaClient({
    baseUrl: "https://mock.test",
    apiKey: "k",
    fetchFn: mockFetch([
      {
        match: (u) => u.endsWith("/v1/chat/completions"),
        respond: (_u, init) => {
          const body = z.object({ model: z.string() }).parse(JSON.parse(String(init?.body)));
          models.push(body.model);
          if (body.model === "rt") {
            return jsonResponse({
              choices: [{ message: { content: JSON.stringify({ route: "simple", reason: "llm router ran" }) } }],
            });
          }
          return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
        },
      },
      { match: (u) => u.endsWith("/api/show"), respond: () => jsonResponse({ capabilities: ["completion"] }) },
    ]),
  });
  return { client, models: () => models };
}

const REQ: ChatCompletionRequest = { model: "smart-1", messages: [{ role: "user", content: "design a schema" }] };

describe("typesafe wiring — smart router", () => {
  beforeEach(() => __resetRouterCacheForTesting());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("routes on the typed choice and never calls the LLM router", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => routeAnswer("fusion", 0.88, 0.02));
    const { client, models } = smartUpstream();

    const res = await smartStrategy.execute(ctxWith(client, REQ, "smart-1"));
    expect(res.status).toBe(200);
    expect(ts.calls()).toBe(1);
    expect(models()).not.toContain("rt"); // the prose router was not consulted at all
    expect(models()).toContain("syn"); // fusion ran
  });

  it("uses the model's default route when confidence is below the floor", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    stubTypeSafe(() => routeAnswer("fusion", 0.31, 0.02));
    const { client, models } = smartUpstream();

    await smartStrategy.execute(ctxWith(client, REQ, "smart-1"));
    expect(models()).not.toContain("rt");
    expect(models()).toContain("deepseek"); // default: simple
    expect(models()).not.toContain("syn");
  });

  it("escalates a stuck loop to fusion even when the choice says simple", async () => {
    // The regex escalation it replaces has the same precedence: recovering from a
    // repeated failure is the step deliberation exists for.
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    stubTypeSafe(() => routeAnswer("simple", 0.95, 0.91));
    const { client, models } = smartUpstream();

    await smartStrategy.execute(ctxWith(client, REQ, "smart-1"));
    expect(models()).toContain("syn");
    expect(models()).not.toContain("rt");
  });

  it("falls back to the LLM router when TypeSafe fails", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    const ts = stubTypeSafe(() => jsonResponse({ error: "down" }, 529));
    const { client, models } = smartUpstream();

    const res = await smartStrategy.execute(ctxWith(client, REQ, "smart-1"));
    expect(res.status).toBe(200);
    expect(ts.calls()).toBe(1);
    expect(models()).toContain("rt"); // the previous path, untouched
  });

  it("falls back to the LLM router when the key is missing, however the config reads", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "");
    const ts = stubTypeSafe(() => routeAnswer("fusion", 0.99, 0.0));
    const { client, models } = smartUpstream();

    await smartStrategy.execute(ctxWith(client, REQ, "smart-1"));
    expect(ts.calls()).toBe(0);
    expect(models()).toContain("rt");
  });

  it("rejects an option outside the set it was given", async () => {
    // A choice the caller cannot switch on is not a weak answer, it is no answer.
    vi.stubEnv("TYPESAFE_API_KEY", "ts-test-key");
    stubTypeSafe(() =>
      jsonResponse({
        answers: {
          route: { type: "choice", choice: "deliberate", probabilities: { deliberate: 1 }, confidence: 0.99 },
          stuck: { type: "noul", noul: 0 },
        },
      }),
    );
    const { client, models } = smartUpstream();

    await smartStrategy.execute(ctxWith(client, REQ, "smart-1"));
    expect(models()).toContain("rt"); // fell through to the LLM router
  });
});
