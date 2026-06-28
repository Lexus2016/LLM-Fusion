import { describe, it, expect, beforeEach } from "vitest";
import type { UpstreamClient, ChatCompletionRequest, StrategyContext, FetchFn } from "../src/types";
import type { FusionModelConfig } from "../src/config";
import {
  aggregateVerdicts,
  buildBinevalUserPrompt,
  DEFAULT_DIMENSIONS,
  parseBinaryEvaluation,
  runBineval,
  type BinaryQuestion,
  type BinaryVerdict,
} from "../src/bineval";
import { createResilience } from "../src/concurrency";
import { realTimer } from "../src/timeout";
import { createLogger } from "../src/logging";
import { parseConfig } from "../src/config";
import { fusionStrategy } from "../src/strategies/fusion";
import { smartStrategy, __resetRouterCacheForTesting } from "../src/strategies/smart";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { jsonResponse, sseResponse } from "./helpers";

const logger = createLogger({ level: "silent" });

function makeRequest(): ChatCompletionRequest {
  return { model: "fusion-1", messages: [{ role: "user", content: "Summarize the key points." }] };
}

function makeCtx(client: UpstreamClient): StrategyContext {
  return {
    request: makeRequest(),
    config: {
      upstream: {
        base_url: "https://mock.test",
        api_key_env: "X",
        api_mode: "openai",
        max_concurrency: 4,
        request_timeout_s: 170,
      },
      server: { bind: "127.0.0.1", port: 8080 },
      defaults: {
        panel_member_timeout_s: 90,
        judge_timeout_s: 170,
        router_timeout_s: 30,
        min_panel_success: 1,
        promote_reasoning_to_content: false,
      },
      models: {},
      overrides: {},
    },
    client,
    capabilities: {
      discover: async () => ({ capability: { vision: false, tools: false, context: null }, source: "default" }),
      clear: () => {},
    },
    logger,
    modelConfig: {
      strategy: "fusion",
      panel: ["m"],
      judge: "j",
      synth: "s",
      tool_mode: "deliberate",
      fusion_planning_turn_only: false,
    } satisfies FusionModelConfig,
  };
}

function allYesVerdicts(questions = DEFAULT_DIMENSIONS): BinaryVerdict[] {
  return questions.map((q) => ({
    dimension: q.dimension,
    question: q.question,
    verdict: true,
    explanation: "yes",
  }));
}

function mockClient(verdicts: BinaryVerdict[]): UpstreamClient {
  return {
    chatCompletions: async () => ({
      kind: "json",
      status: 200,
      data: {
        choices: [{ message: { role: "assistant", content: JSON.stringify({ verdicts }) } }],
      },
      usage: { promptTokens: 50, completionTokens: 50, totalTokens: 100 },
    }),
    show: async () => ({}),
    chatNative: async () => ({
      kind: "json",
      status: 200,
      data: {},
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    }),
  };
}

describe("buildBinevalUserPrompt", () => {
  it("includes the request, output, and numbered questions", () => {
    const prompt = buildBinevalUserPrompt("req", "out", DEFAULT_DIMENSIONS);
    expect(prompt).toContain("req");
    expect(prompt).toContain("out");
    expect(prompt).toContain("[factual_consistency]");
    expect(prompt).toContain("1. [factual_consistency]");
    expect(prompt).toContain(`${DEFAULT_DIMENSIONS.length}. [clarity]`);
  });
});

describe("aggregateVerdicts", () => {
  it("returns perfect overall when every verdict is yes", () => {
    const result = aggregateVerdicts(allYesVerdicts());
    expect(result.overall).toBe(1);
    for (const score of Object.values(result.dimensions)) {
      expect(score).toBe(1);
    }
  });

  it("computes per-dimension averages when a dimension has multiple questions", () => {
    const verdicts: BinaryVerdict[] = [
      { dimension: "a", question: "a1", verdict: true },
      { dimension: "a", question: "a2", verdict: false },
      { dimension: "b", question: "b1", verdict: true },
    ];
    const result = aggregateVerdicts(verdicts);
    expect(result.dimensions["a"]).toBe(0.5);
    expect(result.dimensions["b"]).toBe(1);
    expect(result.overall).toBe(0.75);
  });

  it("handles an empty verdict list", () => {
    const result = aggregateVerdicts([]);
    expect(result.overall).toBe(0);
    expect(Object.keys(result.dimensions)).toHaveLength(0);
  });
});

describe("parseBinaryEvaluation", () => {
  it("parses a valid verdict JSON object", () => {
    const verdicts = allYesVerdicts();
    const content = JSON.stringify({ verdicts });
    const result = parseBinaryEvaluation(content, DEFAULT_DIMENSIONS);
    expect(result).not.toBeNull();
    expect(result?.overall).toBe(1);
  });

  it("returns null when verdict count does not match question count", () => {
    const content = JSON.stringify({ verdicts: allYesVerdicts().slice(1) });
    expect(parseBinaryEvaluation(content, DEFAULT_DIMENSIONS)).toBeNull();
  });

  it("accepts a lightly paraphrased question text (matches by dimension, not verbatim)", () => {
    const verdicts = allYesVerdicts();
    const first = verdicts[0];
    if (!first) throw new Error("expected at least one verdict");
    const modified: BinaryVerdict[] = [
      { ...first, question: first.question + " (rephrased)" },
      ...verdicts.slice(1),
    ];
    const content = JSON.stringify({ verdicts: modified });
    const result = parseBinaryEvaluation(content, DEFAULT_DIMENSIONS);
    expect(result).not.toBeNull();
    expect(result?.overall).toBe(1);
  });

  it("returns null when a verdict has an unknown dimension", () => {
    const verdicts = allYesVerdicts();
    const first = verdicts[0];
    if (!first) throw new Error("expected at least one verdict");
    const modified: BinaryVerdict[] = [
      { ...first, dimension: "not_a_real_dimension" },
      ...verdicts.slice(1),
    ];
    const content = JSON.stringify({ verdicts: modified });
    expect(parseBinaryEvaluation(content, DEFAULT_DIMENSIONS)).toBeNull();
  });

  it("returns null when a dimension appears twice", () => {
    const verdicts = allYesVerdicts();
    const first = verdicts[0];
    const second = verdicts[1];
    if (!first || !second) throw new Error("expected at least two verdicts");
    const modified: BinaryVerdict[] = [
      { ...first },
      { ...second, dimension: first.dimension },
      ...verdicts.slice(2),
    ];
    const content = JSON.stringify({ verdicts: modified });
    expect(parseBinaryEvaluation(content, DEFAULT_DIMENSIONS)).toBeNull();
  });

  it("accepts multiple questions per dimension (BinEval core use case)", () => {
    // Two questions share the "consistency" dimension — BinEval decomposes a
    // criterion into several atomic checks. The parser must accept this and
    // aggregate both verdicts under the same dimension.
    const questions: BinaryQuestion[] = [
      { dimension: "consistency", question: "Are named entities accurate?" },
      { dimension: "consistency", question: "Are numbers/quantities correct?" },
      { dimension: "clarity", question: "Is the prose readable?" },
    ];
    const verdicts: BinaryVerdict[] = [
      { dimension: "consistency", question: "Are named entities accurate?", verdict: true },
      { dimension: "consistency", question: "Are numbers/quantities correct?", verdict: false },
      { dimension: "clarity", question: "Is the prose readable?", verdict: true },
    ];
    const content = JSON.stringify({ verdicts });
    const result = parseBinaryEvaluation(content, questions);
    expect(result).not.toBeNull();
    expect(result?.dimensions["consistency"]).toBe(0.5);
    expect(result?.dimensions["clarity"]).toBe(1);
    expect(result?.overall).toBe(0.75);
  });

  it("returns null when a multi-question dimension is missing a verdict", () => {
    const questions: BinaryQuestion[] = [
      { dimension: "consistency", question: "Q1" },
      { dimension: "consistency", question: "Q2" },
      { dimension: "clarity", question: "Q3" },
    ];
    // Only 2 verdicts (both consistency) — clarity missing, count mismatches.
    const verdicts: BinaryVerdict[] = [
      { dimension: "consistency", question: "Q1", verdict: true },
      { dimension: "consistency", question: "Q2", verdict: true },
    ];
    const content = JSON.stringify({ verdicts });
    expect(parseBinaryEvaluation(content, questions)).toBeNull();
  });

  it("returns null for malformed JSON", () => {
    expect(parseBinaryEvaluation("not json", DEFAULT_DIMENSIONS)).toBeNull();
  });

  it("strips surrounding prose and code fences", () => {
    const verdicts = allYesVerdicts();
    const content = `Here is the JSON:\n\`\`\`json\n${JSON.stringify({ verdicts })}\n\`\`\`\nDone.`;
    const result = parseBinaryEvaluation(content, DEFAULT_DIMENSIONS);
    expect(result?.overall).toBe(1);
  });
});

describe("runBineval", () => {
  it("returns aggregated scores on a successful evaluator call", async () => {
    const client = mockClient(allYesVerdicts());
    const ctx = makeCtx(client);
    const resilience = createResilience({ maxConcurrency: 4 });
    const result = await runBineval(
      ctx,
      resilience,
      "eval-model",
      "request text",
      "output text",
      DEFAULT_DIMENSIONS,
      realTimer,
      1000,
    );
    expect(result).not.toBeNull();
    expect(result?.overall).toBe(1);
    expect(Object.keys(result?.dimensions ?? {})).toHaveLength(DEFAULT_DIMENSIONS.length);
  });

  it("returns null and records failure when the upstream returns a 5xx", async () => {
    const client: UpstreamClient = {
      chatCompletions: async () => ({
        kind: "json",
        status: 503,
        data: { error: "upstream down" },
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      }),
      show: async () => ({}),
      chatNative: async () => ({
        kind: "json",
        status: 200,
        data: {},
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      }),
    };
    const ctx = makeCtx(client);
    const resilience = createResilience({ maxConcurrency: 4, failureThreshold: 1 });
    const result = await runBineval(
      ctx,
      resilience,
      "eval-model",
      "request text",
      "output text",
      DEFAULT_DIMENSIONS,
      realTimer,
      1000,
    );
    expect(result).toBeNull();
    expect(resilience.breaker.canAttempt("eval-model")).toBe(false);
  });

  it("returns null without tripping the breaker on a 4xx", async () => {
    const client: UpstreamClient = {
      chatCompletions: async () => ({
        kind: "json",
        status: 400,
        data: { error: "bad request" },
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      }),
      show: async () => ({}),
      chatNative: async () => ({
        kind: "json",
        status: 200,
        data: {},
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      }),
    };
    const ctx = makeCtx(client);
    const resilience = createResilience({ maxConcurrency: 4 });
    const result = await runBineval(
      ctx,
      resilience,
      "eval-model",
      "request text",
      "output text",
      DEFAULT_DIMENSIONS,
      realTimer,
      1000,
    );
    expect(result).toBeNull();
    expect(resilience.breaker.canAttempt("eval-model")).toBe(true);
  });
});

// --- Integration: full fusion strategy with bineval wired in ---------------

const fusionBinevalConfig = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 4 },
  models: {
    "fusion-bineval": {
      strategy: "fusion",
      panel: ["m1", "m2", "m3"],
      judge: "j",
      synth: "s",
      bineval: { enabled: true, model: "eval" },
    },
    "fusion-bineval-custom": {
      strategy: "fusion",
      panel: ["m1", "m2"],
      judge: "j",
      synth: "s",
      bineval: {
        enabled: true,
        model: "eval",
        threshold: 0.5,
        dimensions: [
          { dimension: "code_correctness", question: "Is the code syntactically valid and correct?" },
          { dimension: "explanation_clarity", question: "Is the accompanying explanation clear?" },
        ],
      },
    },
    "fusion-bineval-planning": {
      strategy: "fusion",
      panel: ["m1", "m2", "m3"],
      judge: "j",
      synth: "s",
      fusion_planning_turn_only: true,
      bineval: { enabled: true, model: "eval" },
    },
  },
});

type ChatHandler = (body: Record<string, unknown>) => Response | Promise<Response>;

function makeUpstream(chat: ChatHandler): UpstreamClient {
  const fetchFn: FetchFn = async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (url.endsWith("/api/show")) return jsonResponse({ capabilities: ["completion"], model_info: {} });
    if (url.endsWith("/v1/chat/completions") || url.endsWith("/api/chat")) {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return chat(body);
    }
    return jsonResponse({ error: `no route for ${url}` }, 404);
  };
  return new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
}

function fusionCtx(client: UpstreamClient, request: ChatCompletionRequest, model: string): StrategyContext {
  const entry = fusionBinevalConfig.models[model];
  if (!entry) throw new Error(`test config missing '${model}'`);
  const capabilities = new CapabilityService({
    client,
    getOverrides: () => fusionBinevalConfig.overrides,
    logger,
  });
  return { request, config: fusionBinevalConfig, client, capabilities, logger, modelConfig: entry };
}

function verdictsFor(questions: BinaryQuestion[], allYes: boolean) {
  return questions.map((q) => ({ ...q, verdict: allYes, explanation: allYes ? "yes" : "no" }));
}

describe("fusion + bineval integration", () => {
  it("attaches bineval score headers on a non-streaming fusion response", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        return jsonResponse({ choices: [{ message: { content: "final synthesized answer" } }] });
      }
      if (model === "eval") {
        return jsonResponse({
          choices: [
            {
              message: {
                content: JSON.stringify({ verdicts: verdictsFor(DEFAULT_DIMENSIONS, true) }),
              },
            },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });

    const ctx = fusionCtx(client, { model: "fusion-bineval", messages: [{ role: "user", content: "hi" }] }, "fusion-bineval");
    const response = await fusionStrategy.execute(ctx);

    expect(response.status).toBe(200);
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBe("1.000");
    expect(response.headers.get("X-Fusion-Bineval-Dimensions")).not.toBeNull();
    expect(response.headers.has("X-Fusion-Bineval-Low-Score")).toBe(false);
    // Body preserved.
    const body = await response.json();
    expect(body.choices[0].message.content).toBe("final synthesized answer");
  });

  it("sets the low-score header when the overall score is below threshold", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        return jsonResponse({ choices: [{ message: { content: "shaky answer" } }] });
      }
      if (model === "eval") {
        // All verdicts false -> overall 0, below the default 0.7 threshold.
        return jsonResponse({
          choices: [
            { message: { content: JSON.stringify({ verdicts: verdictsFor(DEFAULT_DIMENSIONS, false) }) } },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });

    const ctx = fusionCtx(client, { model: "fusion-bineval", messages: [{ role: "user", content: "hi" }] }, "fusion-bineval");
    const response = await fusionStrategy.execute(ctx);

    expect(response.headers.get("X-Fusion-Bineval-Score")).toBe("0.000");
    expect(response.headers.get("X-Fusion-Bineval-Low-Score")).toBe("true");
  });

  it("uses custom configured dimensions when present", async () => {
    const fusionCfg = fusionBinevalConfig.models["fusion-bineval-custom"];
    if (!fusionCfg || fusionCfg.strategy !== "fusion" || !fusionCfg.bineval || !fusionCfg.bineval.dimensions) {
      throw new Error("missing fusion-bineval-custom bineval config");
    }
    const customDims = fusionCfg.bineval.dimensions;
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        return jsonResponse({ choices: [{ message: { content: "code answer" } }] });
      }
      if (model === "eval") {
        // Verify the evaluator received the CUSTOM questions (2, not the default 5).
        return jsonResponse({
          choices: [
            { message: { content: JSON.stringify({ verdicts: verdictsFor(customDims, true) }) } },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });

    const ctx = fusionCtx(
      client,
      { model: "fusion-bineval-custom", messages: [{ role: "user", content: "write code" }] },
      "fusion-bineval-custom",
    );
    const response = await fusionStrategy.execute(ctx);

    const dimsHeader = response.headers.get("X-Fusion-Bineval-Dimensions");
    expect(dimsHeader).not.toBeNull();
    const dims = JSON.parse(dimsHeader ?? "{}");
    expect(Object.keys(dims)).toEqual(expect.arrayContaining(["code_correctness", "explanation_clarity"]));
    expect(Object.keys(dims)).toHaveLength(2);
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBe("1.000");
    // threshold 0.5, score 1.0 -> no low-score header
    expect(response.headers.has("X-Fusion-Bineval-Low-Score")).toBe(false);
  });

  it("does NOT run bineval on streaming responses", async () => {
    let evalCalled = false;
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        return sseResponse([{ choices: [{ delta: { content: "streamed" } }] }]);
      }
      if (model === "eval") {
        evalCalled = true;
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ verdicts: verdictsFor(DEFAULT_DIMENSIONS, true) }) } }],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });

    const ctx = fusionCtx(
      client,
      { model: "fusion-bineval", stream: true, messages: [{ role: "user", content: "hi" }] },
      "fusion-bineval",
    );
    const response = await fusionStrategy.execute(ctx);

    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBeNull();
    // bineval was configured but cannot run on a streaming response — surface WHY,
    // so a client always streaming never mistakes "no score" for "not configured".
    expect(response.headers.get("X-Fusion-Bineval-Skipped")).toBe("streaming");
    expect(evalCalled).toBe(false);
  });

  it("marks X-Fusion-Bineval-Skipped=synth_error when the synth returns a 4xx/5xx", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        return jsonResponse({ error: "overloaded" }, 500);
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });
    const ctx = fusionCtx(
      client,
      { model: "fusion-bineval", messages: [{ role: "user", content: "hi" }] },
      "fusion-bineval",
    );
    const response = await fusionStrategy.execute(ctx);
    expect(response.status).toBe(500);
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBeNull();
    expect(response.headers.get("X-Fusion-Bineval-Skipped")).toBe("synth_error");
  });

  it("marks X-Fusion-Bineval-Skipped=eval_failed when the evaluator model errors", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        return jsonResponse({ choices: [{ message: { content: "final answer" } }] });
      }
      if (model === "eval") {
        // Evaluator upstream fails — runBineval returns null, no score header.
        return jsonResponse({ error: "bad" }, 500);
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });
    const ctx = fusionCtx(
      client,
      { model: "fusion-bineval", messages: [{ role: "user", content: "hi" }] },
      "fusion-bineval",
    );
    const response = await fusionStrategy.execute(ctx);
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBeNull();
    expect(response.headers.get("X-Fusion-Bineval-Skipped")).toBe("eval_failed");
  });

  it("marks X-Fusion-Bineval-Skipped=empty_output when the synth returns tool_calls only", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "s") {
        // 200 but no text content — a tool-only response. extractAnswer -> "" -> nothing to score.
        return jsonResponse({
          choices: [{ message: { content: "", tool_calls: [{ id: "t1", type: "function", function: { name: "run", arguments: "{}" } }] } }],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });
    const ctx = fusionCtx(
      client,
      { model: "fusion-bineval", messages: [{ role: "user", content: "hi" }] },
      "fusion-bineval",
    );
    const response = await fusionStrategy.execute(ctx);
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBeNull();
    expect(response.headers.get("X-Fusion-Bineval-Skipped")).toBe("empty_output");
  });

  it("marks X-Fusion-Bineval-Skipped=synth_only on a planning_turn_only mid-loop continuation", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      if (model === "s") {
        return jsonResponse({ choices: [{ message: { content: "next step" } }] });
      }
      // eval should NOT be called on the synth-only path.
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });
    const ctx = fusionCtx(
      client,
      {
        model: "fusion-bineval-planning",
        messages: [
          { role: "user", content: "do the task" },
          { role: "assistant", content: "calling tool", tool_calls: [{ id: "t1", type: "function", function: { name: "run", arguments: "{}" } }] },
          { role: "tool", tool_call_id: "t1", content: "tool output" },
        ],
      },
      "fusion-bineval-planning",
    );
    const response = await fusionStrategy.execute(ctx);
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBeNull();
    expect(response.headers.get("X-Fusion-Bineval-Skipped")).toBe("synth_only");
  });
});

// --- Regression: smart inline fusion block must propagate bineval ------------

const smartBinevalConfig = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 4 },
  models: {
    "smart-inline-bineval": {
      strategy: "smart",
      router: "rt",
      default: "simple",
      simple: { target: "deepseek" },
      // INLINE fusion block WITH bineval — resolveFusion must propagate it.
      fusion: {
        panel: ["p1", "p2", "p3"],
        judge: "jdg",
        synth: "syn",
        bineval: { enabled: true, model: "eval" },
      },
    },
  },
});

function smartCtx(client: UpstreamClient, request: ChatCompletionRequest, model: string): StrategyContext {
  const entry = smartBinevalConfig.models[model];
  if (!entry) throw new Error(`test config missing '${model}'`);
  const capabilities = new CapabilityService({
    client,
    getOverrides: () => smartBinevalConfig.overrides,
    logger,
  });
  return { request, config: smartBinevalConfig, client, capabilities, logger, modelConfig: entry };
}

describe("smart inline fusion block propagates bineval", () => {
  beforeEach(() => {
    __resetRouterCacheForTesting();
  });

  it("runs bineval when the smart router routes to an inline fusion block with bineval enabled", async () => {
    const client = makeUpstream((body) => {
      const model = String(body.model ?? "");
      // Router classifier -> route=fusion.
      if (model === "rt") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ route: "fusion" }) } }],
        });
      }
      if (model === "jdg") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify({ consensus: "ok", disagreements: [] }) } }],
        });
      }
      if (model === "syn") {
        return jsonResponse({ choices: [{ message: { content: "synthesized" } }] });
      }
      if (model === "eval") {
        return jsonResponse({
          choices: [
            { message: { content: JSON.stringify({ verdicts: verdictsFor(DEFAULT_DIMENSIONS, true) }) } },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${model}` } }] });
    });

    const ctx = smartCtx(
      client,
      { model: "smart-inline-bineval", messages: [{ role: "user", content: "hi" }] },
      "smart-inline-bineval",
    );
    const response = await smartStrategy.execute(ctx);

    expect(response.status).toBe(200);
    // If resolveFusion dropped bineval, these headers would be null.
    expect(response.headers.get("X-Fusion-Bineval-Score")).toBe("1.000");
    expect(response.headers.get("X-Fusion-Bineval-Dimensions")).not.toBeNull();
  });
});
