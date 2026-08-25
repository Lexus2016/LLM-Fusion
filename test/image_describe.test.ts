import { describe, it, expect } from "vitest";
import type {
  ChatCompletionRequest,
  ChatCompletionResult,
  StrategyContext,
  UpstreamClient,
} from "../src/types";
import type { FusionModelConfig, ImageDescribeConfig } from "../src/config";
import { parseConfig } from "../src/config";
import { collectImageLocations, describeRequestImages } from "../src/image_describe";
import { createResilience } from "../src/concurrency";
import { realTimer } from "../src/timeout";
import { createLogger } from "../src/logging";

const logger = createLogger({ level: "silent" });

const resilience = createResilience({ maxConcurrency: 4 });

function makeCtx(
  request: ChatCompletionRequest,
  client: UpstreamClient,
): StrategyContext {
  return {
    request,
    config: {
      upstream: {
        base_url: "https://mock.test",
        api_key_env: "X",
        api_mode: "openai",
        max_concurrency: 4,
        request_timeout_s: 170,
        connector_cooldown_s: 60,
        connector_down_recheck_s: 900,
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
      discover: async () => ({
        capability: { vision: false, tools: false, context: null },
        source: "default",
      }),
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
    usage: undefined,
  };
}

function imageRequest(urls: string[] = ["data:image/png;base64,AAAA"]): ChatCompletionRequest {
  return {
    model: "fusion-1",
    messages: [
      { role: "user", content: "What is in this picture?" },
      {
        role: "user",
        content: urls.map((url) => ({ type: "image_url", image_url: { url } })),
      },
    ],
  };
}

const cfg = (over: Partial<ImageDescribeConfig> = {}): ImageDescribeConfig => ({
  enabled: true,
  model: "vision-m",
  max_chars: 12000,
  timeout_s: 60,
  ...over,
});

interface CapturedCall {
  body: Record<string, unknown>;
}

/** Mock client whose chatCompletions returns `content` and records bodies. */
function describerClient(
  content: string | null,
  opts: { status?: number; calls?: CapturedCall[] } = {},
): UpstreamClient {
  return {
    chatCompletions: async (body) => {
      opts.calls?.push({ body });
      const result: ChatCompletionResult = {
        kind: "json",
        status: opts.status ?? 200,
        data: { choices: [{ message: { role: "assistant", content } }] },
        usage: { promptTokens: 10, completionTokens: 10, totalTokens: 20 },
      };
      if (result.status !== (opts.status ?? 200)) result.status = opts.status ?? 200;
      return result;
    },
    show: async () => ({}),
    chatNative: async () => ({
      kind: "json",
      status: 200,
      data: {},
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    }),
  };
}

describe("collectImageLocations", () => {
  it("finds image parts across messages in order", () => {
    const req = imageRequest(["data:a", "data:b"]);
    const locs = collectImageLocations(req);
    expect(locs).toHaveLength(2);
    expect(locs[0]).toMatchObject({ mi: 1, pi: 0, url: "data:a" });
    expect(locs[1]).toMatchObject({ mi: 1, pi: 1, url: "data:b" });
  });

  it("returns empty for a text-only request", () => {
    expect(collectImageLocations({ model: "m", messages: [{ role: "user", content: "hi" }] })).toEqual([]);
  });
});

describe("describeRequestImages", () => {
  it("describes multiple images CONCURRENTLY, not one round-trip at a time", async () => {
    // Sequential describes cost N * timeout_s in the worst case; a 4-image paste
    // would stall the whole fusion for minutes before the panel even starts.
    // Gate every describer call on a barrier that only opens once ALL of them
    // have arrived: this deadlocks (and the test times out) on a sequential loop.
    const urls = ["data:a", "data:b", "data:c"];
    let arrived = 0;
    let openBarrier: () => void = () => {};
    const barrier = new Promise<void>((resolve) => {
      openBarrier = resolve;
    });
    const client: UpstreamClient = {
      chatCompletions: async () => {
        arrived += 1;
        if (arrived === urls.length) openBarrier();
        await barrier;
        return {
          kind: "json",
          status: 200,
          data: { choices: [{ message: { role: "assistant", content: "described" } }] },
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        } satisfies ChatCompletionResult;
      },
      show: async () => ({}),
      chatNative: async () => ({ kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } }),
    };
    const ctx = makeCtx(imageRequest(urls), client);
    const out = await describeRequestImages(ctx, resilience, cfg(), realTimer);
    expect(out).not.toBeNull();
    expect(arrived).toBe(urls.length);
  });

  it("keeps [IMAGE n] numbering aligned with source order under concurrency", async () => {
    // Promise.all resolves in input order, but the calls COMPLETE out of order.
    // Make the first image the slowest so a naive "append as they finish" would
    // mislabel the blocks.
    const urls = ["data:a", "data:b", "data:c"];
    const delayFor: Record<string, number> = { "data:a": 20, "data:b": 5, "data:c": 0 };
    const client: UpstreamClient = {
      chatCompletions: async (body) => {
        const messages = (body as { messages?: unknown[] }).messages ?? [];
        const parts = (messages[0] as { content?: unknown[] } | undefined)?.content ?? [];
        const imagePart = parts.find(
          (p): p is { image_url: { url: string } } =>
            typeof p === "object" && p !== null && (p as { type?: string }).type === "image_url",
        );
        const url = imagePart?.image_url.url ?? "";
        await new Promise((r) => setTimeout(r, delayFor[url] ?? 0));
        return {
          kind: "json",
          status: 200,
          data: { choices: [{ message: { role: "assistant", content: `desc-of-${url}` } }] },
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        } satisfies ChatCompletionResult;
      },
      show: async () => ({}),
      chatNative: async () => ({ kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } }),
    };
    const ctx = makeCtx(imageRequest(urls), client);
    const out = await describeRequestImages(ctx, resilience, cfg(), realTimer);
    expect(out).not.toBeNull();
    const parts = out!.messages![1]!.content as { type: string; text: string }[];
    expect(parts[0]!.text).toBe("[IMAGE 1]\ndesc-of-data:a");
    expect(parts[1]!.text).toBe("[IMAGE 2]\ndesc-of-data:b");
    expect(parts[2]!.text).toBe("[IMAGE 3]\ndesc-of-data:c");
  });

  it("returns null without touching the client when the request has no images", async () => {
    let calls = 0;
    const client: UpstreamClient = {
      chatCompletions: async () => {
        calls += 1;
        throw new Error("must not be called");
      },
      show: async () => ({}),
      chatNative: async () => ({ kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } }),
    };
    const ctx = makeCtx({ model: "fusion-1", messages: [{ role: "user", content: "plain" }] }, client);
    await expect(describeRequestImages(ctx, resilience, cfg(), realTimer)).resolves.toBeNull();
    expect(calls).toBe(0);
  });

  it("replaces every image with a numbered text block and leaves the ORIGINAL request untouched", async () => {
    const calls: CapturedCall[] = [];
    const req = imageRequest(["data:one", "data:two"]);
    const ctx = makeCtx(req, describerClient("A red square. Text says: HELLO.", { calls }));
    const out = await describeRequestImages(ctx, resilience, cfg(), realTimer);
    expect(out).not.toBeNull();
    // Original untouched.
    const origSecond = ctx.request.messages?.[1];
    expect(origSecond && Array.isArray(origSecond.content) ? origSecond.content[0] : null).toMatchObject({
      type: "image_url",
    });
    // New request text-only, numbered blocks, focus hint included upstream.
    const second = out?.messages?.[1];
    expect(second && Array.isArray(second.content) ? second.content : null).toEqual([
      { type: "text", text: "[IMAGE 1]\nA red square. Text says: HELLO." },
      { type: "text", text: "[IMAGE 2]\nA red square. Text says: HELLO." },
    ]);
    expect(collectImageLocations(out as ChatCompletionRequest)).toEqual([]);
    // Describer received the image + a focus hint from the latest user turn.
    const sent = calls[0]?.body.messages as Array<{ content: Array<{ type: string; text?: string }> }>;
    expect(JSON.stringify(sent)).toContain("data:one");
    expect(JSON.stringify(sent)).toContain("What is in this picture?");
  });

  it("returns null on a describer error (all-or-nothing fallback)", async () => {
    const failing: UpstreamClient = {
      chatCompletions: async () => {
        throw new Error("boom");
      },
      show: async () => ({}),
      chatNative: async () => ({ kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } }),
    };
    const ctx = makeCtx(imageRequest(), failing);
    await expect(describeRequestImages(ctx, resilience, cfg(), realTimer)).resolves.toBeNull();
  });

  it("returns null on a non-OK describer status", async () => {
    const calls: CapturedCall[] = [];
    const ctx = makeCtx(imageRequest(), describerClient(null, { status: 500, calls }));
    await expect(describeRequestImages(ctx, resilience, cfg(), realTimer)).resolves.toBeNull();
    expect(calls).toHaveLength(1);
  });

  it("returns null when the description is empty", async () => {
    const ctx = makeCtx(imageRequest(), describerClient(""));
    await expect(describeRequestImages(ctx, resilience, cfg(), realTimer)).resolves.toBeNull();
  });

  it("caps long descriptions at max_chars with a truncation marker", async () => {
    const long = "x".repeat(50);
    const ctx = makeCtx(imageRequest(), describerClient(long));
    const out = await describeRequestImages(ctx, resilience, cfg({ max_chars: 10 }), realTimer);
    const second = out?.messages?.[1];
    const part = second && Array.isArray(second.content) ? second.content[0] : null;
    expect(part).toMatchObject({ type: "text" });
    const text = (part as { text?: string }).text ?? "";
    expect(text.startsWith("[IMAGE 1]\nxxxxxxxxxx")).toBe(true);
    expect(text).toContain("[description truncated]");
  });

  it("BACK-COMPAT: a config WITHOUT image_describe parses unchanged and the field is undefined", () => {
    const parsed = parseConfig({
      upstream: {
        base_url: "https://ollama.com",
        api_key_env: "OLLAMA_API_KEY",
        api_mode: "openai",
        max_concurrency: 8,
        request_timeout_s: 180,
        connector_cooldown_s: 60,
        connector_down_recheck_s: 900,
      },
      server: { bind: "127.0.0.1", port: 8081 },
      defaults: {
        panel_member_timeout_s: 90,
        judge_timeout_s: 120,
        min_panel_success: 2,
      },
      models: {
        "fusion-coder": {
          strategy: "fusion",
          panel: ["kimi-k2.7-code", "deepseek-v4-pro:0813-cloud"],
          judge: "glm-5.2",
          synth: "deepseek-v4-flash:0731-cloud",
          tool_mode: "deliberate",
          fusion_planning_turn_only: true,
        },
      },
    });
    const coder = parsed.models["fusion-coder"];
    expect(coder).toBeDefined();
    if (coder?.strategy !== "fusion") throw new Error("unreachable");
    expect(coder.image_describe).toBeUndefined();
    // Defaults inside an explicitly provided block still apply.
    const withBlock = parseConfig({
      ...parsed,
      models: {
        "fusion-coder": {
          ...coder,
          image_describe: { enabled: true, model: "vision-m" },
        },
      },
    });
    const blk = withBlock.models["fusion-coder"];
    if (blk?.strategy !== "fusion") throw new Error("unreachable");
    expect(blk.image_describe).toEqual({
      enabled: true,
      model: "vision-m",
      max_chars: 12000,
      timeout_s: 60,
    });
  });
});
