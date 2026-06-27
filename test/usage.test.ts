import { describe, it, expect } from "vitest";
import pino from "pino";
import type { Logger } from "pino";
import { z } from "zod";
import { createApp } from "../src/server";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import type { Config } from "../src/config";
import type { FetchFn } from "../src/types";

/**
 * Upstream usage/cost accounting (spec §3 / §12). Exercised end-to-end through
 * the server so recording (strategies) + decoration (server) + the
 * `x-fusion-usage` header + the streamed final usage chunk + the structured log
 * line are all covered on the real path.
 */

// --- Per-model fixed usage (so sums are exact + assertable) ----------------

const USAGE: Record<string, { prompt_tokens: number; completion_tokens: number; total_tokens: number }> = {
  "single-target": { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  p1: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
  p2: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
  judge: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
  synth: { prompt_tokens: 20, completion_tokens: 8, total_tokens: 28 },
  router: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};

function usageFor(model: string) {
  return USAGE[model] ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
}

// --- Capturing logger (no cast: pino destination object) -------------------

function capturingLogger(): { logger: Logger; lines: Array<Record<string, unknown>> } {
  const lines: Array<Record<string, unknown>> = [];
  const logger = pino(
    { level: "info" },
    {
      write(s: string) {
        const parsed = z.record(z.string(), z.unknown()).safeParse(JSON.parse(s));
        if (parsed.success) lines.push(parsed.data);
      },
    },
  );
  return { logger, lines };
}

function usageLogLines(lines: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  return lines.filter((l) => l.msg === "request usage");
}

// --- Mock upstream: usage in JSON + a final stream chunk on include_usage ---

const BodySchema = z
  .object({
    model: z.string(),
    stream: z.boolean().optional(),
    stream_options: z.object({ include_usage: z.boolean().optional() }).passthrough().optional(),
    messages: z.array(z.object({ role: z.string(), content: z.unknown().optional() }).passthrough()).default([]),
  })
  .passthrough();

interface Recorded {
  model: string;
  stream: boolean;
  includeUsage: boolean;
}

function sseWithUsage(
  contentChunks: unknown[],
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null,
): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of contentChunks) {
        controller.enqueue(enc.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      }
      if (usage) {
        controller.enqueue(
          enc.encode(`data: ${JSON.stringify({ object: "chat.completion.chunk", choices: [], usage })}\n\n`),
        );
      }
      controller.enqueue(enc.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

function jsonResp(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

function makeUpstream(): { client: OllamaClient; recorded: Recorded[] } {
  const recorded: Recorded[] = [];
  const fetchFn: FetchFn = async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (url.endsWith("/api/show")) {
      return jsonResp({ capabilities: ["completion"], model_info: {} });
    }
    const body = BodySchema.parse(JSON.parse(String(init?.body)));
    recorded.push({
      model: body.model,
      stream: body.stream === true,
      includeUsage: body.stream_options?.include_usage === true,
    });
    const usage = usageFor(body.model);

    // Router: decide route from the transcript (so the test can steer smart).
    if (body.model === "router") {
      const transcript = JSON.stringify(body.messages);
      const route = transcript.includes("FUSION") ? "fusion" : "simple";
      return jsonResp({ choices: [{ message: { content: JSON.stringify({ route }) } }], usage });
    }
    // Judge: valid JSON analysis.
    if (body.model === "judge") {
      return jsonResp({
        choices: [{ message: { content: JSON.stringify({ consensus: "agree" }) } }],
        usage,
      });
    }
    // Synth / single target: stream when asked, else plain JSON.
    if (body.stream === true) {
      // reasoning then real content -> exercises the reasoning->content transform.
      return sseWithUsage(
        [
          { choices: [{ delta: { reasoning: "think " } }] },
          { choices: [{ delta: { content: "answer" } }] },
        ],
        body.stream_options?.include_usage === true ? usage : null,
      );
    }
    return jsonResp({ choices: [{ message: { content: "final" } }], usage });
  };
  const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
  return { client, recorded };
}

// --- Config builders -------------------------------------------------------

function buildConfig(pricing?: Config["pricing"]): Config {
  return parseConfig({
    upstream: { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 8 },
    models: {
      "single-m": { strategy: "single", target: "single-target" },
      "fusion-m": { strategy: "fusion", panel: ["p1", "p2"], judge: "judge", synth: "synth" },
      "smart-m": {
        strategy: "smart",
        router: "router",
        default: "simple",
        simple: { target: "single-target" },
        fusion: { panel: ["p1", "p2"], judge: "judge", synth: "synth" },
      },
    },
    ...(pricing ? { pricing } : {}),
  });
}

function makeApp(config: Config) {
  const up = makeUpstream();
  const cap = capturingLogger();
  const capabilities = new CapabilityService({ client: up.client, getOverrides: () => config.overrides, logger: cap.logger });
  const app = createApp({
    getConfig: () => config,
    client: up.client,
    capabilities,
    getAuthToken: () => undefined,
    logger: cap.logger,
  });
  return { app, recorded: up.recorded, logLines: cap.lines };
}

function post(app: ReturnType<typeof makeApp>["app"], body: unknown) {
  return app.request("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function parseHeaderUsage(res: Response): { calls: number; total: number } {
  const raw = res.headers.get("x-fusion-usage");
  if (!raw) throw new Error("missing x-fusion-usage header");
  return z.object({ calls: z.number(), total: z.number() }).parse(JSON.parse(raw));
}

/** Parse SSE text into the array of JSON chunk objects (drops [DONE] + blanks). */
function parseSse(text: string): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trimStart();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice("data:".length).trim();
    if (payload.length === 0 || payload === "[DONE]") continue;
    out.push(z.record(z.string(), z.unknown()).parse(JSON.parse(payload)));
  }
  return out;
}

const UsageObjSchema = z.object({
  prompt_tokens: z.number(),
  completion_tokens: z.number(),
  total_tokens: z.number(),
});

describe("usage accounting", () => {
  it("single non-stream: returns upstream usage; upstream_calls == 1", async () => {
    const { app, logLines } = makeApp(buildConfig());
    const res = await post(app, { model: "single-m", messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    const body = z.object({ choices: z.array(z.unknown()), usage: UsageObjSchema }).parse(JSON.parse(await res.text()));
    expect(body.usage).toEqual({ prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 });

    const header = parseHeaderUsage(res);
    expect(header).toEqual({ calls: 1, total: 15 });

    const log = usageLogLines(logLines).at(-1);
    expect(log?.upstream_calls).toBe(1);
    expect(log?.strategy).toBe("single");
    expect(log?.total_tokens).toBe(15);
    expect(log?.cost_usd).toBeNull();
  });

  it("fusion non-stream: aggregate == panel+judge+synth; calls == panel_size+2; header + body usage", async () => {
    const { app, logLines } = makeApp(buildConfig());
    const res = await post(app, { model: "fusion-m", messages: [{ role: "user", content: "hard" }] });
    expect(res.status).toBe(200);
    const body = z.object({ usage: UsageObjSchema }).parse(JSON.parse(await res.text()));
    // panel 2x{7,3,10} + judge {4,2,6} + synth {20,8,28}
    expect(body.usage).toEqual({ prompt_tokens: 38, completion_tokens: 16, total_tokens: 54 });

    const header = parseHeaderUsage(res);
    expect(header).toEqual({ calls: 4, total: 54 });

    const log = usageLogLines(logLines).at(-1);
    expect(log?.upstream_calls).toBe(4);
    expect(log?.strategy).toBe("fusion");
  });

  it("fusion stream: emits a final aggregate usage chunk; reasoning->content transform still works", async () => {
    const { app, recorded } = makeApp(buildConfig());
    const res = await post(app, { model: "fusion-m", stream: true, messages: [{ role: "user", content: "hard" }] });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    const chunks = parseSse(text);

    // Reasoning promoted into content, then real content (no regression).
    const ContentChunk = z.object({ choices: z.array(z.object({ delta: z.object({ content: z.string().optional() }).passthrough() }).passthrough()) });
    const contents: string[] = [];
    for (const c of chunks) {
      const parsed = ContentChunk.safeParse(c);
      if (!parsed.success) continue;
      const d = parsed.data.choices[0]?.delta.content;
      if (typeof d === "string") contents.push(d);
    }
    expect(contents).toContain("think "); // reasoning was promoted
    expect(contents).toContain("answer");

    // Exactly one usage chunk, carrying the full aggregate.
    const usageChunks = chunks.filter((c) => "usage" in c);
    expect(usageChunks).toHaveLength(1);
    const usage = UsageObjSchema.parse(usageChunks[0]!.usage);
    expect(usage).toEqual({ prompt_tokens: 38, completion_tokens: 16, total_tokens: 54 });

    // include_usage was set on the streamed (synth) request.
    const synth = recorded.find((r) => r.model === "synth");
    expect(synth?.stream).toBe(true);
    expect(synth?.includeUsage).toBe(true);
  });

  it("smart -> simple: usage = router + single (no double counting)", async () => {
    const { app, recorded } = makeApp(buildConfig());
    const res = await post(app, { model: "smart-m", messages: [{ role: "user", content: "trivial lookup" }] });
    const body = z.object({ usage: UsageObjSchema }).parse(JSON.parse(await res.text()));
    // router {3,1,4} + single-target {10,5,15}
    expect(body.usage).toEqual({ prompt_tokens: 13, completion_tokens: 6, total_tokens: 19 });
    expect(parseHeaderUsage(res)).toEqual({ calls: 2, total: 19 });
    expect(recorded.map((r) => r.model).sort()).toEqual(["router", "single-target"]);
  });

  it("smart -> fusion: usage = router + panel+judge+synth (no double counting)", async () => {
    const { app, recorded } = makeApp(buildConfig());
    const res = await post(app, { model: "smart-m", messages: [{ role: "user", content: "please FUSION this" }] });
    const body = z.object({ usage: UsageObjSchema }).parse(JSON.parse(await res.text()));
    // router {3,1,4} + panel 2x{7,3,10} + judge {4,2,6} + synth {20,8,28}
    expect(body.usage).toEqual({ prompt_tokens: 41, completion_tokens: 17, total_tokens: 58 });
    expect(parseHeaderUsage(res)).toEqual({ calls: 5, total: 58 });
    expect(recorded.map((r) => r.model).sort()).toEqual(["judge", "p1", "p2", "router", "synth"]);
  });

  it("cost: computed from a pricing map; null when no pricing", async () => {
    const pricing = {
      // synth priced; panel/judge unpriced -> only synth contributes.
      synth: { input_per_mtok: 1_000_000, output_per_mtok: 2_000_000 },
    };
    const { app, logLines } = makeApp(buildConfig(pricing));
    await post(app, { model: "fusion-m", messages: [{ role: "user", content: "hard" }] });
    const log = usageLogLines(logLines).at(-1);
    // synth usage {prompt:20, completion:8}: 20/1e6*1e6 + 8/1e6*2e6 = 20 + 16 = 36
    expect(log?.cost_usd).toBeCloseTo(36, 6);

    const noPrice = makeApp(buildConfig());
    await post(noPrice.app, { model: "fusion-m", messages: [{ role: "user", content: "hard" }] });
    expect(usageLogLines(noPrice.logLines).at(-1)?.cost_usd).toBeNull();
  });

  it("include_usage is NOT set on non-stream upstream requests", async () => {
    const { app, recorded } = makeApp(buildConfig());
    await post(app, { model: "fusion-m", messages: [{ role: "user", content: "hard" }] });
    expect(recorded.length).toBeGreaterThan(0);
    expect(recorded.every((r) => r.includeUsage === false)).toBe(true);
  });
});
