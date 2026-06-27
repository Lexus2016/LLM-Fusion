import { describe, it, expect } from "vitest";
import { z } from "zod";
import { smartStrategy } from "../src/strategies/smart";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { jsonResponse, sseResponse } from "./helpers";
import type { ChatCompletionRequest, FetchFn, StrategyContext, UpstreamClient } from "../src/types";

const logger = createLogger({ level: "silent" });

const upstream = { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 4 };

const config = parseConfig({
  upstream,
  models: {
    "fast-1": { strategy: "single", target: "deepseek" },
    "fusion-1": { strategy: "fusion", panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
    "smart-inline": {
      strategy: "smart",
      router: "rt",
      default: "simple",
      simple: { target: "deepseek" },
      fusion: { panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
    },
    "smart-ref": {
      strategy: "smart",
      router: "rt",
      default: "simple",
      simple: "fast-1",
      fusion: "fusion-1",
    },
    "smart-default-fusion": {
      strategy: "smart",
      router: "rt",
      default: "fusion",
      simple: { target: "deepseek" },
      fusion: { panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
    },
    "smart-no-escalate": {
      strategy: "smart",
      router: "rt",
      default: "simple",
      escalate_on_tool_error: false,
      simple: { target: "deepseek" },
      fusion: { panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
    },
    "fusion-pto": {
      strategy: "fusion",
      panel: ["p1", "p2", "p3"],
      judge: "jdg",
      synth: "syn",
      fusion_planning_turn_only: true,
    },
    "smart-ref-pto": {
      strategy: "smart",
      router: "rt",
      default: "simple",
      simple: { target: "deepseek" },
      fusion: "fusion-pto",
    },
  },
});

// --- Recording mock upstream (mirrors fusion.test harness) ----------------

const RecordedBodySchema = z
  .object({
    model: z.string(),
    stream: z.boolean().optional(),
    response_format: z.unknown().optional(),
    temperature: z.number().optional(),
    messages: z.array(z.unknown()).default([]),
  })
  .passthrough();
type RecordedBody = z.infer<typeof RecordedBodySchema>;

type ChatHandler = (body: RecordedBody) => Response | Promise<Response>;

interface Upstream {
  client: UpstreamClient;
  recorded: RecordedBody[];
  modelsCalled: () => string[];
  routerBodies: () => RecordedBody[];
}

function makeUpstream(chat: ChatHandler): Upstream {
  const recorded: RecordedBody[] = [];
  const fetchFn: FetchFn = async (input, init) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (url.endsWith("/api/show")) {
      return jsonResponse({ capabilities: ["completion"], model_info: {} });
    }
    if (url.endsWith("/v1/chat/completions") || url.endsWith("/api/chat")) {
      const body = RecordedBodySchema.parse(JSON.parse(String(init?.body)));
      recorded.push(body);
      return chat(body);
    }
    return jsonResponse({ error: `no route for ${url}` }, 404);
  };
  const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
  return {
    client,
    recorded,
    modelsCalled: () => recorded.map((b) => b.model),
    routerBodies: () => recorded.filter((b) => b.model === "rt"),
  };
}

function ctx(client: UpstreamClient, request: ChatCompletionRequest, model: string): StrategyContext {
  const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
  const entry = config.models[model];
  if (!entry) throw new Error(`test config missing '${model}'`);
  return { request, config, client, capabilities, logger, modelConfig: entry };
}

// --- Router responders ----------------------------------------------------

const routeSimple = (): Response =>
  jsonResponse({ choices: [{ message: { content: JSON.stringify({ route: "simple", reason: "easy" }) } }] });
const routeFusion = (): Response =>
  jsonResponse({ choices: [{ message: { content: JSON.stringify({ route: "fusion", reason: "hard" }) } }] });
const routerError = (): Response => jsonResponse({ error: "boom" }, 500);
const routerGarbage = (): Response =>
  jsonResponse({ choices: [{ message: { content: "not json {{{ definitely not" } }] });

/** Standard sub-route chat handler; the router model `rt` is delegated to `routerResp`. */
function chatWith(routerResp: () => Response): ChatHandler {
  const analysis = { consensus: "agree", disagreements: [], unique_insights: [], blind_spots: [] };
  return (body) => {
    if (body.model === "rt") return routerResp();
    if (body.model === "jdg") {
      return jsonResponse({ choices: [{ message: { content: JSON.stringify(analysis) } }] });
    }
    if (body.model === "syn") {
      if (body.stream === true) return sseResponse([{ choices: [{ delta: { content: "final" } }] }]);
      return jsonResponse({ choices: [{ message: { content: "final" } }] });
    }
    if (body.model === "deepseek" || body.model === "simp-t") {
      if (body.stream === true) {
        return sseResponse([{ choices: [{ delta: { content: "simple-answer" } }] }]);
      }
      return jsonResponse({ choices: [{ message: { content: "simple-answer" } }] });
    }
    // Panel members p1 / p2 / p3.
    return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
  };
}

const PANEL = ["p1", "p2", "p3"];
function req(model: string, extra: Record<string, unknown> = {}): ChatCompletionRequest {
  return { model, messages: [{ role: "user", content: "hello" }], ...extra };
}

/** A mid-agent-loop request whose latest tool result carries `toolContent`. */
function reqWithToolResult(model: string, toolContent: string): ChatCompletionRequest {
  return {
    model,
    messages: [
      { role: "user", content: "run the tests" },
      { role: "assistant", content: "calling bash" },
      { role: "tool", content: toolContent },
    ],
  };
}

describe("smart strategy", () => {
  it("route=simple runs only the simple path; router called once, non-streamed", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);

    const called = up.modelsCalled();
    expect(called).toContain("deepseek"); // simple target ran
    for (const p of [...PANEL, "jdg", "syn"]) expect(called).not.toContain(p); // fusion did NOT run

    const routerBodies = up.routerBodies();
    expect(routerBodies).toHaveLength(1); // exactly one router call
    expect(routerBodies[0]?.stream).not.toBe(true); // never streamed
  });

  it("escalates to fusion on a failing tool result WITHOUT calling the router", async () => {
    const up = makeUpstream(chatWith(routeSimple)); // router WOULD say "simple"
    const res = await smartStrategy.execute(
      ctx(up.client, reqWithToolResult("smart-inline", "AssertionError: expected 200, got 500"), "smart-inline"),
    );
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    // Fusion ran (panel + judge + synth) despite the router preferring "simple"...
    for (const m of [...PANEL, "jdg", "syn"]) expect(called).toContain(m);
    // ...and the router was never consulted — deterministic escalation, 0 round-trips.
    expect(up.routerBodies()).toHaveLength(0);
    expect(called).not.toContain("deepseek");
  });

  it("does NOT escalate on a successful tool result; the router decides", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const res = await smartStrategy.execute(
      ctx(up.client, reqWithToolResult("smart-inline", "All 12 tests passed in 1.2s"), "smart-inline"),
    );
    expect(res.status).toBe(200);
    expect(up.routerBodies()).toHaveLength(1); // router consulted as normal
    expect(up.modelsCalled()).toContain("deepseek"); // routed simple, fusion did not run
  });

  it("escalate_on_tool_error=false defers to the router even on a failing tool result", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const res = await smartStrategy.execute(
      ctx(
        up.client,
        reqWithToolResult("smart-no-escalate", "Traceback (most recent call last): ZeroDivisionError"),
        "smart-no-escalate",
      ),
    );
    expect(res.status).toBe(200);
    expect(up.routerBodies()).toHaveLength(1); // knob off -> router still runs
    expect(up.modelsCalled()).toContain("deepseek"); // routed simple by the router
  });

  it("escalation runs FULL fusion even when the referenced fusion model is planning-turn-only", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const res = await smartStrategy.execute(
      ctx(up.client, reqWithToolResult("smart-ref-pto", "Error: connection refused"), "smart-ref-pto"),
    );
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    // The panel must run — escalation overrides planning_turn_only's mid-loop degrade.
    for (const m of PANEL) expect(called).toContain(m);
    expect(up.routerBodies()).toHaveLength(0);
  });

  it("does NOT escalate when a tool result merely mentions an error mid-line (file read)", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const fileRead =
      "src/math.py:\n    def divide(a, b):\n        # raises ValueError: when b is 0\n        return a / b";
    const res = await smartStrategy.execute(
      ctx(up.client, reqWithToolResult("smart-inline", fileRead), "smart-inline"),
    );
    expect(res.status).toBe(200);
    // A successful file read whose CONTENT names an exception is not a failure.
    expect(up.routerBodies()).toHaveLength(1); // router decided; no escalation
    expect(up.modelsCalled()).toContain("deepseek");
  });

  it("does NOT escalate on clean tool output that contains the word 'errors'", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const res = await smartStrategy.execute(
      ctx(up.client, reqWithToolResult("smart-inline", "Lint complete: 0 errors, 0 warnings"), "smart-inline"),
    );
    expect(res.status).toBe(200);
    expect(up.routerBodies()).toHaveLength(1);
    expect(up.modelsCalled()).toContain("deepseek");
  });

  it("escalates on an npm ERR! failure and on a multi-line traceback", async () => {
    for (const failure of [
      "npm ERR! code ELIFECYCLE\nnpm ERR! errno 1",
      "Traceback (most recent call last):\n  File \"app.py\", line 7, in <module>\n    main()\nZeroDivisionError: division by zero",
    ]) {
      const up = makeUpstream(chatWith(routeSimple));
      const res = await smartStrategy.execute(
        ctx(up.client, reqWithToolResult("smart-inline", failure), "smart-inline"),
      );
      expect(res.status).toBe(200);
      expect(up.routerBodies()).toHaveLength(0); // escalated, router skipped
      for (const m of PANEL) expect(up.modelsCalled()).toContain(m);
    }
  });

  it("route=simple with an image to a non-vision target -> 400 capability gate (no upstream simple call)", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const imageReq = req("smart-inline", {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe" },
            {
              type: "image_url",
              image_url: {
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
              },
            },
          ],
        },
      ],
    });
    await expect(smartStrategy.execute(ctx(up.client, imageReq, "smart-inline"))).rejects.toThrow(
      /does not support image input/,
    );
    // The gate fires before the single executor, so the simple target never ran.
    expect(up.modelsCalled()).not.toContain("deepseek");
  });

  it("route=fusion runs panel+judge+synth; router called once, non-streamed", async () => {
    const up = makeUpstream(chatWith(routeFusion));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);

    const called = up.modelsCalled();
    for (const m of [...PANEL, "jdg", "syn"]) expect(called).toContain(m); // full panel→judge→synth
    expect(called).not.toContain("deepseek"); // simple path skipped

    const routerBodies = up.routerBodies();
    expect(routerBodies).toHaveLength(1);
    expect(routerBodies[0]?.stream).not.toBe(true);
  });

  it("router error (500) falls back to default=simple", async () => {
    const up = makeUpstream(chatWith(routerError));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);

    const called = up.modelsCalled();
    expect(called).toContain("deepseek");
    for (const p of PANEL) expect(called).not.toContain(p);
  });

  it("router error (500) falls back to default=fusion", async () => {
    const up = makeUpstream(chatWith(routerError));
    const res = await smartStrategy.execute(
      ctx(up.client, req("smart-default-fusion"), "smart-default-fusion"),
    );
    expect(res.status).toBe(200);

    const called = up.modelsCalled();
    for (const m of [...PANEL, "jdg", "syn"]) expect(called).toContain(m);
    expect(called).not.toContain("deepseek");
  });

  it("router returns garbage JSON -> falls back to default; request still succeeds", async () => {
    const up = makeUpstream(chatWith(routerGarbage));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("simple-answer");

    const called = up.modelsCalled();
    expect(called).toContain("deepseek");
    for (const p of PANEL) expect(called).not.toContain(p);
  });

  it("inline sub-configs route correctly for both simple and fusion", async () => {
    const upS = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(upS.client, req("smart-inline"), "smart-inline"));
    expect(upS.modelsCalled()).toContain("deepseek");

    const upF = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(ctx(upF.client, req("smart-inline"), "smart-inline"));
    for (const m of [...PANEL, "jdg", "syn"]) expect(upF.modelsCalled()).toContain(m);
  });

  it("string-reference sub-configs produce routing identical to inline blocks", async () => {
    // simple route
    const inlineS = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(inlineS.client, req("smart-inline"), "smart-inline"));
    const refS = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(refS.client, req("smart-ref"), "smart-ref"));
    expect([...refS.modelsCalled()].sort()).toEqual([...inlineS.modelsCalled()].sort());

    // fusion route
    const inlineF = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(ctx(inlineF.client, req("smart-inline"), "smart-inline"));
    const refF = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(ctx(refF.client, req("smart-ref"), "smart-ref"));
    expect([...refF.modelsCalled()].sort()).toEqual([...inlineF.modelsCalled()].sort());
  });

  it("stream=true + route=simple pipes the sub-route SSE; router stays non-streamed", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    const res = await smartStrategy.execute(
      ctx(up.client, req("smart-inline", { stream: true }), "smart-inline"),
    );
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("simple-answer");
    expect(text).toContain("[DONE]");

    // Router call carried stream:false; the simple sub-route carried stream:true.
    const routerBodies = up.routerBodies();
    expect(routerBodies).toHaveLength(1);
    expect(routerBodies[0]?.stream).not.toBe(true);
    const simpleBody = up.recorded.find((b) => b.model === "deepseek");
    expect(simpleBody?.stream).toBe(true);
  });
});

describe("smart config validation", () => {
  const base = { upstream };

  it("rejects a simple ref pointing at a missing model", () => {
    expect(() =>
      parseConfig({
        ...base,
        models: {
          "smart-x": {
            strategy: "smart",
            router: "r",
            simple: "does-not-exist",
            fusion: { panel: ["a"], judge: "a", synth: "b" },
          },
        },
      }),
    ).toThrow(/unknown model 'does-not-exist'/);
  });

  it("rejects a fusion ref pointing at a non-fusion model", () => {
    expect(() =>
      parseConfig({
        ...base,
        models: {
          "fast-1": { strategy: "single", target: "deepseek" },
          "smart-x": {
            strategy: "smart",
            router: "r",
            simple: "fast-1",
            fusion: "fast-1", // single, not fusion
          },
        },
      }),
    ).toThrow(/must point to a 'fusion' model/);
  });

  it("rejects a simple ref pointing at a non-single model", () => {
    expect(() =>
      parseConfig({
        ...base,
        models: {
          "fusion-1": { strategy: "fusion", panel: ["a"], judge: "a", synth: "b" },
          "smart-x": {
            strategy: "smart",
            router: "r",
            simple: "fusion-1", // fusion, not single
            fusion: "fusion-1",
          },
        },
      }),
    ).toThrow(/must point to a 'single' model/);
  });

  it("rejects a smart->smart self-reference", () => {
    expect(() =>
      parseConfig({
        ...base,
        models: {
          "smart-x": {
            strategy: "smart",
            router: "r",
            simple: "smart-x", // references itself (a smart model)
            fusion: { panel: ["a"], judge: "a", synth: "b" },
          },
        },
      }),
    ).toThrow(/cannot reference other smart models/);
  });
});
