import { describe, it, expect, beforeEach, vi } from "vitest";
import { z } from "zod";
import { smartStrategy, __resetRouterCacheForTesting } from "../src/strategies/smart";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { createResilience } from "../src/concurrency";
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
    // Vision-capable simple target, for the router-hallucination-guard "image present" test.
    "vision-single": { strategy: "single", target: "vdeepseek" },
    "smart-vision": {
      strategy: "smart",
      router: "rt",
      default: "fusion",
      simple: "vision-single",
      fusion: { panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
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
type ShowHandler = (model: string) => Response;

interface Upstream {
  client: UpstreamClient;
  recorded: RecordedBody[];
  modelsCalled: () => string[];
  routerBodies: () => RecordedBody[];
}

function makeUpstream(chat: ChatHandler, show?: ShowHandler): Upstream {
  const recorded: RecordedBody[] = [];
  const fetchFn: FetchFn = async (input, init) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (url.endsWith("/api/show")) {
      const model = z.object({ model: z.string() }).parse(JSON.parse(String(init?.body))).model;
      return show ? show(model) : jsonResponse({ capabilities: ["completion"], model_info: {} });
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
    if (body.model === "deepseek" || body.model === "simp-t" || body.model === "vdeepseek") {
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
  // The router-decision cache is module-level; reset it between tests so a
  // prior test's cached decision does not short-circuit the router call the
  // next test asserts on.
  beforeEach(() => __resetRouterCacheForTesting());

  it("a router cache hit never consults the circuit breaker (no half-open probe leak)", async () => {
    // Regression for the half-open probe leak: canAttempt() reserves the probe slot
    // as a side effect, so consulting it before the cache short-circuit would leak the
    // probe on every cache hit and wedge the router breaker half-open forever. The
    // cache must be checked BEFORE the breaker. (single-path also calls canAttempt for
    // its target, so we count only router-model "rt" calls.)
    const up = makeUpstream(chatWith(routeSimple));
    const resilience = createResilience({ maxConcurrency: 4 });
    const spy = vi.spyOn(resilience.breaker, "canAttempt");
    const routerProbes = (): number => spy.mock.calls.filter((c) => c[0] === "rt").length;

    const first = { ...ctx(up.client, req("smart-inline"), "smart-inline"), resilience };
    await smartStrategy.execute(first);
    expect(routerProbes()).toBe(1); // uncached -> one real router attempt

    const second = { ...ctx(up.client, req("smart-inline"), "smart-inline"), resilience };
    await smartStrategy.execute(second);
    expect(routerProbes()).toBe(1); // cache hit -> breaker NOT consulted again
  });

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

  it("honors a router decision returned in `reasoning` with empty content (M-8 regression)", async () => {
    // A "thinking" router model may put its JSON route in `reasoning` and leave
    // `content` empty; the router must still pick that route, not the default.
    const routeFusionViaReasoning = (): Response =>
      jsonResponse({
        choices: [
          { message: { content: "", reasoning: JSON.stringify({ route: "fusion", reason: "hard" }) } },
        ],
      });
    const up = makeUpstream(chatWith(routeFusionViaReasoning));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);
    expect(up.routerBodies()).toHaveLength(1);
    const called = up.modelsCalled();
    for (const m of PANEL) expect(called).toContain(m); // fusion ran => reasoning was promoted
    expect(called).not.toContain("deepseek"); // did NOT silently fall back to the simple default
  });

  it("parses a router decision wrapped in ```json fences (no false default fallback)", async () => {
    const routeFusionFenced = (): Response =>
      jsonResponse({
        choices: [
          { message: { content: "```json\n" + JSON.stringify({ route: "fusion", reason: "hard" }) + "\n```" } },
        ],
      });
    const up = makeUpstream(chatWith(routeFusionFenced));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    for (const m of PANEL) expect(called).toContain(m); // fence stripped -> fusion route honored
    expect(called).not.toContain("deepseek");
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

  it("router-selected fusion forces the FULL panel mid-loop, even for a planning-turn-only fusion ref", async () => {
    // Non-error tool result -> escalation does NOT fire, so this exercises the ROUTER
    // path. The router chooses fusion; without the fix, fusion-pto's planning_turn_only
    // would degrade this tool-continuation step back to synth-only, silently overriding
    // the router's decision. With the fix, the full panel runs.
    const up = makeUpstream(chatWith(routeFusion));
    const res = await smartStrategy.execute(
      ctx(up.client, reqWithToolResult("smart-ref-pto", "All 12 tests passed in 1.2s."), "smart-ref-pto"),
    );
    expect(res.status).toBe(200);
    expect(up.routerBodies()).toHaveLength(1); // the router was consulted (not escalation)
    const called = up.modelsCalled();
    for (const m of PANEL) expect(called).toContain(m); // full panel ran, not synth-only
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

  // --- router hallucination guard: an image/screenshot "reason" with no image present ---
  const routeSimpleClaimingImage = (): Response =>
    jsonResponse({
      choices: [
        {
          message: {
            content: JSON.stringify({
              route: "simple",
              reason: "User sent a screenshot/image in response to a completed task; routine interpretation.",
            }),
          },
        },
      ],
    });
  const routeFusionClaimingImage = (): Response =>
    jsonResponse({
      choices: [
        {
          message: {
            content: JSON.stringify({
              route: "fusion",
              reason: "User attached a screenshot; needs deep analysis.",
            }),
          },
        },
      ],
    });

  it("router claims an image that is NOT present -> treats as untrustworthy, falls back to default=simple", async () => {
    __resetRouterCacheForTesting();
    const up = makeUpstream(chatWith(routeSimpleClaimingImage));
    // Plain-text request, no image_url blocks.
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    // Default is simple -> deepseek ran; the router's "simple" choice happened to
    // match default, but the guard is what made the decision trustworthy.
    expect(called).toContain("deepseek");
  });

  it("router claims an image that is NOT present, chose fusion -> falls back to default=simple (cost control)", async () => {
    __resetRouterCacheForTesting();
    const up = makeUpstream(chatWith(routeFusionClaimingImage));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    // Guard caught the fabricated image; default=simple runs, fusion panel does NOT.
    expect(called).toContain("deepseek");
    for (const p of PANEL) expect(called).not.toContain(p);
  });

  it("router claims an image that IS present -> decision is trusted (no fallback)", async () => {
    __resetRouterCacheForTesting();
    // smart-vision has default=fusion, so a guard fallback would run the PANEL.
    // If the guard correctly trusts the router (an image really is present), the
    // router's "simple" choice runs the vision-capable simple target and the
    // panel does NOT.
    const show = (model: string): Response =>
      jsonResponse({
        capabilities: model === "vdeepseek" ? ["completion", "vision"] : ["completion"],
        model_info: {},
      });
    const up = makeUpstream(chatWith(routeSimpleClaimingImage), show);
    const res = await smartStrategy.execute(
      ctx(
        up.client,
        req("smart-vision", {
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "look at this" },
                { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
              ],
            },
          ],
        }),
        "smart-vision",
      ),
    );
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    // Router said simple and the image claim is true -> trusted -> simple runs,
    // panel does NOT (which it WOULD have, had the guard fallen back to default=fusion).
    expect(called).toContain("vdeepseek");
    for (const p of PANEL) expect(called).not.toContain(p);
  });

  it("router MENTIONS an image in a negating context (no image present) -> NOT flagged, decision trusted", async () => {
    // Real production case: user pasted text; the router correctly observed
    // "no actual image attachment / plain-text" but mentioned the word "image".
    // The guard must NOT treat that as a hallucination and override a correct call.
    __resetRouterCacheForTesting();
    const routeNegatedImage = (): Response =>
      jsonResponse({
        choices: [
          {
            message: {
              content: JSON.stringify({
                route: "simple",
                reason:
                  "The latest user message is a plain-text marker '[multimodal content]' with no actual image attachment; routine conversational interaction.",
              }),
            },
          },
        ],
      });
    // smart-default-fusion: default=fusion. If the guard WRONGLY fired, it would
    // fall back to fusion (panel runs). If it correctly trusts the router, simple
    // (deepseek) runs and the panel does NOT.
    const up = makeUpstream(chatWith(routeNegatedImage));
    const res = await smartStrategy.execute(
      ctx(up.client, req("smart-default-fusion"), "smart-default-fusion"),
    );
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    expect(called).toContain("deepseek"); // router's "simple" trusted
    for (const p of PANEL) expect(called).not.toContain(p); // no fallback to fusion
  });

  it("router claims an image that is NOT present with 'not plain-text' reason -> treats as untrustworthy, falls back to default=simple", async () => {
    __resetRouterCacheForTesting();
    const routeNotPlainTextClaimingImage = (): Response =>
      jsonResponse({
        choices: [
          {
            message: {
              content: JSON.stringify({
                route: "fusion",
                reason: "The request is not a plain-text query, it has an image screenshot attached.",
              }),
            },
          },
        ],
      });
    const up = makeUpstream(chatWith(routeNotPlainTextClaimingImage));
    const res = await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    expect(called).toContain("deepseek");
    for (const p of PANEL) expect(called).not.toContain(p);
  });

  it("inline sub-configs route correctly for both simple and fusion", async () => {
    const upS = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(upS.client, req("smart-inline"), "smart-inline"));
    expect(upS.modelsCalled()).toContain("deepseek");

    // Same `req("smart-inline")` yields an identical router body, so the decision
    // is cached from the simple half; reset to force a fresh router call here.
    __resetRouterCacheForTesting();
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
    // The router model `rt` is the same across both; compare only the sub-route.
    expect([...refS.modelsCalled()].filter((m) => m !== "rt").sort()).toEqual(
      [...inlineS.modelsCalled()].filter((m) => m !== "rt").sort(),
    );

    // fusion route — reset so the cached simple decision does not short-circuit.
    __resetRouterCacheForTesting();
    const inlineF = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(ctx(inlineF.client, req("smart-inline"), "smart-inline"));
    const refF = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(ctx(refF.client, req("smart-ref"), "smart-ref"));
    expect([...refF.modelsCalled()].filter((m) => m !== "rt").sort()).toEqual(
      [...inlineF.modelsCalled()].filter((m) => m !== "rt").sort(),
    );
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

  it("router prompt stays bounded on a huge conversation (system + first task + recent window)", async () => {
    // A multi-day agent loop can carry a ~350k-token history. The router must NOT
    // ingest all of it — that full-context call measured ~57s live and intermittently
    // hit the 120s upstream timeout, silently dropping to the default route. It now
    // sees a compact view: the system role, the original task, and the recent turns.
    const huge = "X".repeat(40_000); // file-dump-sized messages
    const messages: ChatCompletionRequest["messages"] = [
      { role: "system", content: "You are a coding agent. SYSTEM_MARKER" },
      { role: "user", content: "ORIGINAL_TASK_MARKER: refactor the auth module" },
    ];
    for (let i = 0; i < 20; i++) {
      messages.push({ role: "assistant", content: `step ${i} ${huge}` });
      messages.push({ role: "tool", content: `output ${i} ${huge}` });
    }
    messages.push({ role: "tool", content: "LATEST_MARKER: the next step reads a file" });
    const rawTotal = messages.reduce((n, m) => n + String(m.content).length, 0);

    const up = makeUpstream(chatWith(routeFusion));
    const res = await smartStrategy.execute(ctx(up.client, { model: "smart-inline", messages }, "smart-inline"));
    expect(res.status).toBe(200);

    const routerBodies = up.routerBodies();
    expect(routerBodies).toHaveLength(1);
    const userMsg = z.object({ content: z.string() }).parse((routerBodies[0]?.messages ?? [])[1]);
    const transcript = userMsg.content;

    // Bounded: the router prompt is a small constant, NOT the ~1.6M-char history.
    expect(rawTotal).toBeGreaterThan(800_000);
    expect(transcript.length).toBeLessThan(50_000);
    // Still carries every signal the router actually needs to classify the next action.
    expect(transcript).toContain("SYSTEM_MARKER"); // agent role
    expect(transcript).toContain("ORIGINAL_TASK_MARKER"); // overall intent (first user message)
    expect(transcript).toContain("LATEST_MARKER"); // recent state (latest message)
    expect(transcript).toContain("earlier messages omitted"); // the middle is summarized, not dropped silently
  });

  it("keeps the active user instruction even when it predates the recent window", async () => {
    // Multi-day session: an OLD task first, then a NEW sub-task, then more than
    // ROUTER_RECENT_WINDOW mechanical tool turns. The new instruction must still
    // reach the router — otherwise the substantive coding step that follows gets
    // misrouted to `simple` (the exact failure this strategy exists to prevent).
    const huge = "X".repeat(20_000);
    const messages: ChatCompletionRequest["messages"] = [
      { role: "system", content: "coding agent" },
      { role: "user", content: "OLD_TASK: set up the repo" },
      { role: "assistant", content: "repo ready" },
      { role: "user", content: "ACTIVE_TASK: refactor the auth module to support JWT" },
    ];
    for (let i = 0; i < 10; i++) {
      messages.push({ role: "assistant", content: `grep step ${i}` });
      messages.push({ role: "tool", content: `file ${i} ${huge}` });
    }

    const up = makeUpstream(chatWith(routeFusion));
    const res = await smartStrategy.execute(ctx(up.client, { model: "smart-inline", messages }, "smart-inline"));
    expect(res.status).toBe(200);

    const userMsg = z.object({ content: z.string() }).parse((up.routerBodies()[0]?.messages ?? [])[1]);
    expect(userMsg.content).toContain("ACTIVE_TASK"); // survived 20 mechanical turns of truncation
    expect(userMsg.content).toContain("OLD_TASK"); // first-message framing kept too
  });

  it("a router that hangs past router_timeout_s degrades to the default route (HIGH-2)", async () => {
    // Without a stage timeout the router could hang up to the full upstream timeout;
    // it must instead time out fast and fall back to the configured default route.
    const fastCfg = parseConfig({
      upstream,
      defaults: { router_timeout_s: 1 },
      models: {
        "smart-to": {
          strategy: "smart",
          router: "rt",
          default: "simple",
          simple: { target: "deepseek" },
          fusion: { panel: ["p1", "p2", "p3"], judge: "jdg", synth: "syn" },
        },
      },
    });
    const up = makeUpstream((body) => {
      if (body.model === "rt") return new Promise<Response>(() => {}); // router hangs forever
      return jsonResponse({ choices: [{ message: { content: "simple-answer" } }] });
    });
    const entry = fastCfg.models["smart-to"];
    if (!entry) throw new Error("missing smart-to");
    const capabilities = new CapabilityService({ client: up.client, getOverrides: () => fastCfg.overrides, logger });
    const sctx: StrategyContext = {
      request: req("smart-to"),
      config: fastCfg,
      client: up.client,
      capabilities,
      logger,
      modelConfig: entry,
    };
    const res = await smartStrategy.execute(sctx);
    expect(res.status).toBe(200);
    expect(up.modelsCalled()).toContain("deepseek"); // degraded to default=simple after the router timed out
  }, 6000);

  it("router decision is cached: an identical second request skips the router call", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(up.routerBodies()).toHaveLength(1);

    // Same request -> identical router body -> cache hit, no second router call.
    const up2 = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(up2.client, req("smart-inline"), "smart-inline"));
    expect(up2.routerBodies()).toHaveLength(0);
    // The cached decision still dispatches to the simple sub-route.
    expect(up2.modelsCalled()).toContain("deepseek");
  });

  it("concurrent identical requests coalesce onto a single router call", async () => {
    // Block the router call until both requests are in flight, so a cache-hit
    // shortcut cannot masquerade as coalescing: the router cannot resolve until
    // we release it, by which point both classify() calls must have entered.
    let release!: () => void;
    const routerBlocked = new Promise<void>((r) => (release = r));
    const chat = chatWith(routeSimple);
    const up = makeUpstream((body) => {
      if (body.model === "rt") return routerBlocked.then(() => chat(body));
      return chat(body);
    });

    // Fire both without awaiting so they race into classify() concurrently.
    const pA = smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    const pB = smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    // Let the event loop turn so both have reached the in-flight pending entry.
    await new Promise((r) => setImmediate(r));
    release();

    const [a, b] = await Promise.all([pA, pB]);
    expect(a.status).toBe(200);
    expect(b.status).toBe(200);
    // Coalesced: only ONE router call serviced both concurrent requests. Without
    // coalescing the blocked router would have been invoked twice.
    expect(up.routerBodies()).toHaveLength(1);
  });

  it("router cache is keyed by the request: a different message forces a fresh router call", async () => {
    const up = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(ctx(up.client, req("smart-inline"), "smart-inline"));
    expect(up.routerBodies()).toHaveLength(1);

    // A different user message changes the router body -> cache miss -> new call.
    const up2 = makeUpstream(chatWith(routeFusion));
    await smartStrategy.execute(
      ctx(up2.client, { ...req("smart-inline"), messages: [{ role: "user", content: "different" }] }, "smart-inline"),
    );
    // A body-keyed cache means the second request does NOT reuse the first
    // decision, so the router must be called again.
    expect(up2.routerBodies()).toHaveLength(1);
  });

  it("a router failure is NOT cached: an identical retry re-invokes the router", async () => {
    // First request: router returns garbage -> fallback to default, uncached.
    const up1 = makeUpstream(chatWith(routerGarbage));
    await smartStrategy.execute(ctx(up1.client, req("smart-inline"), "smart-inline"));
    expect(up1.routerBodies()).toHaveLength(1);

    // Identical request must still call the router (no cached fallback), and this
    // time it succeeds -> the decision is now cached.
    const up2 = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(up2.client, req("smart-inline"), "smart-inline"));
    expect(up2.routerBodies()).toHaveLength(1);
    expect(up2.modelsCalled()).toContain("deepseek");

    // Third identical request now hits the cache.
    const up3 = makeUpstream(chatWith(routeSimple));
    await smartStrategy.execute(ctx(up3.client, req("smart-inline"), "smart-inline"));
    expect(up3.routerBodies()).toHaveLength(0);
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
    ).toThrow(/must point to a 'single or failover' model/);
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

  // --- Issue 1: routerMessageLine extracts text from array content ----------

  it("router sees actual text from array-formatted messages, not '[multimodal content]'", async () => {
    const up = makeUpstream(chatWith(routeSimple));
    // Send a message with content as an array of text parts (standard OpenAI format).
    // Before the fix, the router would see "[multimodal content]" instead of the actual text.
    const request: ChatCompletionRequest = {
      model: "smart-inline",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "design a new authentication system with OAuth2 and JWT" },
          ],
        },
      ],
    };
    const res = await smartStrategy.execute(ctx(up.client, request, "smart-inline"));
    expect(res.status).toBe(200);

    // Verify the router was called and its prompt contains the actual text,
    // not the old "[multimodal content]" placeholder.
    const routerCalls = up.routerBodies();
    expect(routerCalls.length).toBe(1);
    const routerMessages = routerCalls[0]?.messages ?? [];
    const userMsg = routerMessages.find(
      (m: unknown) => typeof m === "object" && m !== null && (m as Record<string, unknown>).role === "user",
    );
    expect(userMsg).toBeDefined();
    const userContent = (userMsg as Record<string, unknown>).content;
    expect(typeof userContent).toBe("string");
    expect(userContent).toContain("OAuth2");
    expect(userContent).toContain("JWT");
    expect(userContent).not.toContain("[multimodal content]");
  });

  it("router sees '[has image]' prefix when array content includes an image_url part", async () => {
    const show = (model: string): Response =>
      jsonResponse({
        capabilities: model === "vdeepseek" ? ["completion", "vision"] : ["completion"],
        model_info: {},
      });
    const up = makeUpstream(chatWith(routeSimple), show);
    const request: ChatCompletionRequest = {
      model: "smart-vision",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this screenshot" },
            { type: "image_url", image_url: { url: "data:image/png;base64,abc" } },
          ],
        },
      ],
    };
    const res = await smartStrategy.execute(ctx(up.client, request, "smart-vision"));
    expect(res.status).toBe(200);

    const routerCalls = up.routerBodies();
    expect(routerCalls.length).toBe(1);
    const routerMessages = routerCalls[0]?.messages ?? [];
    const userMsg = routerMessages.find(
      (m: unknown) => typeof m === "object" && m !== null && (m as Record<string, unknown>).role === "user",
    );
    const userContent = (userMsg as Record<string, unknown>).content;
    expect(typeof userContent).toBe("string");
    expect(userContent).toContain("[has image]");
    expect(userContent).toContain("describe this screenshot");
  });

  // --- Issue 3: fusion panel failure falls back to simple -------------------

  it("auto-falls back to simple when fusion panel fails with AllMembersFailedError", async () => {
    let callCount = 0;
    const up = makeUpstream((body) => {
      callCount++;
      if (body.model === "rt") return routeFusion(); // router picks fusion
      // Panel members all fail with 400 (simulating context overflow)
      if (body.model === "p1" || body.model === "p2" || body.model === "p3") {
        return jsonResponse(
          { error: { message: "prompt too long; exceeded max context length" } },
          400,
        );
      }
      // Simple fallback target
      if (body.model === "deepseek") {
        return jsonResponse({ choices: [{ message: { content: "fallback-answer" } }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await smartStrategy.execute(
      ctx(up.client, req("smart-inline"), "smart-inline"),
    );
    // Should succeed (200) via simple fallback, not fail with 502.
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("fallback-answer");
    // Verify the simple model was called (the fallback path).
    expect(up.modelsCalled()).toContain("deepseek");
  });

  it("auto-falls back to simple when fusion fails with a non-AllMembersFailedError (e.g. CircuitOpenError for synth)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "rt") return routeFusion(); // router picks fusion
      if (body.model === "p1" || body.model === "p2" || body.model === "p3") {
        return jsonResponse({ choices: [{ message: { content: "panel-ans" } }] });
      }
      if (body.model === "jdg") {
        return jsonResponse({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  disagreements: [],
                  consensus: "consensus",
                  fragile_claims: [],
                  confidence: "high",
                }),
              },
            },
          ],
        });
      }
      if (body.model === "syn") {
        return jsonResponse({ error: { message: "synth circuit open / network error" } }, 503);
      }
      if (body.model === "deepseek") {
        return jsonResponse({ choices: [{ message: { content: "fallback-answer-from-synth-error" } }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await smartStrategy.execute(
      ctx(up.client, req("smart-inline"), "smart-inline"),
    );
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("fallback-answer-from-synth-error");
    expect(up.modelsCalled()).toContain("deepseek");
  });
});
