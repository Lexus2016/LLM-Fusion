import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { z } from "zod";
import { createFusionStrategy, fusionStrategy, compressPanelMessages } from "../src/strategies/fusion";
import type { TimerFactory } from "../src/strategies/fusion";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createLogger } from "../src/logging";
import { jsonResponse, sseResponse } from "./helpers";
import type { ChatCompletionRequest, FetchFn, StrategyContext, UpstreamClient } from "../src/types";

const logger = createLogger({ level: "silent" });

const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 4 },
  models: {
    "fusion-1": { strategy: "fusion", panel: ["m1", "m2", "m3"], judge: "j", synth: "s" },
    "fusion-bypass": {
      strategy: "fusion",
      panel: ["m1", "m2"],
      judge: "j",
      synth: "s",
      tool_mode: "bypass",
    },
    "fusion-planning": {
      strategy: "fusion",
      panel: ["m1", "m2"],
      judge: "j",
      synth: "s",
      fusion_planning_turn_only: true,
    },
    "fusion-vision": { strategy: "fusion", panel: ["vm1", "vm2"], judge: "j", synth: "vs" },
    // Synth-only (bypass) with a NON-vision panel but a vision-capable synth: the
    // vision gate must validate only the synth here (the panel never runs).
    "fusion-bypass-vision": {
      strategy: "fusion",
      panel: ["nv1", "nv2"],
      judge: "j",
      synth: "vs",
      tool_mode: "bypass",
    },
    "fusion-no-promote": {
      strategy: "fusion",
      panel: ["m1", "m2", "m3"],
      judge: "j",
      synth: "s",
      promote_reasoning_to_content: false,
    },
    // Adversarial panel slot: m2 runs with a contrarian prompt.
    "fusion-adv": {
      strategy: "fusion",
      panel: ["m1", "m2", "m3"],
      judge: "j",
      synth: "s",
      adversarial: "m2",
    },
    // Web grounding: opt-in via web_search.enabled; needs TAVILY_API_KEY at runtime.
    "fusion-web": {
      strategy: "fusion",
      panel: ["m1", "m2", "m3"],
      judge: "j",
      synth: "s",
      web_search: { enabled: true, max_results: 3, timeout_s: 10, max_context_chars: 4000 },
    },
  },
});

const TOOLS = [
  { type: "function", function: { name: "read_file", description: "Read a file from disk" } },
];

// --- Recording mock upstream ----------------------------------------------

const RecordedBodySchema = z
  .object({
    model: z.string(),
    stream: z.boolean().optional(),
    tools: z.unknown().optional(),
    tool_choice: z.unknown().optional(),
    response_format: z.unknown().optional(),
    temperature: z.number().optional(),
    messages: z.array(z.unknown()).default([]),
  })
  .passthrough();
type RecordedBody = z.infer<typeof RecordedBodySchema>;

type ChatHandler = (body: RecordedBody, signal?: AbortSignal) => Response | Promise<Response>;
type ShowHandler = (model: string) => Response;

interface Upstream {
  client: UpstreamClient;
  recorded: RecordedBody[];
  modelsCalled: () => string[];
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
      return chat(body, init?.signal ?? undefined);
    }
    return jsonResponse({ error: `no route for ${url}` }, 404);
  };
  const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
  return { client, recorded, modelsCalled: () => recorded.map((b) => b.model) };
}

function ctx(client: UpstreamClient, request: ChatCompletionRequest, model = "fusion-1"): StrategyContext {
  const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
  const entry = config.models[model];
  if (!entry) throw new Error(`test config missing '${model}'`);
  return { request, config, client, capabilities, logger, modelConfig: entry };
}

/** Default chat handler: panel members answer `ans-<model>`, judge returns valid JSON, synth `final`. */
function defaultChat(judgeJson = true, synthStream = false): ChatHandler {
  const analysis = { consensus: "they agree", disagreements: [], unique_insights: [], blind_spots: [] };
  return (body) => {
    if (body.model === "j") {
      return jsonResponse({
        choices: [{ message: { content: judgeJson ? JSON.stringify(analysis) : "this is not json{{" } }],
      });
    }
    if (body.model === "s" || body.model === "vs") {
      if (synthStream && body.stream === true) {
        return sseResponse([{ choices: [{ delta: { content: "final" } }] }]);
      }
      return jsonResponse({ choices: [{ message: { content: "final" } }] });
    }
    return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
  };
}

/** Extract string-content system messages from a recorded body. */
function systemContents(body: RecordedBody): string[] {
  const out: string[] = [];
  const MsgSchema = z.object({ role: z.string(), content: z.string() }).passthrough();
  for (const m of body.messages) {
    const parsed = MsgSchema.safeParse(m);
    if (parsed.success && parsed.data.role === "system") out.push(parsed.data.content);
  }
  return out;
}

function userContents(body: RecordedBody): string[] {
  const out: string[] = [];
  const MsgSchema = z.object({ role: z.string(), content: z.string() }).passthrough();
  for (const m of body.messages) {
    const parsed = MsgSchema.safeParse(m);
    if (parsed.success && parsed.data.role === "user") out.push(parsed.data.content);
  }
  return out;
}

/** Non-empty `delta.content` fragments, in order, from a client SSE transcript. */
function streamedContents(text: string): string[] {
  const ChunkSchema = z
    .object({
      choices: z
        .array(
          z
            .object({ delta: z.object({ content: z.string().optional() }).passthrough().optional() })
            .passthrough(),
        )
        .optional(),
    })
    .passthrough();
  const out: string[] = [];
  for (const line of text.split("\n")) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice("data:".length).trim();
    if (payload.length === 0 || payload === "[DONE]") continue;
    let raw: unknown;
    try {
      raw = JSON.parse(payload);
    } catch {
      continue;
    }
    const parsed = ChunkSchema.safeParse(raw);
    if (!parsed.success) continue;
    for (const choice of parsed.data.choices ?? []) {
      const content = choice.delta?.content;
      if (typeof content === "string" && content.length > 0) out.push(content);
    }
  }
  return out;
}

const req = (over: Partial<ChatCompletionRequest> = {}): ChatCompletionRequest => ({
  model: "fusion-1",
  messages: [{ role: "user", content: "hello" }],
  ...over,
});

describe("fusion strategy — panel/judge/synth", () => {
  it("fans out to all panel members in PARALLEL, strips tools from panel, gives tools to synth", async () => {
    // A barrier of size 3: each panel member blocks until all three have entered.
    // If the panel ran sequentially this deadlocks and the test times out.
    let entered = 0;
    let release!: () => void;
    const allEntered = new Promise<void>((r) => (release = r));
    const chat = defaultChat();
    const up = makeUpstream(async (body) => {
      if (body.model.startsWith("m")) {
        entered += 1;
        if (entered >= 3) release();
        await allEntered;
      }
      return chat(body);
    });

    const res = await fusionStrategy.execute(ctx(up.client, req({ tools: TOOLS })));
    expect(res.status).toBe(200);

    const called = up.modelsCalled();
    expect(called).toContain("m1");
    expect(called).toContain("m2");
    expect(called).toContain("m3");

    // Tool gate: NO panel call carried `tools`/`tool_choice`.
    const panelBodies = up.recorded.filter((b) => b.model.startsWith("m"));
    expect(panelBodies).toHaveLength(3);
    for (const b of panelBodies) {
      expect(b.tools).toBeUndefined();
      expect(b.tool_choice).toBeUndefined();
      // The tool list was injected as prose context instead.
      expect(systemContents(b).join("\n")).toContain("read_file");
    }

    // Synth DID receive the real tools schema.
    const synthBody = up.recorded.find((b) => b.model === "s");
    expect(synthBody).toBeDefined();
    expect(synthBody?.tools).toEqual(TOOLS);
  });

  it("proceeds on partial panel failure (1 of 3 fails, min_panel_success=1)", async () => {
    const chat = defaultChat();
    const up = makeUpstream((body) => {
      if (body.model === "m2") return jsonResponse({ error: "boom" }, 500);
      return chat(body);
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);

    // Judge saw the two survivors, not the failed member.
    const judgeBody = up.recorded.find((b) => b.model === "j");
    expect(judgeBody).toBeDefined();
    const judgeInput = userContents(judgeBody!).join("\n");
    expect(judgeInput).toContain("ans-m1");
    expect(judgeInput).toContain("ans-m3");
    expect(judgeInput).not.toContain("ans-m2");
  });

  it("runs the adversarial member with a contrarian prompt, others without it", async () => {
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(ctx(up.client, req(), "fusion-adv"));
    expect(res.status).toBe(200);

    const panelBodies = up.recorded.filter((b) => b.model.startsWith("m"));
    expect(panelBodies).toHaveLength(3);
    const m2 = panelBodies.find((b) => b.model === "m2");
    const others = panelBodies.filter((b) => b.model !== "m2");
    expect(m2).toBeDefined();
    // The adversarial member got the red-team system prompt...
    expect(systemContents(m2!).join("\n")).toContain("adversarial reviewer");
    expect(systemContents(m2!).join("\n")).toMatch(/find what is wrong|steelman|edge cases/i);
    // ...the other members did NOT.
    for (const b of others) {
      expect(systemContents(b).join("\n")).not.toContain("adversarial reviewer");
    }
    // Invariant untouched: no panel member carried real tools.
    for (const b of panelBodies) {
      expect(b.tools).toBeUndefined();
      expect(b.tool_choice).toBeUndefined();
    }
  });

  it("rejects an adversarial member that is not in the panel (config validation)", () => {
    expect(() =>
      parseConfig({
        upstream: { base_url: "https://mock.test", api_key_env: "X" },
        models: {
          "bad-adv": {
            strategy: "fusion",
            panel: ["m1", "m2"],
            judge: "j",
            synth: "s",
            adversarial: "m9", // not a panel member
          },
        },
      }),
    ).toThrow(/adversarial='m9'.*not listed in its panel/);
  });

  it("cancels other panel members early if min_panel_success is met", async () => {
    let m2Aborted = false;
    let m3Aborted = false;
    const chat = defaultChat();
    const up = makeUpstream((body, signal) => {
      if (body.model === "m1") {
        return chat(body);
      }
      if (body.model === "m2") {
        return new Promise<Response>((resolve, reject) => {
          signal?.addEventListener("abort", () => {
            m2Aborted = true;
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      }
      if (body.model === "m3") {
        return new Promise<Response>((resolve, reject) => {
          signal?.addEventListener("abort", () => {
            m3Aborted = true;
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      }
      return chat(body);
    });

    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);

    // Wait a brief moment for async promises to settle
    await new Promise((r) => setTimeout(r, 10));

    expect(m2Aborted).toBe(true);
    expect(m3Aborted).toBe(true);

    // Judge saw only the answer from m1
    const judgeBody = up.recorded.find((b) => b.model === "j");
    expect(judgeBody).toBeDefined();
    const judgeInput = userContents(judgeBody!).join("\n");
    expect(judgeInput).toContain("ans-m1");
    expect(judgeInput).not.toContain("ans-m2");
    expect(judgeInput).not.toContain("ans-m3");
  });

  it("waits for the adversarial member even after min_panel_success is met (does not drop it)", async () => {
    // min_panel_success default = 1. m1 answers instantly -> success met. The
    // adversarial member (m2) is slow. Without the wait-for-adversarial fix the
    // promise would resolve at m1's success and drop m2's in-flight red-team answer.
    // With the fix, m2 is waited for and its answer reaches the judge; m3 (a non-
    // adversarial straggler) IS early-cancelled.
    let m3Aborted = false;
    const chat = defaultChat(true);
    const up = makeUpstream((body, signal) => {
      if (body.model === "m1") return chat(body); // instant
      if (body.model === "m2") {
        // adversarial: deliver after a short delay so it is NOT yet done at m1's success
        return new Promise<Response>((resolve) => {
          setTimeout(() => resolve(chat(body)), 40);
        });
      }
      if (body.model === "m3") {
        return new Promise<Response>((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            m3Aborted = true;
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      }
      return chat(body);
    });

    const res = await fusionStrategy.execute(ctx(up.client, req(), "fusion-adv"));
    expect(res.status).toBe(200);

    // The adversarial member's answer was waited for and reached the judge.
    const judgeBody = up.recorded.find((b) => b.model === "j");
    expect(judgeBody).toBeDefined();
    const judgeInput = userContents(judgeBody!).join("\n");
    expect(judgeInput).toContain("ans-m2");
    // The non-adversarial straggler was cancelled as soon as success was met.
    expect(m3Aborted).toBe(true);
  });

  it("cancels panel members early even if they have started delivering tokens to free up concurrency slots", async () => {
    let m2Aborted = false;
    const chat = defaultChat();
    const up = makeUpstream((body, signal) => {
      if (body.model === "m1") {
        return chat(body);
      }
      if (body.model === "m2") {
        // Stream one token, then block
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"chunk"}}]}\n'));
            signal?.addEventListener("abort", () => {
              m2Aborted = true;
            });
          }
        });
        return jsonResponse(stream);
      }
      return chat(body);
    });

    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);

    // Wait a brief moment for async promises to settle
    await new Promise((r) => setTimeout(r, 10));

    // m2 has started delivering, but should be aborted on early success to free up limit slots!
    expect(m2Aborted).toBe(true);
  });

  it("times out a slow panel member and proceeds with the survivors", async () => {
    // Injected timer fires after 5ms regardless of the configured 90s.
    const fastTimer: TimerFactory = () => {
      let h: ReturnType<typeof setTimeout>;
      const expired = new Promise<void>((resolve) => {
        h = setTimeout(resolve, 5);
      });
      return { expired, cancel: () => clearTimeout(h) };
    };
    const strategy = createFusionStrategy({ timer: fastTimer });
    const chat = defaultChat();
    const up = makeUpstream((body) => {
      if (body.model === "m2") return new Promise<Response>(() => {}); // never resolves
      return chat(body);
    });
    const res = await strategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    expect(up.modelsCalled().filter((m) => m === "m2")).toHaveLength(1); // attempted
    const judgeBody = up.recorded.find((b) => b.model === "j");
    expect(userContents(judgeBody!).join("\n")).not.toContain("ans-m2");
  });

  it("aborts the in-flight upstream call when a panel member times out (frees the slot, H-1)", async () => {
    const fastTimer: TimerFactory = () => {
      let h: ReturnType<typeof setTimeout>;
      const expired = new Promise<void>((resolve) => {
        h = setTimeout(resolve, 5);
      });
      return { expired, cancel: () => clearTimeout(h) };
    };
    let m2Aborted = false;
    const fetchFn: FetchFn = async (input, init) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url.endsWith("/api/show")) return jsonResponse({ capabilities: ["completion"], model_info: {} });
      const body = RecordedBodySchema.parse(JSON.parse(String(init?.body)));
      if (body.model === "m2") {
        // Hang until the caller's (combined) signal aborts — proves the stage
        // timeout actually cancels the in-flight request instead of letting it
        // linger and hold its concurrency-limiter slot.
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            m2Aborted = true;
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      }
      return defaultChat()(body);
    };
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
    const strategy = createFusionStrategy({ timer: fastTimer });
    const res = await strategy.execute(ctx(client, req()));
    expect(res.status).toBe(200); // survivors carried the request
    await new Promise((resolve) => setTimeout(resolve, 0)); // let the abort listener settle
    expect(m2Aborted).toBe(true); // the slow member's call was cancelled, not abandoned
  });

  it("returns 502 when every panel member fails", async () => {
    const up = makeUpstream((body) => {
      if (body.model.startsWith("m")) return jsonResponse({ error: "down" }, 500);
      return jsonResponse({ choices: [{ message: { content: "x" } }] });
    });
    await expect(fusionStrategy.execute(ctx(up.client, req()))).rejects.toMatchObject({
      httpStatus: 502,
    });
  });

  it("gives synth the judge analysis AND the raw panel answers (no artifact loss on judge success)", async () => {
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    const synthBody = up.recorded.find((b) => b.model === "s");
    const ctxText = systemContents(synthBody!).join("\n");
    expect(ctxText).toContain("JUDGE ANALYSIS");
    expect(ctxText).toContain("they agree");
    // Judge SUCCESS must NOT discard the experts' actual content (code, formulas,
    // exact text). The synth synthesizes from the artifacts, guided by the analysis.
    expect(ctxText).toContain("ans-m1");
    expect(ctxText).toContain("ans-m2");
  });

  it("gives the judge the original user request, not just the panel answers (2a)", async () => {
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(
      ctx(up.client, req({ messages: [{ role: "user", content: "CAPITAL-OF-FRANCE-MARKER" }] })),
    );
    expect(res.status).toBe(200);
    const judgeBody = up.recorded.find((b) => b.model === "j");
    const judgeUser = userContents(judgeBody!).join("\n");
    expect(judgeUser).toContain("CAPITAL-OF-FRANCE-MARKER"); // judge can see what was asked
    expect(judgeUser).toContain("EXPERT ANSWERS"); // ...alongside the panel answers
  });

  it("falls back to raw panel answers when the judge returns invalid JSON", async () => {
    const up = makeUpstream(defaultChat(false)); // judge emits non-JSON
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200); // request still succeeds
    const synthBody = up.recorded.find((b) => b.model === "s");
    const ctxText = systemContents(synthBody!).join("\n");
    expect(ctxText).not.toContain("JUDGE ANALYSIS");
    expect(ctxText).toContain("ans-m1"); // synth got the raw panel answers
  });

  it("parses a judge response wrapped in ```json fences (no false raw-panel fallback)", async () => {
    const analysis = { consensus: "they agree", disagreements: [], unique_insights: [], blind_spots: [] };
    const up = makeUpstream((body) => {
      if (body.model === "j") {
        // Thinking models intermittently wrap JSON in fences despite json_object.
        return jsonResponse({
          choices: [{ message: { content: "```json\n" + JSON.stringify(analysis) + "\n```" } }],
        });
      }
      return defaultChat()(body);
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    const synthBody = up.recorded.find((b) => b.model === "s");
    const ctxText = systemContents(synthBody!).join("\n");
    expect(ctxText).toContain("JUDGE ANALYSIS"); // fence stripped -> analysis used
    expect(ctxText).toContain("they agree");
  });

  it("instructs the judge to emit calibrated confidence and fragile_claims", async () => {
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    const judgeBody = up.recorded.find((b) => b.model === "j");
    const judgeSystem = systemContents(judgeBody!).join("\n");
    // The judge must be told to calibrate, not just report consensus — agreement
    // alone is not high confidence when models share a training lineage.
    expect(judgeSystem).toContain("confidence");
    expect(judgeSystem).toContain("fragile_claims");
    expect(judgeSystem).toContain("high");
    expect(judgeSystem).toMatch(/shared.*lineage|training lineage/i);
  });

  it("passes judge confidence + fragile_claims to synth and tells it to hedge them", async () => {
    const analysis = {
      consensus: "they agree",
      disagreements: [],
      unique_insights: [],
      blind_spots: [],
      confidence: "low",
      fragile_claims: ["the redis lua claim from m2"],
    };
    const up = makeUpstream((body) => {
      if (body.model === "j") {
        return jsonResponse({
          choices: [{ message: { content: JSON.stringify(analysis) } }],
        });
      }
      return defaultChat()(body);
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    const synthBody = up.recorded.find((b) => b.model === "s");
    const ctxText = systemContents(synthBody!).join("\n");
    // The calibrated fields survive into the synth context (JSON-serialized)...
    expect(ctxText).toContain("fragile_claims");
    expect(ctxText).toContain("the redis lua claim from m2");
    expect(ctxText).toContain('"low"');
    // ...and the synth is explicitly told to hedge, not assert, fragile claims.
    expect(ctxText).toMatch(/hedge|surface that uncertainty|false certainty/i);
  });

  it("streams synth SSE to the client when stream:true; returns JSON otherwise", async () => {
    const upStream = makeUpstream(defaultChat(true, true));
    const streamed = await fusionStrategy.execute(ctx(upStream.client, req({ stream: true })));
    expect(streamed.headers.get("content-type")).toContain("text/event-stream");
    const text = await streamed.text();
    expect(text).toContain("final");
    expect(text).toContain("[DONE]");
    // Only the synth call carried stream:true.
    const synthBody = upStream.recorded.find((b) => b.model === "s");
    expect(synthBody?.stream).toBe(true);
    const panelStreamed = upStream.recorded.filter((b) => b.model.startsWith("m") && b.stream === true);
    expect(panelStreamed).toHaveLength(3);

    const upJson = makeUpstream(defaultChat(true, false));
    const jsonRes = await fusionStrategy.execute(ctx(upJson.client, req({ stream: false })));
    expect(jsonRes.headers.get("content-type")).toContain("application/json");
    expect(JSON.parse(await jsonRes.text()).choices[0].message.content).toBe("final");
  });

  it("tool_mode bypass: skips panel+judge, one synth call WITH tools", async () => {
    const up = makeUpstream(defaultChat());
    const res = await fusionStrategy.execute(
      ctx(up.client, req({ model: "fusion-bypass", tools: TOOLS }), "fusion-bypass"),
    );
    expect(res.status).toBe(200);
    const called = up.modelsCalled();
    expect(called).toEqual(["s"]); // exactly one call, to synth
    expect(called).not.toContain("m1");
    expect(called).not.toContain("j");
    const synthBody = up.recorded.find((b) => b.model === "s");
    expect(synthBody?.tools).toEqual(TOOLS);
  });

  it("fusion_planning_turn_only: synth-only on a tool-result continuation; full panel on every fresh user turn", async () => {
    // Mid agent-loop: the LATEST message is a tool result -> synth only.
    const upDegraded = makeUpstream(defaultChat());
    const midLoop = req({
      model: "fusion-planning",
      messages: [
        { role: "user", content: "do it" },
        { role: "assistant", content: null },
        { role: "tool", content: "tool result" },
      ],
    });
    await fusionStrategy.execute(ctx(upDegraded.client, midLoop, "fusion-planning"));
    expect(upDegraded.modelsCalled()).toEqual(["s"]);

    // First/planning turn (no tool history) -> full panel.
    const upFull = makeUpstream(defaultChat());
    const planningTurn = req({ model: "fusion-planning", messages: [{ role: "user", content: "do it" }] });
    await fusionStrategy.execute(ctx(upFull.client, planningTurn, "fusion-planning"));
    expect(upFull.modelsCalled()).toContain("m1");
    expect(upFull.modelsCalled()).toContain("j");
    expect(upFull.modelsCalled()).toContain("s");

    // THE FIX: a NEW user instruction deep in a session that ALREADY has older tool
    // messages in history (latest message is the fresh user turn, not a tool result)
    // -> full panel again. The old "any tool message anywhere" check failed this.
    const upNewTurn = makeUpstream(defaultChat());
    const newInstruction = req({
      model: "fusion-planning",
      messages: [
        { role: "user", content: "build X" },
        { role: "assistant", content: null },
        { role: "tool", content: "old tool result from earlier work" },
        { role: "assistant", content: "done with X" },
        { role: "user", content: "now finish the webpage" }, // fresh instruction = latest message
      ],
    });
    await fusionStrategy.execute(ctx(upNewTurn.client, newInstruction, "fusion-planning"));
    expect(upNewTurn.modelsCalled()).toContain("m1");
    expect(upNewTurn.modelsCalled()).toContain("j");
    expect(upNewTurn.modelsCalled()).toContain("s");
  });
});

describe("fusion strategy — vision gate", () => {
  const imageReq = (model: string): ChatCompletionRequest => ({
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "what is this?" },
          { type: "image_url", image_url: { url: "data:image/png;base64,aGVsbG8=" } },
        ],
      },
    ],
  });

  it("rejects an image request when no fusion member is vision-capable (400)", async () => {
    const up = makeUpstream(defaultChat(), () => jsonResponse({ capabilities: ["completion"], model_info: {} }));
    await expect(
      fusionStrategy.execute(ctx(up.client, imageReq("fusion-vision"), "fusion-vision")),
    ).rejects.toMatchObject({ httpStatus: 400 });
  });

  it("proceeds when panel members and synth are vision-capable", async () => {
    const visionShow: ShowHandler = (model) =>
      // vm1, vm2 and synth vs are vision-capable; judge j need not be.
      jsonResponse({
        capabilities: model.startsWith("v") ? ["vision", "completion"] : ["completion"],
        model_info: {},
      });
    const up = makeUpstream(defaultChat(), visionShow);
    const res = await fusionStrategy.execute(ctx(up.client, imageReq("fusion-vision"), "fusion-vision"));
    expect(res.status).toBe(200);
    expect(up.modelsCalled()).toContain("vs"); // synth ran
    expect(up.modelsCalled()).toContain("vm1"); // vision panel ran
  });

  it("synth-only (bypass) image request validates the SYNTH, not the panel (HIGH-3)", async () => {
    // The panel never runs on a synth-only path, so a non-vision panel must NOT
    // block a valid image request whose synth IS vision-capable. (Old code ran the
    // vision gate before the degrade check and 400'd on the non-vision panel.)
    const visionShow: ShowHandler = (model) =>
      jsonResponse({
        capabilities: model === "vs" ? ["vision", "completion"] : ["completion"], // only the synth has vision
        model_info: {},
      });
    const up = makeUpstream(defaultChat(), visionShow);
    const res = await fusionStrategy.execute(
      ctx(up.client, imageReq("fusion-bypass-vision"), "fusion-bypass-vision"),
    );
    expect(res.status).toBe(200);
    expect(up.modelsCalled()).toContain("vs"); // synth ran
    expect(up.modelsCalled()).not.toContain("nv1"); // panel was correctly skipped (bypass)
  });

  it("synth-only image request still rejects when the SYNTH is not vision-capable (400)", async () => {
    const noVisionShow: ShowHandler = () =>
      jsonResponse({ capabilities: ["completion"], model_info: {} }); // nothing is vision-capable
    const up = makeUpstream(defaultChat(), noVisionShow);
    await expect(
      fusionStrategy.execute(ctx(up.client, imageReq("fusion-bypass-vision"), "fusion-bypass-vision")),
    ).rejects.toMatchObject({ httpStatus: 400 });
  });
});

describe("fusion strategy — reasoning→content normalization", () => {
  const validJudge = jsonResponse({
    choices: [{ message: { content: JSON.stringify({ consensus: "ok" }) } }],
  });

  it("panel member answering in `reasoning` (empty content) still reaches the judge", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") return jsonResponse({ choices: [{ message: { content: "final" } }] });
      if (body.model === "m2") {
        // Thinking model: final answer lands in `reasoning`, content is empty.
        return jsonResponse({ choices: [{ message: { content: "", reasoning: "REASONED-ANSWER-m2" } }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    const judgeBody = up.recorded.find((b) => b.model === "j");
    expect(judgeBody).toBeDefined();
    const judgeInput = userContents(judgeBody!).join("\n");
    expect(judgeInput).toContain("REASONED-ANSWER-m2"); // reasoning text fed to the judge
    expect(judgeInput).toContain("ans-m1"); // ordinary content member still present
  });

  it("synth non-stream: promotes reasoning into content when content empty and no tool calls (flag on)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") {
        return jsonResponse({ choices: [{ message: { content: "", reasoning: "SYNTH-REASONING" } }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    const payload = JSON.parse(await res.text());
    expect(payload.choices[0].message.content).toBe("SYNTH-REASONING");
  });

  it("synth non-stream: leaves empty content untouched when promotion disabled per-model (flag off)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") {
        return jsonResponse({ choices: [{ message: { content: "", reasoning: "SYNTH-REASONING" } }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(
      ctx(up.client, req({ model: "fusion-no-promote" }), "fusion-no-promote"),
    );
    expect(res.status).toBe(200);
    const payload = JSON.parse(await res.text());
    expect(payload.choices[0].message.content).toBe(""); // not promoted
  });

  it("synth non-stream: does NOT promote when tool_calls are present (flag on)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") {
        return jsonResponse({
          choices: [
            {
              message: {
                content: "",
                reasoning: "should-not-surface",
                tool_calls: [{ id: "c1", function: { name: "read_file", arguments: "{}" } }],
              },
            },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req({ tools: TOOLS })));
    expect(res.status).toBe(200);
    const payload = JSON.parse(await res.text());
    expect(payload.choices[0].message.content).toBe(""); // tool path: content stays empty
    expect(payload.choices[0].message.tool_calls).toHaveLength(1);
  });

  it("synth stream: re-emits reasoning deltas as content until real content appears (flag on)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") {
        return sseResponse([
          { choices: [{ delta: { reasoning: "thinking-1 " } }] },
          { choices: [{ delta: { reasoning_content: "thinking-2 " } }] },
          { choices: [{ delta: { content: "REAL-ANSWER" } }] },
          // A reasoning fragment AFTER real content must NOT be promoted (latch).
          { choices: [{ delta: { reasoning: "late-thought" } }] },
          { choices: [{ delta: {}, finish_reason: "stop" }] },
        ]);
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req({ stream: true })));
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    // Reasoning before real content surfaces as content; once real content lands,
    // promotion stops — "late-thought" is NOT promoted into content.
    expect(streamedContents(text)).toEqual(["thinking-1 ", "thinking-2 ", "REAL-ANSWER"]);
    // The late reasoning passes through verbatim (proves the latch turned off).
    expect(text).toContain("late-thought");
    expect(text).toContain("[DONE]");
  });

  it("synth stream: a normal content stream passes through unchanged, not duplicated (flag on)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") {
        return sseResponse([
          { choices: [{ delta: { content: "Hello " } }] },
          { choices: [{ delta: { content: "world" } }] },
          { choices: [{ delta: {}, finish_reason: "stop" }] },
        ]);
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req({ stream: true })));
    const text = await res.text();
    expect(streamedContents(text)).toEqual(["Hello ", "world"]);
    // Each fragment appears exactly once — the transform added nothing.
    expect(text.split("Hello ").length - 1).toBe(1);
    expect(text.split("world").length - 1).toBe(1);
  });

  it("synth stream: a tool-call stream is left untouched (flag on)", async () => {
    const up = makeUpstream((body) => {
      if (body.model === "j") return validJudge;
      if (body.model === "s") {
        return sseResponse([
          {
            choices: [
              {
                delta: {
                  tool_calls: [{ index: 0, id: "call_1", function: { name: "read_file", arguments: "{}" } }],
                },
              },
            ],
          },
          { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
        ]);
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req({ stream: true, tools: TOOLS })));
    const text = await res.text();
    expect(text).toContain("tool_calls"); // tool_calls deltas preserved
    expect(text).toContain("read_file");
    expect(text).toContain("finish_reason"); // finish_reason path preserved
    expect(streamedContents(text)).toEqual([]); // nothing promoted into content
  });
});

describe("fusion strategy — synth completeness guard", () => {
  const judgeOk = { choices: [{ message: { content: JSON.stringify({ consensus: "ok" }) } }] };

  it("retries a synth that stopped mid-plan and adopts the completed answer", async () => {
    let synthCalls = 0;
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        const nudged = systemContents(body).some((c) => c.includes("stopped while still planning"));
        if (nudged) {
          return jsonResponse({ choices: [{ message: { content: "FINAL ARTIFACT" }, finish_reason: "stop" }] });
        }
        // Thinking model: deep plan in `reasoning`, empty content, declared done mid-plan.
        return jsonResponse({
          choices: [
            {
              message: { content: "", reasoning: "step 1 ... step 2 ... Let's produce the final answer." },
              finish_reason: "stop",
            },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    const parsed = z
      .object({ choices: z.array(z.object({ message: z.object({ content: z.string() }) })) })
      .parse(await res.json());
    expect(synthCalls).toBe(2);
    expect(parsed.choices[0]?.message.content).toBe("FINAL ARTIFACT");
  });

  it("retries when the synth stops with an empty answer (no content, no reasoning)", async () => {
    let synthCalls = 0;
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        const nudged = systemContents(body).some((c) => c.includes("stopped while still planning"));
        if (nudged) return jsonResponse({ choices: [{ message: { content: "recovered" }, finish_reason: "stop" }] });
        return jsonResponse({ choices: [{ message: { content: "" }, finish_reason: "stop" }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    const parsed = z
      .object({ choices: z.array(z.object({ message: z.object({ content: z.string() }) })) })
      .parse(await res.json());
    expect(synthCalls).toBe(2);
    expect(parsed.choices[0]?.message.content).toBe("recovered");
  });

  it("does NOT retry when the synth stops with tool_calls (a complete final action)", async () => {
    let synthCalls = 0;
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        return jsonResponse({
          choices: [
            {
              message: {
                content: "",
                tool_calls: [{ id: "c1", type: "function", function: { name: "read_file", arguments: "{}" } }],
              },
              finish_reason: "stop",
            },
          ],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req({ tools: TOOLS })));
    await res.text();
    expect(synthCalls).toBe(1);
  });

  it("does NOT retry a complete answer that happens to carry finish_reason:stop", async () => {
    let synthCalls = 0;
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        return jsonResponse({
          choices: [{ message: { content: "here is the complete, real final answer" }, finish_reason: "stop" }],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    await res.text();
    expect(synthCalls).toBe(1);
  });

  it("keeps the original answer when the retry is also incomplete (no infinite loop)", async () => {
    let synthCalls = 0;
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        // Always stops mid-plan, even after the nudge.
        return jsonResponse({
          choices: [{ message: { content: "", reasoning: "still planning... let's write the code." }, finish_reason: "stop" }],
        });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    await res.text();
    expect(synthCalls).toBe(2); // one original + exactly one retry, then give up
  });

  it("does NOT retry a complete content answer that ends on a planning-like phrase", async () => {
    // Regression: a real `content` answer must never be second-guessed, even if its
    // tail matches a planning marker — only reasoning-only answers are suspect.
    let synthCalls = 0;
    const finalText = "Here is the cover letter. Finally, let's write a warm closing.";
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        return jsonResponse({ choices: [{ message: { content: finalText }, finish_reason: "stop" }] });
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    const parsed = z
      .object({ choices: z.array(z.object({ message: z.object({ content: z.string() }) })) })
      .parse(await res.json());
    expect(synthCalls).toBe(1);
    expect(parsed.choices[0]?.message.content).toBe(finalText);
  });

  it("streaming: retries a synth that stalls mid-plan and delivers the recovered tool call", async () => {
    // Same failure mode as the non-stream tests above, but the client asked for
    // `stream: true` (the normal shape for an interactive agent client). Before the
    // fix, streaming synth has no completeness guard at all: the client would just
    // receive the stalled, empty stream and the retry would never fire.
    let synthCalls = 0;
    const up = makeUpstream((body) => {
      if (body.model === "j") return jsonResponse(judgeOk);
      if (body.model === "s") {
        synthCalls += 1;
        const nudged = systemContents(body).some((c) => c.includes("stopped while still planning"));
        if (nudged) {
          expect(body.stream).toBe(false); // the recovery retry is always non-streamed
          return jsonResponse({
            choices: [
              {
                message: {
                  content: "",
                  tool_calls: [
                    { id: "c1", type: "function", function: { name: "read_file", arguments: '{"path":"a.txt"}' } },
                  ],
                },
                finish_reason: "tool_calls",
              },
            ],
          });
        }
        // Thinking model: deep plan in `reasoning`, empty content, no tool_calls,
        // declares itself done mid-plan — streamed, not a single JSON body.
        return sseResponse([
          { choices: [{ delta: { reasoning: "step 1 ... let's write the file." } }] },
          { choices: [{ delta: {}, finish_reason: "stop" }] },
        ]);
      }
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    });
    const res = await fusionStrategy.execute(ctx(up.client, req({ stream: true, tools: TOOLS })));
    const text = await res.text();
    expect(synthCalls).toBe(2);
    expect(text).toContain("read_file");
    expect(text).toContain("tool_calls");
    expect(text).toContain("[DONE]");
    // SSE events are blank-line delimited; the recovered chunk and [DONE] must be
    // separate events, not merged into one (regression check for the framing bug
    // an adversarial review caught: the terminal chunk was emitted with only a
    // single trailing "\n" instead of "\n\n", concatenating it with [DONE]).
    const events = text.trimEnd().split("\n\n");
    const lastDataEvent = events.find((e) => e.includes("read_file"));
    if (lastDataEvent === undefined) throw new Error("no SSE event contained the recovered tool call");
    const payload = JSON.parse(lastDataEvent.replace(/^data:\s*/, ""));
    expect(payload.choices[0].delta.tool_calls[0].function.name).toBe("read_file");
    expect(events.at(-1)).toBe("data: [DONE]");
  });
});

describe("fusion strategy — panel compression tool-pairing", () => {
  // A long agent loop big enough to force compression. The trailing assistant makes
  // the non-system count even, so recentStart (= count - 30) lands on a `tool` result
  // — the orphaning case: without the fix, the recent window opens on a tool whose
  // parent assistant(tool_calls) is dropped, leaving an omission marker before it.
  function longLoop(pairs: number): unknown[] {
    const big = "x".repeat(6000);
    const msgs: unknown[] = [{ role: "user", content: "original task " + big }];
    for (let k = 0; k < pairs; k++) {
      msgs.push({ role: "assistant", content: "", tool_calls: [{ id: `c${k}`, type: "function", function: { name: "f", arguments: "{}" } }] });
      msgs.push({ role: "tool", tool_call_id: `c${k}`, content: "result " + big });
    }
    msgs.push({ role: "assistant", content: "", tool_calls: [{ id: "cT", type: "function", function: { name: "f", arguments: "{}" } }] });
    return msgs;
  }

  function roleOf(m: unknown): string | undefined {
    return typeof m === "object" && m !== null ? (m as Record<string, unknown>).role as string | undefined : undefined;
  }

  function assertNoOrphanTool(out: unknown[]): void {
    for (let i = 0; i < out.length; i++) {
      if (roleOf(out[i]) === "tool") {
        // A tool result must be immediately preceded by the assistant that owns it,
        // never by an omission marker (system) or a user turn.
        expect(roleOf(out[i - 1])).toBe("assistant");
      }
    }
  }

  it("never orphans a tool message when the recent window would start on a tool result", () => {
    for (const pairs of [40, 41, 42]) {
      const input = longLoop(pairs);
      const out = compressPanelMessages(input);
      expect(out.length).toBeLessThan(input.length); // compression actually ran
      assertNoOrphanTool(out);
    }
  });

  it("leaves a short tool-using history untouched and valid", () => {
    const msgs: unknown[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "", tool_calls: [{ id: "c0", type: "function", function: { name: "f", arguments: "{}" } }] },
      { role: "tool", tool_call_id: "c0", content: "ok" },
    ];
    const out = compressPanelMessages(msgs);
    expect(out.length).toBe(3); // under cap -> unchanged length
    assertNoOrphanTool(out);
  });
});

describe("fusion strategy — web grounding (gated on TAVILY_API_KEY + web_search.enabled)", () => {
  const TAVILY = "https://api.tavily.com/search";
  let realFetch: typeof globalThis.fetch;
  let savedKey: string | undefined;

  beforeEach(() => {
    realFetch = globalThis.fetch;
    savedKey = process.env.TAVILY_API_KEY;
  });
  afterEach(() => {
    globalThis.fetch = realFetch;
    if (savedKey === undefined) delete process.env.TAVILY_API_KEY;
    else process.env.TAVILY_API_KEY = savedKey;
    vi.restoreAllMocks();
  });

  function stubTavily(results: { title: string; url: string; content: string }[]): void {
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url === TAVILY) return jsonResponse({ results });
      return new Response(JSON.stringify({ error: `no stub for ${url}` }), { status: 404 });
    }) as typeof globalThis.fetch;
  }

  it("injects a WEB CONTEXT user message into every panel member when key is set", async () => {
    process.env.TAVILY_API_KEY = "tvly-test";
    stubTavily([{ title: "Fresh docs", url: "https://example.com/fresh", content: "the freshest fact" }]);
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(
      ctx(up.client, req({ model: "fusion-web", messages: [{ role: "user", content: "latest redis lua API" }] }), "fusion-web"),
    );
    expect(res.status).toBe(200);
    const panelBodies = up.recorded.filter((b) => b.model.startsWith("m"));
    expect(panelBodies.length).toBeGreaterThan(0);
    for (const b of panelBodies) {
      const user = userContents(b).join("\n");
      expect(user).toContain("WEB CONTEXT");
      expect(user).toContain("the freshest fact");
      expect(user).toContain("CURRENT DATE");
    }
    // Tavily was actually called once (shared single search).
    // (We assert effect, not call count, to stay robust to the no-network mock.)
    expect(panelBodies[0]?.tools).toBeUndefined(); // invariant untouched
  });

  it("stays fully OFF when TAVILY_API_KEY is unset, even if config opts in", async () => {
    delete process.env.TAVILY_API_KEY;
    let tavilyCalled = false;
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url === TAVILY) {
        tavilyCalled = true;
        return jsonResponse({ results: [] });
      }
      return new Response("{}", { status: 404 });
    }) as typeof globalThis.fetch;
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(ctx(up.client, req(), "fusion-web"));
    expect(res.status).toBe(200);
    expect(tavilyCalled).toBe(false); // no key → no search call at all
    for (const b of up.recorded.filter((x) => x.model.startsWith("m"))) {
      expect(systemContents(b).join("\n")).not.toContain("WEB CONTEXT");
      expect(userContents(b).join("\n")).not.toContain("WEB CONTEXT");
    }
  });

  it("stays OFF when web_search is not enabled on the model", async () => {
    process.env.TAVILY_API_KEY = "tvly-test";
    let tavilyCalled = false;
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url === TAVILY) {
        tavilyCalled = true;
        return jsonResponse({ results: [] });
      }
      return new Response("{}", { status: 404 });
    }) as typeof globalThis.fetch;
    // fusion-1 has no web_search block → grounding must not run.
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200);
    expect(tavilyCalled).toBe(false);
  });

  it("degrades gracefully to an ungrounded panel when the search fails", async () => {
    process.env.TAVILY_API_KEY = "tvly-test";
    globalThis.fetch = (async () =>
      jsonResponse({ error: "tavily down" }, 500)) as typeof globalThis.fetch;
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(ctx(up.client, req(), "fusion-web"));
    expect(res.status).toBe(200); // still succeeds, just ungrounded
    for (const b of up.recorded.filter((x) => x.model.startsWith("m"))) {
      expect(systemContents(b).join("\n")).not.toContain("WEB CONTEXT");
      expect(userContents(b).join("\n")).not.toContain("WEB CONTEXT");
    }
  });

  it("skips web grounding when the prompt is already large (size gate)", async () => {
    process.env.TAVILY_API_KEY = "tvly-test";
    let tavilyCalled = false;
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url === TAVILY) {
        tavilyCalled = true;
        return jsonResponse({ results: [{ title: "x", url: "https://y", content: "fresh" }] });
      }
      return new Response("{}", { status: 404 });
    }) as typeof globalThis.fetch;
    // A long agent-loop history: well over the 80k-char default size gate.
    const big = "x".repeat(120000);
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(
      ctx(up.client, req({ model: "fusion-web", messages: [{ role: "user", content: "latest redis " + big }] }), "fusion-web"),
    );
    expect(res.status).toBe(200);
    expect(tavilyCalled).toBe(false); // size gate skipped the search entirely
    for (const b of up.recorded.filter((x) => x.model.startsWith("m"))) {
      expect(systemContents(b).join("\n")).not.toContain("WEB CONTEXT");
      expect(userContents(b).join("\n")).not.toContain("WEB CONTEXT");
    }
  });

  it("preserves the one-tool-call invariant with web grounding AND tools (agent-loop safety)", async () => {
    // The flagship safety property: only the synth may emit a tool call. Web
    // grounding inserts an extra user message and CURRENT DATE into the panel
    // prompt; this must NOT leak tools to the panel or break the synth's tools.
    process.env.TAVILY_API_KEY = "tvly-test";
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url === TAVILY) {
        return jsonResponse({ results: [{ title: "docs", url: "https://x", content: "fresh docs" }] });
      }
      return new Response("{}", { status: 404 });
    }) as typeof globalThis.fetch;
    const up = makeUpstream(defaultChat(true));
    const res = await fusionStrategy.execute(
      ctx(up.client, req({ model: "fusion-web", tools: TOOLS }), "fusion-web"),
    );
    expect(res.status).toBe(200);

    // Web context IS injected (as a user message), so grounding ran with tools present.
    const panelBodies = up.recorded.filter((b) => b.model.startsWith("m"));
    expect(panelBodies.length).toBeGreaterThan(0);
    for (const b of panelBodies) {
      // Invariant: panel never carries the real tools schema / tool_choice.
      expect(b.tools).toBeUndefined();
      expect(b.tool_choice).toBeUndefined();
      // Web context landed in a user turn, the tool list as a system (prose) note.
      expect(userContents(b).join("\n")).toContain("WEB CONTEXT");
      expect(systemContents(b).join("\n")).toContain("read_file");
      // The web user message must not itself look like a tool result/tool call.
      expect(userContents(b).join("\n")).not.toContain("tool_calls");
    }

    // Synth is the ONLY stage that received the real tools schema.
    const synthBody = up.recorded.find((b) => b.model === "s");
    expect(synthBody).toBeDefined();
    expect(synthBody?.tools).toEqual(TOOLS);
  });

  // --- Issue 2: Panel context compression for long agent loops ---------------

  it("compresses panel messages when total content exceeds threshold", async () => {
    // Build a request with > 200k chars of message content to trigger compression.
    // Each tool-result message is ~5000 chars; 50 of them = ~250k chars total.
    const bigContent = "x".repeat(5000);
    const messages: Array<Record<string, unknown>> = [
      { role: "system", content: "You are a coding assistant." },
      { role: "user", content: "Implement the adversarial panel slot feature." }, // original task
    ];
    // Add 50 tool-loop iterations (assistant + tool pairs).
    for (let i = 0; i < 50; i++) {
      messages.push({ role: "assistant", content: `calling tool step ${i}` });
      messages.push({ role: "tool", content: `${bigContent} result-${i}` });
    }
    // Final user instruction.
    messages.push({ role: "user", content: "Now write the tests for this feature." });

    const up = makeUpstream(defaultChat());
    const request: ChatCompletionRequest = {
      model: "fusion-1",
      messages: messages as ChatCompletionRequest["messages"],
    };
    const res = await fusionStrategy.execute(ctx(up.client, request, "fusion-1"));
    expect(res.status).toBe(200);

    // Check that panel members received COMPRESSED messages (fewer than original).
    const panelBodies = up.recorded.filter((b) => b.model === "m1" || b.model === "m2" || b.model === "m3");
    expect(panelBodies.length).toBeGreaterThanOrEqual(2); // at least min_panel_success

    for (const pb of panelBodies) {
      // Panel should have far fewer messages than the original 103.
      expect(pb.messages.length).toBeLessThan(messages.length);
      // Panel should still contain the system prompt.
      const sysMsgs = systemContents(pb);
      expect(sysMsgs.some((s) => s.includes("coding assistant"))).toBe(true);
      // Panel should contain an omission marker.
      expect(sysMsgs.some((s) => s.includes("earlier message"))).toBe(true);
    }
  });
  it("compresses array-based multimodal messages in the panel context", async () => {
    const bigContent = "x".repeat(15000); // Exceeds PANEL_MSG_HEAD + PANEL_MSG_TAIL
    const request: ChatCompletionRequest = {
      model: "fusion-vision",
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: "data:..." } },
            { type: "text", text: `Here is a huge log file:\n${bigContent}` },
          ],
        },
      ],
    };
    // Force compression by making total > 200k chars
    for (let i = 0; i < 20; i++) {
      request.messages!.push({ role: "assistant", content: `step ${i}` });
      request.messages!.push({ role: "tool", content: bigContent });
    }

    // Need a custom capability show function since we are using vision model
    const show = (model: string): Response =>
      jsonResponse({
        capabilities: ["vm1", "vm2", "vs"].includes(model) ? ["completion", "vision"] : ["completion"],
        model_info: {},
      });
    const up = makeUpstream(defaultChat(), show);
    await fusionStrategy.execute(ctx(up.client, request, "fusion-vision"));

    const panelBody = up.recorded.find((b) => b.model === "vm1");
    expect(panelBody).toBeDefined();
    
    const userMsg = panelBody!.messages.find((m: any) => m.role === "user" && Array.isArray(m.content)) as any;
    expect(userMsg).toBeDefined();
    const textPart = userMsg.content.find((p: any) => p.type === "text");
    expect(textPart.text.length).toBeLessThan(9000); // Capped to 8000 + omit marker
    expect(textPart.text).toContain("chars omitted");
  });
});

