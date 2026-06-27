import { describe, it, expect } from "vitest";
import { z } from "zod";
import { createFusionStrategy, fusionStrategy } from "../src/strategies/fusion";
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
    "fusion-no-promote": {
      strategy: "fusion",
      panel: ["m1", "m2", "m3"],
      judge: "j",
      synth: "s",
      promote_reasoning_to_content: false,
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

type ChatHandler = (body: RecordedBody) => Response | Promise<Response>;
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
      return chat(body);
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

  it("falls back to raw panel answers when the judge returns invalid JSON", async () => {
    const up = makeUpstream(defaultChat(false)); // judge emits non-JSON
    const res = await fusionStrategy.execute(ctx(up.client, req()));
    expect(res.status).toBe(200); // request still succeeds
    const synthBody = up.recorded.find((b) => b.model === "s");
    const ctxText = systemContents(synthBody!).join("\n");
    expect(ctxText).not.toContain("JUDGE ANALYSIS");
    expect(ctxText).toContain("ans-m1"); // synth got the raw panel answers
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
    expect(panelStreamed).toHaveLength(0);

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

  it("fusion_planning_turn_only: synth-only when a prior tool message exists; full fusion otherwise", async () => {
    // Mid agent-loop (a role:"tool" message present) -> synth only.
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

    // First/planning turn (no tool message) -> full fusion.
    const upFull = makeUpstream(defaultChat());
    const planningTurn = req({ model: "fusion-planning", messages: [{ role: "user", content: "do it" }] });
    await fusionStrategy.execute(ctx(upFull.client, planningTurn, "fusion-planning"));
    const called = upFull.modelsCalled();
    expect(called).toContain("m1");
    expect(called).toContain("j");
    expect(called).toContain("s");
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
