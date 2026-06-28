import { z } from "zod";
import type {
  ChatCompletionRequest,
  ChatCompletionResult,
  ChatMessage,
  Strategy,
  StrategyContext,
  UpstreamClient,
} from "../types";
import type { FusionModelConfig } from "../config";
import type { Resilience } from "../concurrency";
import { createResilience } from "../concurrency";
import {
  AllMembersFailedError,
  CapabilityError,
  CircuitOpenError,
  FusionError,
  NativeStreamingNotImplementedError,
} from "../errors";
import { openAiBodyToNativeChat, requestHasImages } from "../vision";
import { extractJsonObject } from "../json";
import { runBineval, DEFAULT_DIMENSIONS, type BinaryEvaluationResult } from "../bineval";
import { buildWebContext, webGroundingEnabled, type WebGroundingConfig } from "../web";
import {
  failureKindForError,
  failureKindForStatus,
  isAvailabilityFailureStatus,
  logUpstreamFailure,
} from "../attribution";
import { withTimeout, realTimer, combineSignals } from "../timeout";
import type { TimerFactory } from "../timeout";
// Re-exported: the fusion strategy's timer-injection API (FusionDeps.timer) is
// typed by TimerFactory, so tests that inject a deterministic timer import it here.
export type { TimerFactory };
import { extractAnswer, promoteReasoningNonStream, makeReasoningPromotionTransform, stripThinkingTags } from "../reasoning";

/**
 * `fusion` strategy — the core algorithm (spec §5.7 / §8.2):
 *
 *   panel (parallel, tools STRIPPED)  ->  judge (JSON analysis)  ->  synth (streams)
 *
 * Tool gate: in `deliberate` mode the panel never receives `tools`/`tool_choice`;
 * if the request carried tools, their names/descriptions are injected as prose
 * context so panel members deliberate about them without emitting a tool call.
 * The synth stage is the ONLY stage that receives the real `tools` schema and
 * the only one that may stream.
 *
 * Degradations:
 *   - `tool_mode: bypass`            -> skip panel+judge, one synth call with tools.
 *   - `fusion_planning_turn_only`    -> degrade to synth-only ONLY when the latest
 *                                       message is a `role:"tool"` result (a mechanical
 *                                       mid-loop continuation). Every fresh user/system
 *                                       instruction — even deep in a long session — runs
 *                                       the full panel.
 *   - judge failure / invalid JSON   -> synth proceeds from the raw panel answers.
 *
 * Vision gate: image requests are routed only through vision-capable panel
 * members and a vision-capable synth; if none qualify -> HTTP 400.
 */

const STREAM_HEADERS_BASE: Record<string, string> = {
  "cache-control": "no-cache",
  connection: "keep-alive",
};

const JudgeAnalysisSchema = z
  .object({
    consensus: z.union([z.string(), z.array(z.string())]).optional(),
    disagreements: z.union([z.string(), z.array(z.string())]).optional(),
    unique_insights: z.union([z.string(), z.array(z.string())]).optional(),
    blind_spots: z.union([z.string(), z.array(z.string())]).optional(),
    hallucination_flags: z.union([z.string(), z.array(z.string())]).optional(),
    // Calibrated confidence in the analysis as a whole. Agreement is only
    // signal when the experts don't share blind spots, so the judge must say
    // how solid the consensus actually is — not just that one exists. This is
    // the "calibrated uncertainty" lever: it lets the synth hedge fragile
    // claims instead of laundering shared priors into false certainty.
    confidence: z.enum(["high", "medium", "low"]).optional(),
    // The fragile subset: claims that are disputed, thinly supported, or rest
    // on a single expert. The synth hedges these rather than presenting them
    // as established fact.
    fragile_claims: z.union([z.string(), z.array(z.string())]).optional(),
  })
  .passthrough();

export type JudgeAnalysis = z.infer<typeof JudgeAnalysisSchema>;

interface PanelAnswer {
  member: string;
  content: string;
}

// --- Strategy factory ------------------------------------------------------

export interface FusionDeps {
  /** Per-stage timeout scheduler (panel members + judge). Default: real timers. */
  timer?: TimerFactory;
}

export function createFusionStrategy(deps: FusionDeps = {}): Strategy {
  const timer = deps.timer ?? realTimer;
  return {
    async execute(ctx: StrategyContext): Promise<Response> {
      if (ctx.modelConfig.strategy !== "fusion") {
        throw new FusionError(
          "fusion strategy invoked with a non-fusion model config",
          500,
          "internal_error",
        );
      }
      return runFusion(ctx, ctx.modelConfig, timer);
    },
  };
}

/** Default fusion strategy (real timers) wired by the router. */
export const fusionStrategy: Strategy = createFusionStrategy();

async function runFusion(
  ctx: StrategyContext,
  cfg: FusionModelConfig,
  timer: TimerFactory,
): Promise<Response> {
  const { request, logger } = ctx;
  const stream = request.stream === true;
  const resilience =
    ctx.resilience ?? createResilience({ maxConcurrency: ctx.config.upstream.max_concurrency });
  const defaults = ctx.config.defaults;
  const hasImages = requestHasImages(request);
  const hasTools = Array.isArray(request.tools) && request.tools.length > 0;
  const native = ctx.config.upstream.api_mode === "native" && hasImages;
  // Effective reasoning->content promotion: per-model override wins over the default.
  const promote = cfg.promote_reasoning_to_content ?? defaults.promote_reasoning_to_content;

  // Degradations that skip panel+judge entirely. Computed BEFORE the vision gate
  // so a synth-only image request is validated against the SYNTH alone — the panel
  // never runs, so requiring vision-capable panel members would wrongly reject it.
  const planningDegrade =
    cfg.fusion_planning_turn_only && latestMessageIsToolResult(request);
  const synthOnly = cfg.tool_mode === "bypass" || planningDegrade;

  // Vision gate (spec §5.11 / §10.1): only run discovery when images are present
  // so the common text path adds no upstream calls. Synth-only validates just the
  // synth; the full path validates the panel and the synth.
  let panelMembers = cfg.panel;
  if (hasImages) {
    if (synthOnly) {
      await assertSynthVision(ctx, cfg);
    } else {
      panelMembers = await applyVisionGate(ctx, cfg);
    }
  }

  if (synthOnly) {
    logger.info(
      { model: request.model, reason: cfg.tool_mode === "bypass" ? "bypass" : "planning_turn_only" },
      "fusion: synth-only path",
    );
    return runSynth(ctx, resilience, cfg.synth, null, [], { stream, hasTools, native, promote, webContext: null });
  }

  // WEB GROUNDING — one optional Tavily search before the panel fans out; the
  // cleaned results are injected as prose context into every panel member. Gated
  // three ways: model opts in via `web_search.enabled`, the TAVILY_API_KEY env
  // var must be set, and this only runs on the full-panel path (the
  // `fusion_planning_turn_only` synth-only degradation above already keeps it
  // off mid-loop). No panel member ever receives real tools, so the
  // one-`tool_calls`-per-step invariant is untouched.
  const webContext = await buildPanelWebContext(ctx, cfg);
  if (webContext !== null) {
    logger.info({ model: request.model, query_chars: webQuery(ctx.request).length }, "fusion: web grounding applied");
  }

  // PANEL — parallel, tools stripped.
  const panelAnswers = await runPanel(ctx, resilience, panelMembers, timer, {
    hasTools,
    native,
    webContext,
    adversarialModel: cfg.adversarial ?? null,
  });
  if (panelAnswers.length < defaults.min_panel_success) {
    throw new AllMembersFailedError(
      `fusion panel produced ${panelAnswers.length} usable answer(s); need >= ${defaults.min_panel_success} for '${request.model}'`,
    );
  }
  logger.info(
    {
      model: request.model,
      panel: panelMembers,
      panel_total: panelMembers.length,
      panel_ok: panelAnswers.length,
      judge: cfg.judge,
      synth: cfg.synth,
    },
    "fusion: panel complete",
  );

  // JUDGE — one structured-JSON call; failure degrades to raw panel answers.
  const analysis = await runJudge(ctx, resilience, cfg.judge, panelAnswers, timer, defaults);

  // SYNTH — final answer, streams when requested, the only stage with real tools.
  const response = await runSynth(ctx, resilience, cfg.synth, analysis, panelAnswers, { stream, hasTools, native, promote, webContext });

  // BINEVAL — optional post-synth quality evaluation. Streaming bodies are consumed by
  // the client, so we can only evaluate non-streaming JSON responses.
  if (!cfg.bineval?.enabled || stream) {
    return response;
  }
  return attachBinevalHeaders(ctx, resilience, cfg, response, timer, defaults);
}

// --- Vision gate -----------------------------------------------------------

async function applyVisionGate(ctx: StrategyContext, cfg: FusionModelConfig): Promise<string[]> {
  const visionPanel: string[] = [];
  for (const member of cfg.panel) {
    const { capability } = await ctx.capabilities.discover(member);
    if (capability.vision) visionPanel.push(member);
  }
  if (visionPanel.length === 0) {
    throw new CapabilityError(
      `fusion model '${ctx.request.model}' received image input but none of its panel members are vision-capable`,
    );
  }
  await assertSynthVision(ctx, cfg);
  return visionPanel;
}

/** Image input requires a vision-capable synth (the synth always runs, in every path). */
async function assertSynthVision(ctx: StrategyContext, cfg: FusionModelConfig): Promise<void> {
  const { capability: synthCap } = await ctx.capabilities.discover(cfg.synth);
  if (!synthCap.vision) {
    throw new CapabilityError(
      `fusion model '${ctx.request.model}' received image input but its synth model '${cfg.synth}' is not vision-capable`,
    );
  }
}

// --- Panel stage -----------------------------------------------------------

type StreamAccumulatorResult = { content: string; toolCalls: unknown[] };

/** Serialise a captured tool call into prose suitable for the deliberation panel. */
function formatToolCall(tc: unknown): string {
  const parsed = z
    .object({
      id: z.string().optional(),
      type: z.string().optional(),
      function: z.object({ name: z.string(), arguments: z.string().optional() }).passthrough().optional(),
    })
    .passthrough()
    .safeParse(tc);
  if (!parsed.success) return JSON.stringify(tc);
  const fn = parsed.data.function;
  if (!fn) return JSON.stringify(tc);
  return `I would call ${fn.name}${fn.arguments ? ` with ${fn.arguments}` : ""}.`;
}

/** Extract any captured tool calls into a single prose string, or "" if none. */
function toolCallsAsText(toolCalls: unknown[]): string {
  if (toolCalls.length === 0) return "";
  return toolCalls.map(formatToolCall).join(" ");
}

/** Accumulates a stream default output while notifying when the first token starts delivering.
 *  Captures delta.content/reasoning/reasoning_content into `content` and delta.tool_calls /
 *  legacy function_call into `toolCalls` (a deliberation-mode member sometimes emits its
 *  decision as a tool_call with empty content — serialising it keeps the panel from dropping). */
async function accumulateStreamAndTrack(
  stream: ReadableStream<Uint8Array>,
  onFirstToken: () => void,
  signal?: AbortSignal,
): Promise<StreamAccumulatorResult> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let accumulatedRaw = "";
  let calledFirstToken = false;
  const toolCalls: unknown[] = [];

  const captureToolCalls = (delta: unknown): void => {
    if (typeof delta !== "object" || delta === null) return;
    const d = delta as Record<string, unknown>;
    if (Array.isArray(d.tool_calls)) {
      for (const tc of d.tool_calls) toolCalls.push(tc);
    }
    if (typeof d.function_call === "object" && d.function_call !== null) {
      toolCalls.push(d.function_call);
    }
  };

  const hasUsableDelta = (delta: unknown): boolean => {
    if (typeof delta !== "object" || delta === null) return false;
    const d = delta as Record<string, unknown>;
    const text = d.content || d.reasoning || d.reasoning_content || "";
    if (typeof text === "string" && text.length > 0) return true;
    if (Array.isArray(d.tool_calls) && d.tool_calls.length > 0) return true;
    if (typeof d.function_call === "object" && d.function_call !== null) return true;
    return false;
  };

  try {
    for (;;) {
      if (signal?.aborted) {
        throw new DOMException("aborted", "AbortError");
      }
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        const chunkStr = decoder.decode(value, { stream: true });
        accumulatedRaw += chunkStr;

        if (accumulatedRaw.length > 0 && !calledFirstToken) {
          const trimmed = accumulatedRaw.trim();
          if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            // Raw JSON response
            calledFirstToken = true;
            onFirstToken();
          } else if (trimmed.includes("data:")) {
            // Check if we received the first real data chunk in SSE
            const lines = trimmed.split("\n");
            for (const line of lines) {
              const tl = line.trim();
              if (tl.startsWith("data:") && tl !== "data: [DONE]") {
                try {
                  const payload = tl.slice("data:".length).trim();
                  const chunk = JSON.parse(payload);
                  const delta = chunk.choices?.[0]?.delta;
                  if (hasUsableDelta(delta)) {
                    calledFirstToken = true;
                    onFirstToken();
                    break;
                  }
                } catch {
                  // ignore
                }
              }
            }
          }
        }
      }
    }
    accumulatedRaw += decoder.decode();
  } finally {
    reader.releaseLock();
  }

  // Parse the final accumulatedRaw string
  const trimmed = accumulatedRaw.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      const msg = parsed.choices?.[0]?.message ?? parsed.message;
      captureToolCalls(msg);
      const content = msg?.content || msg?.reasoning || msg?.reasoning_content || "";
      return { content, toolCalls };
    } catch {
      return { content: "", toolCalls };
    }
  }

  // Parse SSE lines
  let content = "";
  const lines = accumulatedRaw.split("\n");
  for (const line of lines) {
    const tl = line.trim();
    if (tl.startsWith("data:") && tl !== "data: [DONE]") {
      try {
        const payload = tl.slice("data:".length).trim();
        const chunk = JSON.parse(payload);
        const delta = chunk.choices?.[0]?.delta;
        captureToolCalls(delta);
        const text = delta?.content || delta?.reasoning || delta?.reasoning_content || "";
        content += text;
      } catch {
        // ignore
      }
    }
  }
  return { content, toolCalls };
}

async function runPanel(
  ctx: StrategyContext,
  resilience: Resilience,
  members: string[],
  timer: TimerFactory,
  opts: { hasTools: boolean; native: boolean; webContext: string | null; adversarialModel: string | null },
): Promise<PanelAnswer[]> {
  const timeoutMs = ctx.config.defaults.panel_member_timeout_s * 1000;
  const minSuccess = ctx.config.defaults.min_panel_success;

  const answers: PanelAnswer[] = [];
  let completedCount = 0;

  if (members.length === 0) {
    return [];
  }

  const memberControllers = members.map(() => new AbortController());
  const hasStartedDelivering = members.map(() => false);
  const completed = members.map(() => false);

  return new Promise<PanelAnswer[]>((resolve) => {
    const checkFinished = () => {
      const successfulCount = answers.length;
      if (successfulCount >= minSuccess) {
        // Early cancellation: abort only members that have NOT started delivering yet!
        // EXCEPTION: the adversarial member is never early-cancelled — its red-team
        // contribution is the whole point of the slot, and red-team reasoning is
        // typically slower (it must steelman the opposite and hunt for flaws), so it
        // would be cancelled before delivering precisely when it earns its cost.
        for (let i = 0; i < members.length; i++) {
          if (!completed[i] && !hasStartedDelivering[i]) {
            if (opts.adversarialModel !== null && members[i] === opts.adversarialModel) continue;
            memberControllers[i]?.abort();
          }
        }
        resolve(answers);
        return true;
      }
      if (completedCount === members.length) {
        resolve(answers);
        return true;
      }
      return false;
    };

    if (checkFinished()) return;

    members.forEach((member, i) => {
      const controller = memberControllers[i];
      if (!controller) return;
      const combinedSignal = ctx.signal
        ? AbortSignal.any([ctx.signal, controller.signal])
        : controller.signal;

      callPanelMember(
        ctx,
        resilience,
        member,
        timer,
        timeoutMs,
        opts,
        combinedSignal,
        () => {
          hasStartedDelivering[i] = true;
        },
      )
        .then((ans) => {
          completed[i] = true;
          completedCount++;
          if (ans) {
            answers.push(ans);
          }
          checkFinished();
        })
        .catch((err) => {
          completed[i] = true;
          completedCount++;
          if (!controller.signal.aborted) {
            ctx.logger.warn(
              { member, reason: err instanceof Error ? err.message : String(err) },
              "fusion: panel member failed, dropping",
            );
          } else {
            ctx.logger.debug(
              { member },
              "fusion: panel member cancelled early",
            );
          }
          checkFinished();
        });
    });
  });
}

async function callPanelMember(
  ctx: StrategyContext,
  resilience: Resilience,
  member: string,
  timer: TimerFactory,
  timeoutMs: number,
  opts: { hasTools: boolean; native: boolean; webContext: string | null; adversarialModel: string | null },
  signal?: AbortSignal,
  onFirstToken?: () => void,
): Promise<PanelAnswer | null> {
  if (!resilience.breaker.canAttempt(member)) {
    logUpstreamFailure(ctx.logger, { stage: "panel", model: member, kind: "circuit_open", latencyMs: 0 });
    ctx.logger.warn({ member }, "fusion: skip panel member (circuit open)");
    return null;
  }
  const body = buildPanelBody(ctx.request, member, {
    hasTools: opts.hasTools,
    native: opts.native,
    webContext: opts.webContext,
    adversarial: opts.adversarialModel !== null && member === opts.adversarialModel,
  });
  const startedAt = Date.now();
  let result: ChatCompletionResult;
  const abort = new AbortController();
  const useStream = !opts.native;
  try {
    result = await resilience.limiter(() =>
      withTimeout(
        invokeUpstream(ctx.client, body, { stream: useStream, native: opts.native, signal: combineSignals(signal ?? ctx.signal, abort.signal) }),
        timeoutMs,
        timer,
        `panel member '${member}' timed out after ${timeoutMs}ms`,
        () => abort.abort(),
      ),
    );
  } catch (err) {
    if (signal?.aborted) {
      throw err;
    }
    resilience.breaker.recordFailure(member);
    ctx.usage?.recordError(member);
    logUpstreamFailure(ctx.logger, {
      stage: "panel",
      model: member,
      kind: failureKindForError(err),
      latencyMs: Date.now() - startedAt,
      reason: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }
  ctx.usage?.record(member, result);

  if (result.status >= 400) {
    // Drop the answer either way; only availability failures (429/5xx) trip the
    // breaker and get attributed — a 4xx is a request-shape issue, not a sick model.
    if (isAvailabilityFailureStatus(result.status)) {
      resilience.breaker.recordFailure(member);
      logUpstreamFailure(ctx.logger, {
        stage: "panel",
        model: member,
        kind: failureKindForStatus(result.status),
        status: result.status,
        latencyMs: Date.now() - startedAt,
      });
    } else {
      // A 4xx used to drop the member SILENTLY — e.g. a Gemini panel member
      // returning 400 "missing a thought_signature" on foreign tool-call history,
      // which surfaced only as an unexplained 2/3 panel. Always log it.
      ctx.logger.warn(
        {
          stage: "panel",
          model: member,
          status: result.status,
          latencyMs: Date.now() - startedAt,
          reason: result.kind === "json" ? shortErrorReason(result.data) : "unexpected stream error response",
        },
        "fusion: panel member dropped (client error response)",
      );
    }
    return null;
  }

  let content = "";
  let toolCalls: unknown[] = [];
  if (result.kind === "stream" && result.body) {
    const acc = await accumulateStreamAndTrack(result.body, onFirstToken ?? (() => {}), signal);
    content = acc.content;
    toolCalls = acc.toolCalls;
  } else if (result.kind === "json") {
    content = extractAnswer(result.data) ?? "";
    toolCalls = extractToolCalls(result.data);
    if ((content.length > 0 || toolCalls.length > 0) && onFirstToken) {
      onFirstToken();
    }
  } else {
    resilience.breaker.recordFailure(member);
    logUpstreamFailure(ctx.logger, {
      stage: "panel",
      model: member,
      kind: "error",
      latencyMs: Date.now() - startedAt,
      reason: "unexpected non-json/non-stream panel result",
    });
    return null;
  }

  resilience.breaker.recordSuccess(member);
  let cleanedContent = stripThinkingTags(content);
  // A deliberation-mode member can reply with a tool_call (its "decision") and
  // empty content/reasoning — serialise the tool call into prose so the panel
  // keeps the member's contribution instead of dropping it for being empty.
  if (cleanedContent === "" && toolCalls.length > 0) {
    cleanedContent = stripThinkingTags(toolCallsAsText(toolCalls));
    ctx.logger.info(
      { stage: "panel", model: member, toolCalls: toolCalls.length, latencyMs: Date.now() - startedAt },
      "fusion: panel member emitted tool_calls with no content; serialised to prose",
    );
  }
  if (cleanedContent === "") {
    // 200 but no usable text (empty content AND empty reasoning, and no tool call
    // to serialise). Previously a silent drop; log the raw shape so a thin panel
    // is never a mystery.
    ctx.logger.warn(
      {
        stage: "panel",
        model: member,
        latencyMs: Date.now() - startedAt,
        contentLen: content.length,
        toolCalls: toolCalls.length,
      },
      "fusion: panel member returned no usable content (dropped)",
    );
    return null;
  }
  return { member, content: cleanedContent };
}

/** Pull message.tool_calls / function_call out of a non-streamed completion body. */
function extractToolCalls(data: unknown): unknown[] {
  const parsed = z
    .object({
      choices: z.array(z.object({ message: z.unknown().optional() }).passthrough()).optional(),
      message: z.unknown().optional(),
    })
    .passthrough()
    .safeParse(data);
  if (!parsed.success) return [];
  const out: unknown[] = [];
  const pushFrom = (msg: unknown): void => {
    if (typeof msg !== "object" || msg === null) return;
    const m = msg as Record<string, unknown>;
    if (Array.isArray(m.tool_calls)) for (const tc of m.tool_calls) out.push(tc);
    if (typeof m.function_call === "object" && m.function_call !== null) out.push(m.function_call);
  };
  for (const c of parsed.data.choices ?? []) pushFrom(c.message);
  pushFrom(parsed.data.message);
  return out;
}

/** Best-effort short error string from an upstream error body, for logging only. */
const UpstreamErrorBodySchema = z
  .object({ error: z.union([z.string(), z.object({ message: z.string() }).passthrough()]) })
  .passthrough();
function shortErrorReason(data: unknown): string | undefined {
  const parsed = UpstreamErrorBodySchema.safeParse(data);
  if (!parsed.success) return undefined;
  const e = parsed.data.error;
  const msg = typeof e === "string" ? e : e.message;
  return msg.length > 160 ? `${msg.slice(0, 160)}…` : msg;
}

/**
 * Panel context compression. Long agent loops accumulate 200k+ token contexts;
 * sending them verbatim to every panel member overflows their context windows
 * (kimi: 262k, gpt-oss: ~128k) and causes 400s that kill the panel. The panel
 * only needs the system prompt(s), the original task, the active instruction,
 * and recent state to deliberate — just like the router sees a compressed view
 * (renderRequestForRouter), the panel gets a wider but still bounded view.
 *
 * PANEL_MAX_CHARS: if total message content is under this, skip compression
 * (short conversations are sent verbatim — no fidelity loss).
 * PANEL_RECENT_WINDOW: how many recent non-system messages to keep.
 * PANEL_MSG_CAP: max chars per individual message content (head+tail).
 */
const PANEL_MAX_CHARS = 200_000;
const PANEL_RECENT_WINDOW = 30;
const PANEL_MSG_HEAD = 6000;
const PANEL_MSG_TAIL = 2000;

/** Approximate total character count of all message content in the array. */
function approxTotalChars(msgs: unknown[]): number {
  let total = 0;
  for (const m of msgs) {
    if (typeof m !== "object" || m === null) continue;
    const content = (m as Record<string, unknown>).content;
    if (typeof content === "string") {
      total += content.length;
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === "object" && part !== null && "text" in part) {
          const text = (part as Record<string, unknown>).text;
          if (typeof text === "string") total += text.length;
        }
      }
    }
  }
  return total;
}

/** Cap a single message's text content (head + tail with omission marker). */
function capPanelMessageContent(content: unknown): unknown {
  if (typeof content !== "string") return content;
  const max = PANEL_MSG_HEAD + PANEL_MSG_TAIL;
  if (content.length <= max) return content;
  const omitted = content.length - max;
  return `${content.slice(0, PANEL_MSG_HEAD)}\n…[${omitted} chars omitted]…\n${content.slice(-PANEL_MSG_TAIL)}`;
}

/**
 * Compress the panel message array when total content exceeds PANEL_MAX_CHARS.
 * Strategy: keep system messages intact, keep the first non-system message
 * (original task), keep the most recent user instruction that predates the
 * recent window, and keep the last PANEL_RECENT_WINDOW non-system messages.
 * The middle is replaced with an omission marker. Each kept message is also
 * content-capped to prevent a single huge tool result from dominating.
 */
function compressPanelMessages(msgs: unknown[]): unknown[] {
  if (approxTotalChars(msgs) <= PANEL_MAX_CHARS) return msgs;

  // Separate system messages (kept in full) from non-system.
  const systems: Array<{ idx: number; msg: unknown }> = [];
  const nonSystems: Array<{ idx: number; msg: unknown }> = [];
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    const role = typeof m === "object" && m !== null ? (m as Record<string, unknown>).role : undefined;
    if (role === "system") {
      systems.push({ idx: i, msg: m });
    } else {
      nonSystems.push({ idx: i, msg: m });
    }
  }

  // If non-system messages fit in the window, no compression needed.
  if (nonSystems.length <= PANEL_RECENT_WINDOW + 1) {
    // Just cap individual messages.
    return msgs.map((m) => {
      if (typeof m !== "object" || m === null) return m;
      const rec = m as Record<string, unknown>;
      return { ...rec, content: capPanelMessageContent(rec.content) };
    });
  }

  // Build the set of non-system indices to keep.
  const keep = new Set<number>();
  keep.add(0); // first non-system message = original task
  const recentStart = nonSystems.length - PANEL_RECENT_WINDOW;
  for (let i = recentStart; i < nonSystems.length; i++) keep.add(i);
  // Keep the most recent user instruction before the window.
  for (let i = recentStart - 1; i > 0; i--) {
    const m = nonSystems[i]?.msg;
    if (typeof m === "object" && m !== null && (m as Record<string, unknown>).role === "user") {
      keep.add(i);
      break;
    }
  }

  // Assemble compressed output: all systems first (in order), then non-systems
  // with gaps replaced by omission markers.
  const result: unknown[] = [];
  // System messages at their original positions relative to non-systems is complex;
  // instead, put all system messages first, then the compressed non-system stream.
  // This is safe because panel system prompts are position-independent (they frame
  // the deliberation, not interleave with the conversation).
  for (const s of systems) {
    const rec = s.msg as Record<string, unknown>;
    result.push({ ...rec, content: capPanelMessageContent(rec.content) });
  }

  let prev = -1;
  for (const i of [...keep].sort((a, b) => a - b)) {
    const gap = i - prev - 1;
    if (prev >= 0 && gap > 0) {
      result.push({
        role: "system",
        content: `…[${gap} earlier message${gap === 1 ? "" : "s"} omitted for context window management]…`,
      });
    }
    const m = nonSystems[i]?.msg;
    if (m && typeof m === "object") {
      const rec = m as Record<string, unknown>;
      result.push({ ...rec, content: capPanelMessageContent(rec.content) });
    }
    prev = i;
  }

  return result;
}

function buildPanelBody(
  request: ChatCompletionRequest,
  member: string,
  opts: { hasTools: boolean; native: boolean; webContext: string | null; adversarial: boolean },
): Record<string, unknown> {
  // Strip tools/tool_choice/stream/model so the panel deliberates rather than
  // executes. Also strip request-level OUTPUT controls (response_format,
  // max_tokens, temperature, top_p, …) — those constrain the final answer the
  // client wants, not the panel's deliberation; leaking them forced some
  // models into JSON/tool-call shapes that produced empty prose.
  const {
    tools,
    tool_choice: _toolChoice,
    stream: _stream,
    model: _model,
    messages,
    response_format: _rf,
    max_tokens: _maxTokens,
    max_completion_tokens: _maxCompletionTokens,
    temperature: _temperature,
    top_p: _topP,
    top_k: _topK,
    frequency_penalty: _freq,
    presence_penalty: _pres,
    logit_bias: _logitBias,
    seed: _seed,
    stop: _stop,
    n: _n,
    user: _user,
    stream_options: _streamOptions,
    ...rest
  } = request;
  // Compress the message history to fit within panel member context windows.
  // Long agent loops (50+ tool calls) accumulate 300k+ chars; without compression,
  // kimi (262k) and gpt-oss (~128k) return 400 "prompt too long" and the panel
  // fails with < min_panel_success answers → 502. The compression keeps the full
  // deliberation-relevant context (system, original task, recent state) while
  // trimming the mechanical middle.
  const rawMsgs: unknown[] = Array.isArray(messages) ? [...messages] : [];
  const msgs: unknown[] = compressPanelMessages(rawMsgs);
  if (opts.webContext !== null) {
    // Inject as a `user` message directly before the latest user instruction,
    // not as `system`: some panel members (kimi-k2.7-code) ignore live facts
    // placed in a system role and refuse on a stale training cutoff, while the
    // same facts in a user turn make them answer. (glm/gpt-oss use either.)
    insertBeforeLastUser(msgs, { role: "user", content: opts.webContext });
  }
  if (opts.adversarial) {
    // This panel member runs with a contrarian mandate: find the flaw, not the
    // answer. It still deliberates in prose and gets no tools — the one-tool-call
    // invariant is untouched. Its objections flow into the judge's disagreements
    // and blind_spots.
    msgs.push({ role: "system", content: ADVERSARIAL_PROMPT });
  }
  if (opts.hasTools) {
    msgs.push({ role: "system", content: toolsPrompt(tools) });
  }
  const body: Record<string, unknown> = { ...rest, model: member, messages: msgs, stream: false };
  return opts.native ? openAiBodyToNativeChat(body) : body;
}

// --- Judge stage -----------------------------------------------------------

const JUDGE_SYSTEM_PROMPT =
  "You are an impartial judge. You are given the user's ORIGINAL REQUEST followed by several independent " +
  "expert answers to it. Assess the answers AGAINST THE REQUEST and respond with ONLY a JSON object with these keys: " +
  '"consensus" (where the experts agree), "disagreements" (where they conflict — and, where the request makes it ' +
  'determinable, which side is correct and why), "unique_insights" (correct, useful points raised by only one expert), ' +
  '"blind_spots" (anything the request needs that none of them addressed), ' +
  'and "hallucination_flags" (any claims, function/API names, library versions, or facts stated confidently by one ' +
  "expert but absent from or contradicted by the others — these are likely fabricated). " +
  "Cross-reference the experts against each other: if only one expert mentions a specific API, function, " +
  "command, or factual claim and the others do not corroborate it, flag it as suspect. " +
  "Judge factual correctness and how well each answer actually serves the request; do not reward verbosity. " +
  "Also produce two extra keys. \"confidence\": your overall confidence in the analysis — \"high\" ONLY when " +
  "multiple independent model lineages agree on concrete, verifiable facts with no unresolved contradictions; " +
  "\"medium\" when the broad direction is corroborated but some specifics are disputed or unverified; " +
  "\"low\" when experts disagree on the core question, reasoning is speculative, or evidence is thin. Agreement " +
  "alone is NOT high confidence — models that share a training lineage can agree on the same wrong thing. " +
  "\"fragile_claims\": the specific claims (API names, versions, numbers, causal assertions) that are disputed, " +
  "rest on a single expert, or are otherwise thin — the ones a careful answer must hedge rather than assert. " +
  "Consistency rule: if you list ANY hallucination_flags or fragile_claims, then \"confidence\" MUST be \"medium\" " +
  "or \"low\" — never \"high\". High confidence is reserved for answers whose substance is fully corroborated across " +
  "independent experts with nothing fragile remaining. Each value may be a string or an array of strings. " +
  "Output JSON only — no prose, no code fences.";

async function runJudge(
  ctx: StrategyContext,
  resilience: Resilience,
  judge: string,
  panelAnswers: PanelAnswer[],
  timer: TimerFactory,
  defaults: { judge_timeout_s: number },
): Promise<JudgeAnalysis | null> {
  if (!resilience.breaker.canAttempt(judge)) {
    logUpstreamFailure(ctx.logger, { stage: "judge", model: judge, kind: "circuit_open", latencyMs: 0 });
    ctx.logger.warn({ judge }, "fusion: judge skipped (circuit open); using raw panel answers");
    return null;
  }
  const body: Record<string, unknown> = {
    model: judge,
    temperature: 0,
    response_format: { type: "json_object" },
    stream: false,
    messages: [
      { role: "system", content: JUDGE_SYSTEM_PROMPT },
      {
        role: "user",
        content:
          "ORIGINAL USER REQUEST:\n" +
          renderRequestForJudge(ctx.request) +
          "\n\nEXPERT ANSWERS:\n" +
          renderPanelForJudge(panelAnswers),
      },
    ],
  };
  const timeoutMs = defaults.judge_timeout_s * 1000;
  const startedAt = Date.now();
  let result: ChatCompletionResult;
  const abort = new AbortController();
  try {
    result = await resilience.limiter(() =>
      withTimeout(
        ctx.client.chatCompletions(body, { stream: false, signal: combineSignals(ctx.signal, abort.signal) }),
        timeoutMs,
        timer,
        `judge '${judge}' timed out after ${timeoutMs}ms`,
        () => abort.abort(),
      ),
    );
  } catch (err) {
    resilience.breaker.recordFailure(judge);
    ctx.usage?.recordError(judge);
    logUpstreamFailure(ctx.logger, {
      stage: "judge",
      model: judge,
      kind: failureKindForError(err),
      latencyMs: Date.now() - startedAt,
      reason: err instanceof Error ? err.message : String(err),
    });
    ctx.logger.warn(
      { judge, reason: err instanceof Error ? err.message : String(err) },
      "fusion: judge call failed; falling back to raw panel answers",
    );
    return null;
  }
  ctx.usage?.record(judge, result);
  if (result.kind !== "json" || result.status >= 400) {
    // Availability failures (non-json / 429 / 5xx) trip the breaker + get
    // attributed; a 4xx still degrades to raw panel answers without it.
    if (result.kind !== "json" || isAvailabilityFailureStatus(result.status)) {
      resilience.breaker.recordFailure(judge);
      logUpstreamFailure(ctx.logger, {
        stage: "judge",
        model: judge,
        kind: result.kind !== "json" ? "error" : failureKindForStatus(result.status),
        ...(result.kind === "json" ? { status: result.status } : {}),
        latencyMs: Date.now() - startedAt,
      });
    }
    ctx.logger.warn(
      { judge, status: result.kind === "json" ? result.status : undefined },
      "fusion: judge non-OK; raw panel fallback",
    );
    return null;
  }
  resilience.breaker.recordSuccess(judge);
  const content = extractAnswer(result.data);
  const analysis = parseJudgeAnalysis(content);
  if (analysis === null) {
    ctx.logger.warn({ judge }, "fusion: judge returned unparseable/invalid JSON; raw panel fallback");
  }
  return analysis;
}

function parseJudgeAnalysis(content: string | null): JudgeAnalysis | null {
  if (content === null) return null;
  // Models wrap their JSON in ```json fences or surrounding prose intermittently
  // (even with response_format: json_object) — extract the balanced object first
  // so one stray fence does not waste the whole judge stage.
  const jsonText = extractJsonObject(content);
  if (jsonText === null) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(jsonText);
  } catch {
    return null;
  }
  const parsed = JudgeAnalysisSchema.safeParse(raw);
  return parsed.success ? parsed.data : null;
}

// --- Synth stage -----------------------------------------------------------

async function runSynth(
  ctx: StrategyContext,
  resilience: Resilience,
  synth: string,
  analysis: JudgeAnalysis | null,
  panelAnswers: PanelAnswer[],
  opts: { stream: boolean; hasTools: boolean; native: boolean; promote: boolean; webContext: string | null },
): Promise<Response> {
  if (!resilience.breaker.canAttempt(synth)) {
    logUpstreamFailure(ctx.logger, { stage: "synth", model: synth, kind: "circuit_open", latencyMs: 0 });
    throw new CircuitOpenError(`circuit breaker open for fusion synth model '${synth}'`);
  }
  const body = buildSynthBody(ctx.request, synth, analysis, panelAnswers, opts);
  const startedAt = Date.now();
  let result: ChatCompletionResult;
  try {
    result = await resilience.limiter(() =>
      invokeUpstream(ctx.client, body, { stream: opts.stream, native: opts.native, signal: ctx.signal }),
    );
  } catch (err) {
    resilience.breaker.recordFailure(synth);
    ctx.usage?.recordError(synth);
    logUpstreamFailure(ctx.logger, {
      stage: "synth",
      model: synth,
      kind: failureKindForError(err),
      latencyMs: Date.now() - startedAt,
      reason: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }
  ctx.usage?.record(synth, result);
  if (result.status < 400) resilience.breaker.recordSuccess(synth);
  // 4xx (non-429) passes through to the client without tripping the breaker.
  else if (isAvailabilityFailureStatus(result.status)) {
    resilience.breaker.recordFailure(synth);
    logUpstreamFailure(ctx.logger, {
      stage: "synth",
      model: synth,
      kind: failureKindForStatus(result.status),
      status: result.status,
      latencyMs: Date.now() - startedAt,
    });
  }

  if (result.kind === "stream") {
    const headers: Record<string, string> = {
      ...STREAM_HEADERS_BASE,
      "content-type": result.contentType ?? "text/event-stream",
    };
    // Streaming reasoning->content promotion (the body is a successful upstream
    // stream; ollama only returns `kind:"stream"` when res.ok).
    const streamBody =
      opts.promote && result.body !== null
        ? result.body.pipeThrough(makeReasoningPromotionTransform())
        : result.body;
    return new Response(streamBody, { status: result.status, headers });
  }
  const data =
    opts.promote && result.status < 400 ? promoteReasoningNonStream(result.data) : result.data;
  return new Response(JSON.stringify(data ?? null), {
    status: result.status,
    headers: { "content-type": "application/json" },
  });
}

async function attachBinevalHeaders(
  ctx: StrategyContext,
  resilience: Resilience,
  cfg: FusionModelConfig,
  response: Response,
  timer: TimerFactory,
  defaults: { judge_timeout_s: number },
): Promise<Response> {
  // Skip evaluation on error responses — the synth failed, there is nothing to score.
  if (response.status >= 400) return response;
  const bineval = cfg.bineval;
  if (!bineval) return response;
  const bodyText = await response.text();
  const headers = new Headers(response.headers);
  let result: BinaryEvaluationResult | null = null;

  try {
    const data = JSON.parse(bodyText);
    const outputText = extractAnswer(data) ?? "";
    if (outputText.length > 0) {
      const requestText = renderRequestForJudge(ctx.request);
      const model = bineval.model ?? cfg.judge;
      const timeoutMs = (bineval.timeout_s ?? defaults.judge_timeout_s) * 1000;
      const questions = bineval.dimensions ?? DEFAULT_DIMENSIONS;
      result = await runBineval(
        ctx,
        resilience,
        model,
        requestText,
        outputText,
        questions,
        timer,
        timeoutMs,
      );
    }
  } catch {
    // Non-JSON body: return it unchanged.
  }

  if (result) {
    headers.set("X-Fusion-Bineval-Score", result.overall.toFixed(3));
    headers.set("X-Fusion-Bineval-Dimensions", JSON.stringify(result.dimensions));
    ctx.logger.info({ score: result.overall, dimensions: result.dimensions }, "fusion: bineval complete");
    if (result.overall < bineval.threshold) {
      headers.set("X-Fusion-Bineval-Low-Score", "true");
      ctx.logger.warn(
        { score: result.overall, threshold: bineval.threshold },
        "fusion: bineval score below threshold",
      );
    }
  }

  return new Response(bodyText, { status: response.status, headers });
}

function buildSynthBody(
  request: ChatCompletionRequest,
  synth: string,
  analysis: JudgeAnalysis | null,
  panelAnswers: PanelAnswer[],
  opts: { stream: boolean; hasTools: boolean; native: boolean; webContext: string | null },
): Record<string, unknown> {
  // Synth keeps the real tools (if any) and the original messages; we append a
  // synthesis-context system message only on the full fusion path.
  const { stream: _stream, model: _model, messages, ...rest } = request;
  const msgs: unknown[] = Array.isArray(messages) ? [...messages] : [];
  // Live web context as a `user` turn adjacent to the question (same reason as the
  // panel injection above): a synth with a hard cutoff ignores fresh facts in a
  // system role but uses them in a user turn.
  if (opts.webContext !== null) {
    insertBeforeLastUser(msgs, { role: "user", content: opts.webContext });
  }
  const context = buildSynthContext(analysis, panelAnswers);
  if (context !== null) {
    msgs.push({ role: "system", content: context });
  }
  const body: Record<string, unknown> = { ...rest, model: synth, messages: msgs, stream: opts.stream };
  // `rest` already carries `tools`/`tool_choice` verbatim when present — synth is
  // the only stage that receives them, so no stripping here.
  return opts.native ? openAiBodyToNativeChat(body) : body;
}

/**
 * Build the synthesis context message. `null` on the synth-only path (no panel
 * ran). Otherwise the synth ALWAYS receives the raw panel answers, so it never
 * loses the experts' actual artifacts (code, formulas, exact text); the
 * structured judge analysis, when available, is layered on top as adjudication
 * guidance rather than replacing the answers. When the judge failed, the synth
 * is told to reconcile conflicts itself.
 */
function buildSynthContext(
  analysis: JudgeAnalysis | null,
  panelAnswers: PanelAnswer[],
): string | null {
  if (panelAnswers.length === 0) return null;
  const experts = renderPanelForJudge(panelAnswers);
  if (analysis !== null) {
    return (
      "A panel of expert models answered the user's request, and an impartial judge produced a structured " +
      "analysis of their answers. Write the single best final answer: take the actual content (code, formulas, " +
      "exact text) from the expert answers, and use the judge analysis to resolve disagreements, cover blind " +
      "spots, and weight the consensus. Do not drop detail that only one expert provided unless it is wrong. " +
      "IMPORTANT: if the judge flagged hallucination_flags, treat those items as suspect — omit or explicitly " +
      "caveat them rather than presenting fabricated information as fact. When experts disagree and you cannot " +
      "determine which side is correct, say so honestly instead of inventing an answer. " +
      "If the judge set \"confidence\" to \"low\" or listed \"fragile_claims\", do not present those claims as " +
      "settled fact — hedge them (\"may\", \"one expert held\", \"unverified\") or omit them, and where the " +
      "question is genuinely uncertain, surface that uncertainty in the answer rather than collapsing it into " +
      "false certainty.\n\n" +
      "JUDGE ANALYSIS (JSON):\n" +
      JSON.stringify(analysis) +
      "\n\nEXPERT ANSWERS:\n" +
      experts
    );
  }
  return (
    "A panel of expert models answered the user's request (a structured judge analysis was unavailable). " +
    "Synthesize the single best final answer from these expert answers; where they disagree, reconcile the " +
    "conflict explicitly and prefer the better-supported answer over the more verbose one.\n\nEXPERT ANSWERS:\n" +
    experts
  );
}

function renderPanelForJudge(panelAnswers: PanelAnswer[]): string {
  return panelAnswers
    .map((a, i) => `--- Expert ${i + 1} (${a.member}) ---\n${a.content}`)
    .join("\n\n");
}

/**
 * Render the user's instruction for the judge: the `user`/`system` messages only.
 * The judge needs to know WHAT WAS ASKED to adjudicate factual conflicts, but the
 * assistant/tool history is what the panel already digested into its answers, so
 * re-sending it would just bloat the judge call (often the bulk of a large context).
 */
function renderRequestForJudge(request: ChatCompletionRequest): string {
  const messages: ChatMessage[] = Array.isArray(request.messages) ? request.messages : [];
  const lines: string[] = [];
  for (const m of messages) {
    const role = typeof m.role === "string" ? m.role : "user";
    if (role !== "user" && role !== "system") continue;
    const text =
      typeof m.content === "string" ? m.content : Array.isArray(m.content) ? "[multimodal content]" : "";
    if (text.length > 0) lines.push(`${role}: ${text}`);
  }
  return lines.join("\n");
}

// --- Shared helpers --------------------------------------------------------

/** Dispatch to the native or OpenAI-compat backend; native streaming is deferred. */
function invokeUpstream(
  client: UpstreamClient,
  body: Record<string, unknown>,
  opts: { stream: boolean; native: boolean; signal?: AbortSignal },
): Promise<ChatCompletionResult> {
  if (opts.native) {
    if (opts.stream) {
      throw new NativeStreamingNotImplementedError(
        "native /api/chat streaming is not yet wired; use api_mode openai/auto for streaming image requests",
      );
    }
    return client.chatNative(body, { stream: false, signal: opts.signal });
  }
  return client.chatCompletions(body, { stream: opts.stream, signal: opts.signal });
}

/**
 * System prompt for the adversarial panel member. Its mandate is the OPPOSITE of
 * the other members: find the flaw, not the answer. This is role-based
 * decorrelation on top of the existing lineage-based decorrelation — it forces
 * the panel to contain a genuine dissenting voice, which the judge then folds
 * into its `disagreements` and `blind_spots`. It must NOT invent problems: a
 * clean "I cannot find a flaw" is more useful than a fabricated one, because a
 * fake objection would itself be a hallucination the judge would have to untangle.
 */
const ADVERSARIAL_PROMPT =
  "You are this panel's adversarial reviewer. Your job is NOT to agree with the " +
  "consensus — it is to find what is wrong, fragile, or missing before the group " +
  "commits to an answer. Specifically: steelman the strongest case AGAINST the " +
  "obvious answer; surface hidden assumptions, edge cases, race conditions, " +
  "unsafe defaults, and unverified or thinly-sourced claims; name the scenario " +
  "where the proposed approach breaks. Be concrete — point at the specific " +
  "function, step, number, or claim that is suspect, and say why. You still " +
  "answer in prose and do NOT emit tool calls. Crucially: if you genuinely cannot " +
  "find a real flaw, say so plainly rather than inventing one — a fabricated " +
  "objection is just another hallucination.";

/** Render the request's tool schema as prose for the (tool-stripped) panel. */
function toolsPrompt(tools: unknown): string {
  const lines: string[] = [];
  if (Array.isArray(tools)) {
    for (const tool of tools) {
      const fn = ToolSchema.safeParse(tool);
      if (fn.success) {
        const desc = fn.data.function.description;
        lines.push(`- ${fn.data.function.name}${desc ? `: ${desc}` : ""}`);
      }
    }
  }
  const listed = lines.length > 0 ? lines.join("\n") : "(no tool details available)";
  return (
    "Available tools (for your awareness only — you CANNOT call them in this step):\n" +
    listed +
    "\n\nYou are in DELIBERATION mode. Respond ONLY with reasoning prose: describe the best " +
    "approach and which tool(s) you would use and why. There are no tools available to call " +
    "here, so do NOT emit tool_calls, function_call, or JSON — output plain prose text only."
  );
}

const ToolSchema = z
  .object({
    function: z.object({ name: z.string(), description: z.string().optional() }).passthrough(),
  })
  .passthrough();

/** True when the conversation already contains a `role:"tool"` message. */
/**
 * True when the LATEST message is a tool result — i.e. the agent is mid-loop,
 * mechanically continuing ("read this tool output, pick the next call"). A fresh
 * user/system instruction as the latest message is a PLANNING turn, not this —
 * even deep in a long session that already contains older tool messages, so a new
 * task / refinement / "now build module X" still earns the full panel.
 *
 * (The previous check — "any tool message ANYWHERE in history" — stayed true
 * forever after the first tool call, so the panel never ran again for the rest of
 * a multi-day session, collapsing fusion to a single model.)
 */
function latestMessageIsToolResult(request: ChatCompletionRequest): boolean {
  const messages = request.messages;
  if (!messages || messages.length === 0) return false;
  return messages[messages.length - 1]?.role === "tool";
}

/**
 * Insert `msg` directly before the last `user`-role message in `msgs` (in place).
 * If there is no user message, append. Used to place live web context in a user
 * turn adjacent to the question — some models only act on facts that sit in a
 * user message, ignoring the same facts given as a system message.
 */
function insertBeforeLastUser(msgs: unknown[], msg: { role: "user"; content: string }): void {
  let idx = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m && typeof m === "object" && (m as { role?: string }).role === "user") {
      idx = i;
      break;
    }
  }
  if (idx < 0) msgs.push(msg);
  else msgs.splice(idx, 0, msg);
}

/**
 * Rough character size of the whole conversation — a cheap proxy for token count
 * (≈4 chars/token) used only to decide whether web grounding would risk pushing a
 * smaller-context panel member over its limit. Counts string content and text parts;
 * images contribute a fixed allowance so a vision turn is never mistaken for "small".
 */
function approxPromptChars(request: ChatCompletionRequest): number {
  const messages = request.messages;
  if (!messages) return 0;
  let total = 0;
  for (const m of messages) {
    if (m === undefined) continue;
    const c = m.content;
    if (typeof c === "string") {
      total += c.length;
    } else if (Array.isArray(c)) {
      for (const part of c) {
        if (part && typeof part === "object") {
          const t = (part as { type?: string }).type;
          if (t === "text") total += String((part as { text?: unknown }).text ?? "").length;
          else if (t === "image_url") total += 2000;
        }
      }
    }
  }
  return total;
}

/**
 * Extract the latest user instruction as a plain-text search query. Walks from
 * the end of the conversation to the most recent `user` message, concatenates
 * its text parts, and truncates to a Tavily-friendly length. Returns "" when no
 * usable user text exists (the caller then skips grounding).
 */
function webQuery(request: ChatCompletionRequest): string {
  const messages = request.messages;
  if (!messages) return "";
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m === undefined || m.role !== "user") continue;
    const c = m.content;
    if (typeof c === "string") return c.slice(0, 512).trim();
    if (Array.isArray(c)) {
      const text = c
        .filter((p) => p && typeof p === "object" && (p as { type?: string }).type === "text")
        .map((p) => String((p as { text?: unknown }).text ?? ""))
        .join(" ");
      if (text.trim()) return text.slice(0, 512).trim();
    }
  }
  return "";
}

/**
 * Resolve optional web grounding for a fusion call. Returns the formatted
 * context to inject into panel members, or `null` when the feature is gated off
 * (model did not opt in, no TAVILY_API_KEY, no usable query, or the search
 * failed/returned nothing). Never throws — a grounding failure degrades
 * silently to an ungrounded panel.
 */
async function buildPanelWebContext(
  ctx: StrategyContext,
  cfg: FusionModelConfig,
): Promise<string | null> {
  const ws = cfg.web_search;
  if (!ws?.enabled) return null;
  const apiKey = process.env.TAVILY_API_KEY;
  if (!webGroundingEnabled(apiKey)) {
    ctx.logger.warn(
      { model: ctx.request.model },
      "fusion: web_search enabled in config but TAVILY_API_KEY is unset — grounding disabled",
    );
    return null;
  }
  const query = webQuery(ctx.request);
  if (query.length === 0) return null;
  // Size gate: web context is layered on top of the panel prompt, which in a long
  // agent loop already carries a large tool history. A smaller-context panel member
  // (gpt-oss:120b at 128k) can be pushed over its limit by the added context and
  // get dropped with a 400 — losing the very diversity the panel exists for. So when
  // the request is already large, skip grounding rather than risk overflowing a
  // panel member. Short planning turns (where freshness matters most) still ground.
  const promptChars = approxPromptChars(ctx.request);
  const maxPromptChars = ws.max_prompt_chars;
  if (promptChars > maxPromptChars) {
    ctx.logger.info(
      { model: ctx.request.model, prompt_chars: promptChars, max_prompt_chars: maxPromptChars },
      "fusion: web grounding skipped (prompt already large; would risk overflowing a panel member)",
    );
    return null;
  }
  const gcfg: WebGroundingConfig = {
    apiKey: apiKey as string,
    maxResults: ws.max_results,
    timeoutMs: ws.timeout_s * 1000,
    maxContextChars: ws.max_context_chars,
  };
  try {
    const context = await buildWebContext(query, gcfg, ctx.signal);
    if (context === null) return null;
    // Lead with the current date so models with a stale training cutoff do not
    // treat post-cutoff web results as "the future" and refuse. Models that don't
    // know today's date (e.g. kimi-k2.7-code) otherwise answer "that date is in
    // the future" even with the live facts in front of them. This is the single
    // line that makes post-cutoff research grounding actually usable.
    const today = new Date().toISOString().slice(0, 10);
    return `CURRENT DATE: ${today}\n\n${context}`;
  } catch (err) {
    ctx.logger.warn(
      { reason: err instanceof Error ? err.message : String(err) },
      "fusion: web grounding failed; proceeding ungrounded",
    );
    return null;
  }
}
