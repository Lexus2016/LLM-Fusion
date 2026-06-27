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
import { extractAnswer, promoteReasoningNonStream, makeReasoningPromotionTransform } from "../reasoning";

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
    return runSynth(ctx, resilience, cfg.synth, null, [], { stream, hasTools, native, promote });
  }

  // PANEL — parallel, tools stripped.
  const panelAnswers = await runPanel(ctx, resilience, panelMembers, timer, { hasTools, native });
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
  return runSynth(ctx, resilience, cfg.synth, analysis, panelAnswers, { stream, hasTools, native, promote });
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

async function runPanel(
  ctx: StrategyContext,
  resilience: Resilience,
  members: string[],
  timer: TimerFactory,
  opts: { hasTools: boolean; native: boolean },
): Promise<PanelAnswer[]> {
  const timeoutMs = ctx.config.defaults.panel_member_timeout_s * 1000;
  const tasks = members.map((member) =>
    callPanelMember(ctx, resilience, member, timer, timeoutMs, opts).catch((err: unknown) => {
      ctx.logger.warn(
        { member, reason: err instanceof Error ? err.message : String(err) },
        "fusion: panel member failed, dropping",
      );
      return null;
    }),
  );
  const settled = await Promise.all(tasks);
  const answers: PanelAnswer[] = [];
  for (const a of settled) if (a) answers.push(a);
  return answers;
}

async function callPanelMember(
  ctx: StrategyContext,
  resilience: Resilience,
  member: string,
  timer: TimerFactory,
  timeoutMs: number,
  opts: { hasTools: boolean; native: boolean },
): Promise<PanelAnswer | null> {
  if (!resilience.breaker.canAttempt(member)) {
    logUpstreamFailure(ctx.logger, { stage: "panel", model: member, kind: "circuit_open", latencyMs: 0 });
    ctx.logger.warn({ member }, "fusion: skip panel member (circuit open)");
    return null;
  }
  const body = buildPanelBody(ctx.request, member, opts);
  const startedAt = Date.now();
  let result: ChatCompletionResult;
  const abort = new AbortController();
  try {
    result = await withTimeout(
      resilience.limiter(() =>
        invokeUpstream(ctx.client, body, { stream: false, native: opts.native, signal: combineSignals(ctx.signal, abort.signal) }),
      ),
      timeoutMs,
      timer,
      `panel member '${member}' timed out after ${timeoutMs}ms`,
      () => abort.abort(),
    );
  } catch (err) {
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
  if (result.kind !== "json") {
    resilience.breaker.recordFailure(member);
    logUpstreamFailure(ctx.logger, {
      stage: "panel",
      model: member,
      kind: "error",
      latencyMs: Date.now() - startedAt,
      reason: "unexpected non-json panel result",
    });
    return null;
  }
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
          reason: shortErrorReason(result.data),
        },
        "fusion: panel member dropped (client error response)",
      );
    }
    return null;
  }
  resilience.breaker.recordSuccess(member);
  const content = extractAnswer(result.data);
  if (content === null) {
    // 200 but no usable text (empty content AND empty reasoning, or an unparseable
    // shape). Previously a silent drop; log it so a thin panel is never a mystery.
    ctx.logger.warn(
      { stage: "panel", model: member, latencyMs: Date.now() - startedAt },
      "fusion: panel member returned no usable content (dropped)",
    );
    return null;
  }
  return { member, content };
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

function buildPanelBody(
  request: ChatCompletionRequest,
  member: string,
  opts: { hasTools: boolean; native: boolean },
): Record<string, unknown> {
  // Strip tools/tool_choice and the stream flag; rewrite the model name.
  const { tools, tool_choice: _toolChoice, stream: _stream, model: _model, messages, ...rest } =
    request;
  const msgs: unknown[] = Array.isArray(messages) ? [...messages] : [];
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
  'and "blind_spots" (anything the request needs that none of them addressed). ' +
  "Judge factual correctness and how well each answer actually serves the request; do not reward verbosity. " +
  "Each value may be a string or an array of strings. Output JSON only — no prose, no code fences.";

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
    result = await withTimeout(
      resilience.limiter(() => ctx.client.chatCompletions(body, { stream: false, signal: combineSignals(ctx.signal, abort.signal) })),
      timeoutMs,
      timer,
      `judge '${judge}' timed out after ${timeoutMs}ms`,
      () => abort.abort(),
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
  opts: { stream: boolean; hasTools: boolean; native: boolean; promote: boolean },
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

function buildSynthBody(
  request: ChatCompletionRequest,
  synth: string,
  analysis: JudgeAnalysis | null,
  panelAnswers: PanelAnswer[],
  opts: { stream: boolean; hasTools: boolean; native: boolean },
): Record<string, unknown> {
  // Synth keeps the real tools (if any) and the original messages; we append a
  // synthesis-context system message only on the full fusion path.
  const { stream: _stream, model: _model, messages, ...rest } = request;
  const msgs: unknown[] = Array.isArray(messages) ? [...messages] : [];
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
function buildSynthContext(analysis: JudgeAnalysis | null, panelAnswers: PanelAnswer[]): string | null {
  if (panelAnswers.length === 0) return null;
  const experts = renderPanelForJudge(panelAnswers);
  if (analysis !== null) {
    return (
      "A panel of expert models answered the user's request, and an impartial judge produced a structured " +
      "analysis of their answers. Write the single best final answer: take the actual content (code, formulas, " +
      "exact text) from the expert answers, and use the judge analysis to resolve disagreements, cover blind " +
      "spots, and weight the consensus. Do not drop detail that only one expert provided unless it is wrong.\n\n" +
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
    "Available tools:\n" +
    listed +
    "\n\nDeliberate in prose on the best approach and which tool(s) you would use and why. " +
    "DO NOT emit a tool call — respond with reasoning only."
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
