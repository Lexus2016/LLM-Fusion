import type { ChatCompletionRequest, ChatCompletionResult, ChatMessage, StrategyContext, UpstreamClient } from "./types";
import type { Resilience } from "./concurrency";
import type { ImageDescribeConfig } from "./config";
import { withTimeout, combineSignals } from "./timeout";
import type { TimerFactory } from "./timeout";
import { extractAnswer } from "./reasoning";
import {
  failureKindForError,
  failureKindForStatus,
  isAvailabilityFailureStatus,
  logUpstreamFailure,
} from "./attribution";

/**
 * IMAGE DESCRIBE pre-stage for the fusion strategy (opt-in via `image_describe`).
 *
 * When the request carries OpenAI `image_url` blocks and a multimodal describer
 * model is configured, every image is described ONCE by that model and replaced
 * IN PLACE with a text block (`[IMAGE n]` + description). The rest of the
 * pipeline — panel, judge, synth, bineval — then works on a pure-text request:
 * no panel member or synth needs vision capability, and the vision gate /
 * min_panel_success thinning trap never triggers.
 *
 * All-or-nothing contract: if ANY image cannot be described (describer error,
 * timeout, non-OK status, empty answer) the ORIGINAL request is returned as
 * `null` and the caller falls back to the legacy vision-gate path unchanged.
 * A partial replacement would silently drop an image the user sent.
 */

const DESCRIBE_SYSTEM_PROMPT =
  "You describe images for downstream text-only expert models. Describe the image " +
  "EXHAUSTIVELY in plain text: transcribe ALL visible text VERBATIM (UI labels, menus, " +
  "code, numbers, URLs, captions); then layout, objects, people, colors, spatial " +
  "relationships, chart/graph data points, and anything actionable. Never summarize away " +
  "detail a colleague might need — when text is present, exact transcription beats prose.";

/** Text of the latest user turn, used to focus the description without limiting it. */
function latestUserText(request: ChatCompletionRequest): string {
  const messages = request.messages;
  if (!messages) return "";
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m === undefined || m.role !== "user") continue;
    const c = m.content;
    if (typeof c === "string") return c.slice(0, 400);
    if (Array.isArray(c)) {
      const texts = c
        .filter((p) => p && typeof p === "object" && (p as { type?: string }).type === "text")
        .map((p) => String((p as { text?: unknown }).text ?? ""));
      const joined = texts.join(" ");
      if (joined.trim().length > 0) return joined.slice(0, 400);
    }
  }
  return "";
}

interface ImageLocation {
  mi: number;
  pi: number;
  url: string;
}

/** Collect every `image_url` part location in message order. */
export function collectImageLocations(request: ChatCompletionRequest): ImageLocation[] {
  const out: ImageLocation[] = [];
  const messages = request.messages;
  if (!messages) return out;
  for (let mi = 0; mi < messages.length; mi++) {
    const m = messages[mi];
    if (m === undefined || !Array.isArray(m.content)) continue;
    for (let pi = 0; pi < m.content.length; pi++) {
      const part = m.content[pi];
      if (!part || typeof part !== "object") continue;
      if ((part as { type?: string }).type !== "image_url") continue;
      const url = (part as { image_url?: { url?: unknown } }).image_url?.url;
      if (typeof url === "string" && url.length > 0) out.push({ mi, pi, url });
    }
  }
  return out;
}

async function describeOne(
  ctx: StrategyContext,
  resilience: Resilience,
  cfg: ImageDescribeConfig,
  timer: TimerFactory,
  client: UpstreamClient,
  url: string,
): Promise<string | null> {
  const focus = latestUserText(ctx.request);
  const body: Record<string, unknown> = {
    model: cfg.model,
    stream: false,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              (focus.trim().length > 0
                ? `The user is asking about: ${focus.trim()}\n\n`
                : "") +
              "Describe this image exhaustively per your instructions.",
          },
          { type: "image_url", image_url: { url } },
        ],
      },
    ],
  };
  const startedAt = Date.now();
  const abort = new AbortController();
  let result: ChatCompletionResult;
  try {
    result = await resilience.limiterFor(cfg.model)(() =>
      withTimeout(
        client.chatCompletions(body, { stream: false, signal: combineSignals(ctx.signal, abort.signal) }),
        cfg.timeout_s * 1000,
        timer,
        `image_describe '${cfg.model}' timed out after ${cfg.timeout_s}s`,
        () => abort.abort(),
      ),
    );
  } catch (err) {
    if (ctx.signal?.aborted) {
      resilience.breaker.recordProbeAbandoned(cfg.model);
      return null;
    }
    resilience.breaker.recordFailure(cfg.model);
    ctx.usage?.recordError(cfg.model);
    logUpstreamFailure(ctx.logger, {
      stage: "image_describe",
      model: cfg.model,
      kind: failureKindForError(err),
      latencyMs: Date.now() - startedAt,
      reason: err instanceof Error ? err.message : String(err),
    });
    ctx.logger.warn({ model: cfg.model }, "image_describe: describer call failed");
    return null;
  }
  ctx.usage?.record(cfg.model, result);
  if (result.kind !== "json" || result.status >= 400) {
    if (result.kind !== "json" || isAvailabilityFailureStatus(result.status)) {
      resilience.breaker.recordFailure(cfg.model);
      logUpstreamFailure(ctx.logger, {
        stage: "image_describe",
        model: cfg.model,
        kind: result.kind !== "json" ? "error" : failureKindForStatus(result.status),
        ...(result.kind === "json" ? { status: result.status } : {}),
        latencyMs: Date.now() - startedAt,
      });
    } else {
      resilience.breaker.recordSuccess(cfg.model);
    }
    ctx.logger.warn(
      { model: cfg.model, status: result.kind === "json" ? result.status : undefined },
      "image_describe: describer returned a non-OK response",
    );
    return null;
  }
  resilience.breaker.recordSuccess(cfg.model);
  return extractAnswer(result.data);
}

/**
 * Run the pre-stage. Returns a NEW request object with every image part
 * replaced by a text block, or `null` when there was nothing to describe or
 * any description failed (caller falls back to the legacy vision gate).
 */
export async function describeRequestImages(
  ctx: StrategyContext,
  resilience: Resilience,
  cfg: ImageDescribeConfig,
  timer: TimerFactory,
): Promise<ChatCompletionRequest | null> {
  const locations = collectImageLocations(ctx.request);
  if (locations.length === 0) return null;

  // Read the state BEFORE `canAttempt`, which RESERVES the half-open probe slot
  // as a side effect and would report "half-open" as consumed.
  const wasHalfOpen = resilience.breaker.getState(cfg.model) === "half-open";
  if (!resilience.breaker.canAttempt(cfg.model)) {
    ctx.logger.warn({ model: cfg.model }, "image_describe: skipped (circuit open)");
    return null;
  }

  // Describe every image IN PARALLEL. Sequential describes cost N * timeout_s in
  // the worst case (default 60 s each), which a 4-image paste turns into a 4-min
  // stall before the panel even starts. Fan-out is already bounded by the
  // describer model's own gate in `limiterFor`, so this cannot blow past its
  // per-model concurrency budget. The trade-off on a mid-flight failure is
  // deliberate: the siblings already in flight run to completion instead of being
  // cut short — the contract is still all-or-nothing (we return `null`), and
  // paying for N failed calls once is cheaper than N sequential timeouts on every
  // success.
  //
  // EXCEPT on a half-open breaker, where the contract is "exactly one probe at a
  // time": we hold ONE reserved probe slot, so firing N calls at a model we
  // already believe is sick would spend N quota on a recovery guess and pile
  // extra failures onto its cooldown. Describe image 1 alone as the probe; a
  // success closes the breaker and the rest fan out normally.
  const describeAll = (locs: typeof locations): Promise<(string | null)[]> =>
    Promise.all(locs.map((loc) => describeOne(ctx, resilience, cfg, timer, ctx.client, loc.url)));

  let raws: (string | null)[];
  if (wasHalfOpen && locations.length > 1) {
    const probe = await describeOne(ctx, resilience, cfg, timer, ctx.client, locations[0]!.url);
    raws = probe === null ? [null] : [probe, ...(await describeAll(locations.slice(1)))];
  } else {
    raws = await describeAll(locations);
  }

  const descriptions: string[] = [];
  for (const raw of raws) {
    if (raw === null || raw.trim().length === 0) {
      ctx.logger.warn({ model: cfg.model }, "image_describe: empty/failed description; falling back");
      return null;
    }
    const capped =
      raw.length > cfg.max_chars ? `${raw.slice(0, cfg.max_chars)}\n…[description truncated]` : raw;
    descriptions.push(capped);
  }

  // Rebuild messages immutably, replacing each located image part with its
  // numbered text block. Locations are visited in collection order.
  let k = 0;
  const replaceInMessage = (m: ChatMessage, mi: number): ChatMessage => {
    if (!Array.isArray(m.content)) return m;
    const parts = m.content.map((p, pi) => {
      const loc = locations[k];
      if (loc !== undefined && loc.mi === mi && loc.pi === pi) {
        k += 1;
        return { type: "text", text: `[IMAGE ${k}]\n${descriptions[k - 1] ?? ""}` };
      }
      return p;
    });
    return { ...m, content: parts };
  };
  const messages = (ctx.request.messages ?? []).map(replaceInMessage);
  return { ...ctx.request, messages };
}
