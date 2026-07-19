import { z } from "zod";
import type { ChatCompletionResult, RequestUsage, Usage } from "./types";

/**
 * Per-request upstream usage accounting (spec §3 / §12 — cost is an inherent,
 * accepted property of the fusion design, so the proxy measures it precisely).
 *
 * Two concerns live here:
 *  1. Parsing the OpenAI-compatible `usage` object off a completion body or a
 *     final SSE chunk into a typed, zero-defaulted `Usage`.
 *  2. A per-request `UsageAccumulator` that strategies feed (one `record` per
 *     upstream call) and the server reads to log + return the aggregate. A
 *     streamed call's tokens are only known once the stream drains, so the
 *     accumulator keeps that call's usage as a pending promise the server folds
 *     in at the stream's flush.
 *
 * Prompt CONTENT never reaches this module — only token counts.
 */

export const ZERO_USAGE: Usage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

/** USD price for one model, per 1M tokens (input / output). */
export interface ModelPricing {
  input_per_mtok: number;
  output_per_mtok: number;
}

/** model id -> pricing. An empty/absent map means cost accounting is off. */
export type PricingMap = Record<string, ModelPricing>;

const RawUsageSchema = z
  .object({
    prompt_tokens: z.number().nonnegative().optional(),
    completion_tokens: z.number().nonnegative().optional(),
    total_tokens: z.number().nonnegative().optional(),
  })
  .passthrough();

/** Parse a bare `usage` object (zod), defaulting every field to 0. */
export function parseUsage(raw: unknown): Usage {
  const parsed = RawUsageSchema.safeParse(raw);
  if (!parsed.success) return { ...ZERO_USAGE };
  const promptTokens = parsed.data.prompt_tokens ?? 0;
  const completionTokens = parsed.data.completion_tokens ?? 0;
  const totalTokens = parsed.data.total_tokens ?? promptTokens + completionTokens;
  return { promptTokens, completionTokens, totalTokens };
}

const BodyWithUsageSchema = z.object({ usage: z.unknown().optional() }).passthrough();

/** Extract + parse the `usage` field from a completion body (or SSE chunk). */
export function usageFromBody(data: unknown): Usage {
  const parsed = BodyWithUsageSchema.safeParse(data);
  return parseUsage(parsed.success ? parsed.data.usage : undefined);
}

/** Render a `Usage` into the OpenAI-compatible `usage` object for clients. */
export function toOpenAiUsage(usage: Usage): {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
} {
  return {
    prompt_tokens: usage.promptTokens,
    completion_tokens: usage.completionTokens,
    total_tokens: usage.totalTokens,
  };
}

/** Cost of one call under `pricing`; 0 when the call's model has no entry. */
function costForCall(usage: Usage, model: string, pricing: PricingMap): number {
  const entry = pricing[model];
  if (!entry) return 0;
  return (
    (usage.promptTokens / 1_000_000) * entry.input_per_mtok +
    (usage.completionTokens / 1_000_000) * entry.output_per_mtok
  );
}

interface CallRecord {
  model: string;
  usage: Usage;
}

function aggregate(
  callCount: number,
  records: CallRecord[],
  pricing: PricingMap | undefined,
): RequestUsage {
  // Cost is null unless a non-empty pricing map is configured (default: off).
  const priced = pricing !== undefined && Object.keys(pricing).length > 0;
  let promptTokens = 0;
  let completionTokens = 0;
  let totalTokens = 0;
  let costUsd = priced ? 0 : null;
  for (const r of records) {
    promptTokens += r.usage.promptTokens;
    completionTokens += r.usage.completionTokens;
    totalTokens += r.usage.totalTokens;
    if (priced && costUsd !== null && pricing) costUsd += costForCall(r.usage, r.model, pricing);
  }
  return { upstreamCalls: callCount, promptTokens, completionTokens, totalTokens, costUsd };
}

/**
 * Per-request usage accumulator. One instance per incoming `/v1/chat/completions`
 * request; passed to every strategy through the request context so panel + judge
 * + synth (and router for `smart`) all fold into the same totals.
 */
export class UsageAccumulator {
  private callCount = 0;
  private readonly records: CallRecord[] = [];
  private readonly pendingStreams: Array<{ model: string; usage: Promise<Usage> }> = [];

  /** Record one completed upstream call (its model + result). */
  record(model: string, result: ChatCompletionResult): void {
    this.callCount += 1;
    if (result.kind === "json") {
      this.records.push({ model, usage: result.usage });
    } else {
      // Streamed call: counted now, tokens resolve once the stream drains. A
      // request may stream more than one upstream call, so every pending stream
      // is kept (never overwritten) and all are folded in at finalize().
      this.pendingStreams.push({ model, usage: result.usage });
    }
  }

  /** Record a call that threw before producing a result (network/timeout). */
  recordError(model: string): void {
    this.callCount += 1;
    this.records.push({ model, usage: { ...ZERO_USAGE } });
  }

  /** True when a streamed call's tokens are still pending the stream drain. */
  get hasPendingStream(): boolean {
    return this.pendingStreams.length > 0;
  }

  /** Aggregate known so far — excludes a pending stream's tokens (header at send time). */
  snapshot(pricing?: PricingMap): RequestUsage {
    return aggregate(this.callCount, this.records, pricing);
  }

  /** Aggregate including every streamed call's tokens (awaits each drain). */
  async finalize(pricing?: PricingMap): Promise<RequestUsage> {
    if (this.pendingStreams.length > 0) {
      const pending = [...this.pendingStreams];
      await Promise.all(
        pending.map(async (p) => {
          const usageVal = await p.usage;
          const idx = this.pendingStreams.indexOf(p);
          if (idx === -1) return; // already claimed by an overlapping finalize() call
          this.pendingStreams.splice(idx, 1);
          this.records.push({ model: p.model, usage: usageVal });
        }),
      );
    }
    return aggregate(this.callCount, this.records, pricing);
  }
}

/** Compact header value for `x-fusion-usage`: calls + total tokens (JSON). */
export function usageHeaderValue(usage: RequestUsage): string {
  return JSON.stringify({ calls: usage.upstreamCalls, total: usage.totalTokens });
}

// --- SSE stream usage capture ---------------------------------------------

interface DataLine {
  kind: "done" | "data" | "other";
  obj?: unknown;
}

/** Classify a single SSE line: terminal `[DONE]`, a JSON `data:` chunk, or other. */
export function classifyDataLine(line: string): DataLine {
  const trimmed = line.trimStart();
  if (!trimmed.startsWith("data:")) return { kind: "other" };
  const payload = trimmed.slice("data:".length).trim();
  if (payload.length === 0) return { kind: "other" };
  if (payload === "[DONE]") return { kind: "done" };
  try {
    return { kind: "data", obj: JSON.parse(payload) };
  } catch {
    return { kind: "other" };
  }
}

/** True when a parsed SSE chunk carries a top-level `usage` (the final chunk). */
function chunkHasUsage(obj: unknown): boolean {
  const parsed = BodyWithUsageSchema.safeParse(obj);
  return parsed.success && parsed.data.usage != null;
}

/**
 * Tap an upstream SSE body: forward every byte verbatim while scanning complete
 * lines for the final `usage` chunk. The returned promise resolves with the
 * captured usage (zeros if none) once the body drains. Used by the upstream
 * client to surface a streamed call's usage without disturbing the stream.
 */
export function tapStreamUsage(body: ReadableStream<Uint8Array> | null): {
  stream: ReadableStream<Uint8Array> | null;
  usage: Promise<Usage>;
} {
  if (body === null) return { stream: null, usage: Promise.resolve({ ...ZERO_USAGE }) };
  let resolveUsage: (u: Usage) => void;
  const usage = new Promise<Usage>((resolve) => {
    resolveUsage = resolve;
  });
  const decoder = new TextDecoder();
  let buffer = "";
  let captured: Usage | null = null;

  const scan = (line: string): void => {
    const c = classifyDataLine(line);
    if (c.kind === "data" && chunkHasUsage(c.obj)) {
      captured = usageFromBody(c.obj);
    }
  };

  const reader = body.getReader();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          if (buffer.length > 0) scan(buffer);
          resolveUsage(captured ?? { ...ZERO_USAGE });
          controller.close();
          return;
        }
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          let nl: number;
          while ((nl = buffer.indexOf("\n")) !== -1) {
            scan(buffer.slice(0, nl));
            buffer = buffer.slice(nl + 1);
          }
          controller.enqueue(value);
        }
      } catch (err) {
        resolveUsage(captured ?? { ...ZERO_USAGE });
        controller.error(err);
      }
    },
    cancel(reason) {
      resolveUsage(captured ?? { ...ZERO_USAGE });
      reader.cancel(reason).catch(() => {});
    },
  });

  return { stream, usage };
}

/** Metadata threaded into the stream usage-injection transform for logging. */
export interface StreamUsageMeta {
  reqId: string;
  model: string;
  created: number;
}

/**
 * Build a transform that drops the upstream `[DONE]` and any upstream usage-only
 * chunk, then emits ONE aggregate `usage` chunk followed by `[DONE]`. The
 * aggregate (panel + judge + synth, or the single/router + routed calls) comes
 * from `accumulator.finalize`, which awaits the streamed call's drained tokens.
 * Composes after the fusion reasoning->content transform without disturbing it.
 */
export function makeUsageInjectionTransform(
  accumulator: UsageAccumulator,
  meta: StreamUsageMeta,
  pricing: PricingMap | undefined,
): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";

  const handleLine = (line: string, controller: TransformStreamDefaultController<Uint8Array>): void => {
    const c = classifyDataLine(line);
    if (c.kind === "done") return; // re-emitted in flush, after our usage chunk
    if (c.kind === "data" && chunkHasUsage(c.obj)) {
      // Upstream sometimes sends `usage` in the same chunk as `finish_reason` /
      // `choices`. Drop only the `usage` field and forward the rest so the client
      // does not lose `finish_reason` or final content.
      const obj = c.obj as Record<string, unknown>;
      const { usage: _, ...rest } = obj;
      if (Object.keys(rest).length > 0) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(rest)}\n`));
      }
      return;
    }
    controller.enqueue(encoder.encode(line + "\n"));
  };

  return new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        handleLine(buffer.slice(0, nl), controller);
        buffer = buffer.slice(nl + 1);
      }
    },
    async flush(controller) {
      buffer += decoder.decode();
      if (buffer.length > 0) handleLine(buffer, controller);
      const usage = await accumulator.finalize(pricing);
      const chunk = {
        id: `fusion-usage-${meta.reqId}`,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [],
        usage: toOpenAiUsage({
          promptTokens: usage.promptTokens,
          completionTokens: usage.completionTokens,
          totalTokens: usage.totalTokens,
        }),
      };
      controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
    },
  });
}
