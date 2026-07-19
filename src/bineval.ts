import { z } from "zod";
import type { StrategyContext, ChatCompletionResult } from "./types";
import type { Resilience } from "./concurrency";
import { withTimeout, combineSignals } from "./timeout";
import type { TimerFactory } from "./timeout";
import { extractAnswer } from "./reasoning";
import { extractJsonObject } from "./json";

/**
 * BinEval-style binary quality evaluation for generated outputs.
 *
 * Decomposes quality into atomic yes/no questions, asks an evaluator LLM to
 * answer them independently, and aggregates the verdicts into per-dimension
 * and overall scores. This is intentionally separate from the fusion judge:
 * the judge adjudicates the panel, while bineval scores the *final* output.
 */

export interface BinaryQuestion {
  dimension: string;
  question: string;
}

export interface BinaryVerdict {
  dimension: string;
  question: string;
  verdict: boolean;
  explanation?: string;
}

export interface BinaryEvaluationResult {
  /** Average score across all dimensions, in [0, 1]. */
  overall: number;
  /** Per-dimension average scores, in [0, 1]. */
  dimensions: Record<string, number>;
  /** The raw question-level verdicts. */
  verdicts: BinaryVerdict[];
}

export const BinaryVerdictSchema = z.object({
  dimension: z.string().min(1),
  question: z.string().min(1),
  verdict: z.boolean(),
  explanation: z.string().optional(),
});

export const BinaryEvaluationResponseSchema = z
  .object({
    verdicts: z.array(BinaryVerdictSchema).min(1),
  })
  .passthrough();

/** Default dimensions and their binary questions. Keep them generic so they fit most agent tasks. */
export const DEFAULT_DIMENSIONS: BinaryQuestion[] = [
  {
    dimension: "factual_consistency",
    question:
      "Does the answer contain only claims that are well-supported by the user's request and widely accepted facts, with no invented details?",
  },
  {
    dimension: "instruction_following",
    question:
      "Does the answer follow all explicit instructions in the user's request (format, constraints, requested steps, tone)?",
  },
  {
    dimension: "format_compliance",
    question:
      "Does the answer use the requested output format correctly (for example: valid JSON, Markdown, code block, list) if a format was specified?",
  },
  {
    dimension: "completeness",
    question: "Does the answer address every part of the user's request without omitting required information?",
  },
  {
    dimension: "clarity",
    question: "Is the answer clear, concise, and free of confusing, contradictory, or unparseable statements?",
  },
];

export const BINEVAL_SYSTEM_PROMPT =
  "You are an exacting quality evaluator. You are given the user's ORIGINAL REQUEST and a CANDIDATE ANSWER. " +
  "Evaluate the candidate answer by answering each question below with a binary yes/no verdict. " +
  "For each verdict, provide a one-sentence explanation. " +
  'Respond with ONLY a JSON object of the form {"verdicts": [{"dimension": string, "question": string, "verdict": boolean, "explanation": string}]}. ' +
  "Output JSON only — no prose, no code fences.";

export function buildBinevalUserPrompt(
  requestText: string,
  outputText: string,
  questions: BinaryQuestion[],
): string {
  return (
    "ORIGINAL USER REQUEST:\n" +
    requestText +
    "\n\nCANDIDATE ANSWER:\n" +
    outputText +
    "\n\nEVALUATION QUESTIONS (answer each yes/no):\n" +
    questions.map((q, i) => `${i + 1}. [${q.dimension}] ${q.question}`).join("\n") +
    "\n\nReturn only the JSON verdict object."
  );
}

export function aggregateVerdicts(verdicts: BinaryVerdict[]): BinaryEvaluationResult {
  const byDimension: Record<string, number[]> = {};
  for (const v of verdicts) {
    const arr = byDimension[v.dimension] ?? [];
    arr.push(v.verdict ? 1 : 0);
    byDimension[v.dimension] = arr;
  }

  const dimensions: Record<string, number> = {};
  let total = 0;
  let count = 0;
  for (const [dim, scores] of Object.entries(byDimension)) {
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    dimensions[dim] = avg;
    total += avg;
    count += 1;
  }

  const overall = count === 0 ? 0 : total / count;
  return { overall, dimensions, verdicts };
}

export function parseBinaryEvaluation(
  content: string | null,
  questions: BinaryQuestion[],
): BinaryEvaluationResult | null {
  if (!content) return null;
  const rawText = extractJsonObject(content);
  if (!rawText) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(rawText);
  } catch {
    return null;
  }
  const parsed = BinaryEvaluationResponseSchema.safeParse(raw);
  if (!parsed.success) return null;

  const verdicts = parsed.data.verdicts;
  if (verdicts.length !== questions.length) return null;
  // Match by dimension as a multiset: BinEval deliberately uses MULTIPLE questions
  // per dimension (e.g. several factual-consistency checks), so duplicates are
  // legitimate. We compare the configured dimension counts against the returned
  // dimension counts — this accepts N questions for the same dimension while still
  // rejecting missing, extra, or unknown dimensions. Question text is not matched
  // verbatim because LLMs lightly paraphrase it.
  const expected = new Map<string, number>();
  for (const q of questions) expected.set(q.dimension, (expected.get(q.dimension) ?? 0) + 1);
  const actual = new Map<string, number>();
  for (const v of verdicts) {
    if (!expected.has(v.dimension)) return null;
    actual.set(v.dimension, (actual.get(v.dimension) ?? 0) + 1);
  }
  for (const [dim, count] of expected) {
    if (actual.get(dim) !== count) return null;
  }

  return aggregateVerdicts(verdicts);
}

export async function runBineval(
  ctx: StrategyContext,
  resilience: Resilience,
  model: string,
  requestText: string,
  outputText: string,
  questions: BinaryQuestion[],
  timer: TimerFactory,
  timeoutMs: number,
): Promise<BinaryEvaluationResult | null> {
  if (!resilience.breaker.canAttempt(model)) {
    ctx.logger.warn({ model }, "bineval: skipped (circuit open)");
    return null;
  }

  const body: Record<string, unknown> = {
    model,
    temperature: 0,
    response_format: { type: "json_object" },
    stream: false,
    messages: [
      { role: "system", content: BINEVAL_SYSTEM_PROMPT },
      { role: "user", content: buildBinevalUserPrompt(requestText, outputText, questions) },
    ],
  };

  const abort = new AbortController();
  const startedAt = Date.now();
  let result: ChatCompletionResult;
  try {
    result = await resilience.limiterFor(model)(() =>
      withTimeout(
        ctx.client.chatCompletions(body, { stream: false, signal: combineSignals(ctx.signal, abort.signal) }),
        timeoutMs,
        timer,
        `bineval '${model}' timed out after ${timeoutMs}ms`,
        () => abort.abort(),
      ),
    );
  } catch (err) {
    // Client disconnect is not an evaluator health failure: do not trip the
    // breaker. Still release any reserved half-open probe so the model can be
    // probed again. Detect via the client signal, not the error name — a stage
    // timeout also aborts the fetch and must still count as a failure.
    if (ctx.signal?.aborted) {
      resilience.breaker.recordProbeAbandoned(model);
      return null;
    }
    resilience.breaker.recordFailure(model);
    ctx.usage?.recordError(model);
    ctx.logger.warn(
      { model, reason: err instanceof Error ? err.message : String(err) },
      "bineval: call failed",
    );
    return null;
  }

  ctx.usage?.record(model, result);

  if (result.kind !== "json" || result.status >= 400) {
    if (result.kind !== "json" || result.status >= 500 || result.status === 429) {
      resilience.breaker.recordFailure(model);
    } else {
      // A 4xx non-availability means the model answered, so it is healthy —
      // release any half-open probe so it is not jammed open until process
      // restart (same rationale as single.ts).
      resilience.breaker.recordSuccess(model);
    }
    ctx.logger.warn(
      { model, status: result.kind === "json" ? result.status : undefined },
      "bineval: non-OK response",
    );
    return null;
  }

  resilience.breaker.recordSuccess(model);
  const content = extractAnswer(result.data);
  return parseBinaryEvaluation(content, questions);
}
