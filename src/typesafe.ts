/**
 * Optional TypeSafe (System One / Jev) judgments — the typed-decision seam.
 *
 * WHAT THIS IS FOR: places where the proxy needs a judgment rather than a fact,
 * and where the alternative is a hand-written heuristic (a regex list, a table of
 * phrases) or a second "prompt an LLM and parse its JSON" round trip. TypeSafe
 * answers a named map of yes/no questions with calibrated probabilities, so the
 * threshold lives in our code instead of inside an English sentence addressed to
 * a model.
 *
 * GATING, exactly like web grounding: the feature is OFF unless `TYPESAFE_API_KEY`
 * is set in the environment AND the model opts in via config. Either missing → no
 * call, no latency, no cost. Nothing here runs on a default install.
 *
 * WHY RAW FETCH AND NOT `@typesafe-ai/sdk`: a dependency is installed whether or
 * not the key exists, and this endpoint is one POST with a JSON body — the same
 * trade already made for Tavily in `src/web.ts`. The `fetch` seam keeps the test
 * suite offline.
 *
 * NEVER THROWS into the request path. Every failure returns `{ ok: false, failure }`
 * carrying a reason, so a caller degrades deliberately and logs WHY: an operator
 * whose key expired must be able to tell that from "the judgments came back clean".
 */

import { z } from "zod";
import type { FetchFn } from "./types";

const TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone";

/** The default model alias. Pin a version (`jev-1.13.0`) if you tune thresholds. */
export const DEFAULT_TYPESAFE_MODEL = "jev-latest";

/** A yes/no question. `criteria` is optional but sharpens the two ends of the scale. */
export interface NoulQuestion {
  instructions: string;
  criteria?: { true: string; false: string };
}

export interface TypeSafeConfig {
  apiKey: string;
  model: string;
  timeoutMs: number;
  /** Injected for tests; defaults to global fetch. */
  fetch?: FetchFn;
}

/** A TypeSafe key is "present" iff it is a non-empty trimmed string. */
export function typesafeEnabled(apiKey: string | undefined): apiKey is string {
  return typeof apiKey === "string" && apiKey.trim().length > 0;
}

/**
 * Why a failure REASON and not just `null`: every caller of this module degrades
 * silently by design (the pipeline still answers without the judgment), so the log
 * line is the only way an operator learns their key is dead. `null` would make
 * "401 on every call" indistinguishable from "the answers were unremarkable".
 */
export type TypeSafeFailure =
  /** fetch threw: DNS, TLS, connection reset, timeout/abort, or a refused redirect. */
  | { reason: "network"; detail: string }
  /** Non-2xx. 401 = bad key, 422 = malformed question, 429 = rate limit, 529 = overloaded. */
  | { reason: "http_status"; status: number }
  /** 2xx whose body is not JSON, or JSON that does not carry the answers we asked for. */
  | { reason: "bad_body" };

export interface TypeSafeUsage {
  input_tokens: number;
  output_tokens: number;
}

export type NoulOutcome =
  | { ok: true; nouls: Record<string, number>; usage: TypeSafeUsage }
  | { ok: false; failure: TypeSafeFailure };

const NoulAnswerSchema = z.object({ noul: z.number() }).passthrough();
const ResponseSchema = z
  .object({
    answers: z.record(z.unknown()),
    usage: z
      .object({
        input_tokens: z.number().optional(),
        output_tokens: z.number().optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

/**
 * Ask one batch of yes/no questions about one state.
 *
 * Everything goes in ONE request on purpose. Jev ingests the state once and
 * evaluates every question against it in parallel, so a batch costs one state
 * instead of N — the difference the docs measure at roughly an order of magnitude
 * in both cost and latency. Never split a battery across calls to "keep questions
 * focused"; focus is per question, not per request.
 *
 * A response missing any question we asked for is `bad_body`, not a partial
 * result: a caller thresholds the answers it expects, and silently dropping one
 * would turn a missing judgment into a passing one.
 */
export async function askNouls(
  state: unknown,
  questions: Record<string, NoulQuestion>,
  cfg: TypeSafeConfig,
  signal?: AbortSignal,
): Promise<NoulOutcome> {
  const fetchFn = cfg.fetch ?? (globalThis.fetch as FetchFn);
  const timeoutSignal = AbortSignal.timeout(cfg.timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

  const body = {
    state,
    model: cfg.model,
    questions: Object.fromEntries(
      Object.entries(questions).map(([id, q]) => [
        id,
        { type: "noul", instructions: q.instructions, ...(q.criteria ? { criteria: q.criteria } : {}) },
      ]),
    ),
  };

  let res: Response;
  try {
    res = await fetchFn(TYPESAFE_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        // The key travels in a header, but a 307/308 would replay the whole request —
        // header included — against whatever host the redirect names. Same rule as
        // the Tavily call: treat a redirect as a hard error rather than follow it.
        authorization: `Bearer ${cfg.apiKey}`,
      },
      redirect: "error",
      body: JSON.stringify(body),
      signal: combined,
    });
  } catch (err) {
    return { ok: false, failure: { reason: "network", detail: err instanceof Error ? err.message : String(err) } };
  }

  if (!res.ok) return { ok: false, failure: { reason: "http_status", status: res.status } };

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    return { ok: false, failure: { reason: "bad_body" } };
  }

  const parsed = ResponseSchema.safeParse(data);
  if (!parsed.success) return { ok: false, failure: { reason: "bad_body" } };

  const nouls: Record<string, number> = {};
  for (const id of Object.keys(questions)) {
    const answer = NoulAnswerSchema.safeParse(parsed.data.answers[id]);
    if (!answer.success || !Number.isFinite(answer.data.noul)) {
      return { ok: false, failure: { reason: "bad_body" } };
    }
    nouls[id] = answer.data.noul;
  }

  return {
    ok: true,
    nouls,
    usage: {
      input_tokens: parsed.data.usage?.input_tokens ?? 0,
      output_tokens: parsed.data.usage?.output_tokens ?? 0,
    },
  };
}
