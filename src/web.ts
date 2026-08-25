/**
 * Optional web grounding for the fusion panel (spec: research freshness).
 *
 * Design — a single shared pre-stage, NOT a per-member tool loop:
 *   1. One Tavily search is run ONCE per fusion call, before the panel fans out.
 *   2. The cleaned results are formatted as prose and injected into every panel
 *      member's prompt as a `user` turn (inserted before the latest user message),
 *      NOT as a `system` message: some panel members (kimi-k2.7-code) ignore live
 *      facts placed in a system role and refuse on a stale training cutoff, while
 *      the same facts in a user turn make them answer. No panel member ever
 *      receives real `tools`, so the one-`tool_calls`-per-step invariant for the
 *      agent loop is untouched: the synth is still the only stage that may emit a
 *      client-visible tool call.
 *
 * Gating (the user's hard requirement): the feature is OFF unless
 * `TAVILY_API_KEY` is set in the environment, AND the model opts in via
 * `web_search.enabled` in `fusion.yaml`. Either missing → no search, no
 * latency, no cost. The `fusion_planning_turn_only` knob additionally keeps it
 * off mid-loop (the panel does not even run on tool-result continuations).
 *
 * Hygiene: Tavily returns cleaned `content` per result, so no raw HTML reaches
 * the judge/synth. Output is capped to `max_context_chars` to bound prompt bloat.
 */

import type { FetchFn } from "./types";

const TAVILY_URL = "https://api.tavily.com/search";

interface TavilyResult {
  title?: unknown;
  url?: unknown;
  content?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Narrow an unknown JSON payload to a Tavily `results` array, or null. */
function parseTavilyResponse(data: unknown): { results: TavilyResult[] } | null {
  if (!isRecord(data)) return null;
  const results = data.results;
  if (!Array.isArray(results)) return null;
  const narrowed: TavilyResult[] = [];
  for (const r of results) {
    if (isRecord(r)) narrowed.push(r);
  }
  return { results: narrowed };
}

export interface WebSearchResult {
  title: string;
  url: string;
  content: string;
}

export interface WebGroundingConfig {
  apiKey: string;
  maxResults: number;
  timeoutMs: number;
  maxContextChars: number;
  /** Injected for tests; defaults to global fetch. */
  fetch?: FetchFn;
}

/** A Tavily key is "present" iff it is a non-empty trimmed string. */
export function webGroundingEnabled(apiKey: string | undefined): apiKey is string {
  return typeof apiKey === "string" && apiKey.trim().length > 0;
}

/**
 * Why a failure REASON and not just `null`: grounding degrades silently by
 * design (an ungrounded panel still answers), so the only way an operator ever
 * learns that their Tavily key is dead, expired, or rate-limited is the log
 * line. A bare `null` makes "the key 401s on every call" indistinguishable from
 * "the search legitimately found nothing" — the caller must be able to log the
 * difference.
 */
export type WebSearchFailure =
  /** fetch threw: DNS, TLS, connection reset, timeout/abort, or a refused redirect. */
  | { reason: "network"; detail: string }
  /** Tavily answered with a non-2xx — 401 = bad/expired key, 429 = plan limit. */
  | { reason: "http_status"; status: number }
  /** 2xx whose body is not JSON, or JSON without a `results` array. */
  | { reason: "bad_body" }
  /** A successful search that matched nothing. Benign — not an error. */
  | { reason: "no_results" };

export type WebSearchOutcome =
  | { ok: true; results: WebSearchResult[] }
  | { ok: false; failure: WebSearchFailure };

/**
 * Run a single Tavily search. Never throws into the request path: every failure
 * comes back as `{ ok: false, failure }` so the caller can degrade to an
 * ungrounded panel AND log why.
 */
export async function tavilySearch(
  query: string,
  cfg: WebGroundingConfig,
  signal?: AbortSignal,
): Promise<WebSearchOutcome> {
  const fetchFn = cfg.fetch ?? (globalThis.fetch as FetchFn);
  const timeoutSignal = AbortSignal.timeout(cfg.timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
  let res: Response;
  try {
    res = await fetchFn(TAVILY_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      // The POST body carries the Tavily API key; never follow redirects
      // (a 307/308 would resend this body, with the key, to an attacker-controlled
      // host). Treat a redirect as a hard error instead.
      redirect: "error",
      body: JSON.stringify({
        api_key: cfg.apiKey,
        query,
        max_results: cfg.maxResults,
        include_answer: false,
        search_depth: "basic",
      }),
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
  const payload = parseTavilyResponse(data);
  if (payload === null) return { ok: false, failure: { reason: "bad_body" } };
  const out: WebSearchResult[] = [];
  for (const r of payload.results) {
    const title = typeof r.title === "string" ? r.title : "";
    const url = typeof r.url === "string" ? r.url : "";
    const content = typeof r.content === "string" ? r.content : "";
    if (url) out.push({ title, url, content });
  }
  if (out.length === 0) return { ok: false, failure: { reason: "no_results" } };
  return { ok: true, results: out };
}

/**
 * Format search results into a single grounded-context prose block, capped to
 * `maxContextChars`. `null` when there is nothing to inject.
 */
export function formatWebContext(results: WebSearchResult[], maxContextChars: number): string | null {
  if (results.length === 0) return null;
  const blocks: string[] = [];
  let used = 0;
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    if (r === undefined) continue;
    const head = `[${i + 1}] ${r.title || "(untitled)"} — ${r.url}`;
    const body = r.content.trim();
    const block = `${head}\n${body}`;
    if (used + block.length + 2 > maxContextChars && blocks.length > 0) break;
    blocks.push(block);
    used += block.length + 2;
    if (used >= maxContextChars) break;
  }
  if (blocks.length === 0) return null;
  return (
    "WEB CONTEXT — current information retrieved from a live web search. These results " +
    "are MORE RECENT than your training data, so for any question about recent events, " +
    "current releases, prices, or post-cutoff facts you MUST base your answer on this " +
    "context and treat it as the source of truth (do NOT refuse on the grounds that your " +
    "training is out of date). Cite the URL of the result you rely on where it matters. " +
    "Where two web results conflict, prefer the more recent / authoritative one and note the " +
    "conflict. Do NOT silently fall back to your training-data cutoff when this context " +
    "answers the question.\n" +
    blocks.join("\n\n")
  );
}

export type WebContextOutcome =
  | { ok: true; context: string }
  | { ok: false; failure: WebSearchFailure };

/**
 * One-shot grounding: search the query and return a formatted context block.
 * Never throws — a failure surfaces as `{ ok: false, failure }` carrying the
 * reason, so the caller degrades to an ungrounded panel with a log line rather
 * than in silence.
 */
export async function buildWebContext(
  query: string,
  cfg: WebGroundingConfig,
  signal?: AbortSignal,
): Promise<WebContextOutcome> {
  const outcome = await tavilySearch(query, cfg, signal);
  if (!outcome.ok) return outcome;
  const context = formatWebContext(outcome.results, cfg.maxContextChars);
  // Defensive only, and deliberately not described as a live case: `tavilySearch`
  // already reports an empty result set as `no_results`, and `formatWebContext`
  // always admits its FIRST block regardless of the char budget (the budget check
  // is guarded on `blocks.length > 0`). So a non-empty result set cannot render
  // to null here — this branch exists so a future change to either function
  // cannot turn into an unhandled null.
  if (context === null) return { ok: false, failure: { reason: "no_results" } };
  return { ok: true, context };
}