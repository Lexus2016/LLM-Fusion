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
export function webGroundingEnabled(apiKey: string | undefined): boolean {
  return typeof apiKey === "string" && apiKey.trim().length > 0;
}

/**
 * Run a single Tavily search. Returns `null` on any failure (network, non-2xx,
 * malformed body) so the caller degrades to an ungrounded panel — never throws
 * into the request path.
 */
export async function tavilySearch(
  query: string,
  cfg: WebGroundingConfig,
  signal?: AbortSignal,
): Promise<WebSearchResult[] | null> {
  const fetchFn = cfg.fetch ?? (globalThis.fetch as FetchFn);
  const timeoutSignal = AbortSignal.timeout(cfg.timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
  let res: Response;
  try {
    res = await fetchFn(TAVILY_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        api_key: cfg.apiKey,
        query,
        max_results: cfg.maxResults,
        include_answer: false,
        search_depth: "basic",
      }),
      signal: combined,
    });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  let data: unknown;
  try {
    data = await res.json();
  } catch {
    return null;
  }
  const payload = parseTavilyResponse(data);
  if (payload === null) return null;
  const out: WebSearchResult[] = [];
  for (const r of payload.results) {
    const title = typeof r.title === "string" ? r.title : "";
    const url = typeof r.url === "string" ? r.url : "";
    const content = typeof r.content === "string" ? r.content : "";
    if (url) out.push({ title, url, content });
  }
  return out;
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

/**
 * One-shot grounding: search the query and return a formatted context block, or
 * `null` when the search failed or returned nothing. Never throws.
 */
export async function buildWebContext(
  query: string,
  cfg: WebGroundingConfig,
  signal?: AbortSignal,
): Promise<string | null> {
  const results = await tavilySearch(query, cfg, signal);
  if (results === null) return null;
  return formatWebContext(results, cfg.maxContextChars);
}