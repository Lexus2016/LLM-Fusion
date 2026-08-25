import { describe, expect, it } from "vitest";
import { mockFetch, jsonResponse } from "./helpers";
import type { FetchFn } from "../src/types";
import {
  buildWebContext,
  formatWebContext,
  tavilySearch,
  webGroundingEnabled,
  type WebGroundingConfig,
  type WebSearchResult,
} from "../src/web";

function cfg(overrides: Partial<WebGroundingConfig> = {}): WebGroundingConfig {
  return {
    apiKey: "test-key",
    maxResults: 3,
    timeoutMs: 5000,
    maxContextChars: 4000,
    ...overrides,
  };
}

function tavilyResponse(results: { title: string; url: string; content: string }[]): Response {
  return jsonResponse({ results });
}

describe("web grounding — enabled gate", () => {
  it("is enabled only for a non-empty key", () => {
    expect(webGroundingEnabled("tvly-abc")).toBe(true);
    expect(webGroundingEnabled("  tvly-abc  ")).toBe(true);
    expect(webGroundingEnabled("")).toBe(false);
    expect(webGroundingEnabled("   ")).toBe(false);
    expect(webGroundingEnabled(undefined)).toBe(false);
  });
});

describe("web grounding — formatWebContext", () => {
  it("returns null for an empty result set", () => {
    expect(formatWebContext([], 4000)).toBeNull();
  });

  it("formats results as a labeled, verifiable context block", () => {
    const results: WebSearchResult[] = [
      { title: "Redis docs", url: "https://redis.io/docs", content: "Redis is an in-memory store." },
      { title: "Lua scripting", url: "https://redis.io/lua", content: "EVAL runs server-side Lua." },
    ];
    const out = formatWebContext(results, 4000);
    expect(out).not.toBeNull();
    expect(out).toContain("WEB CONTEXT");
    expect(out).toContain("MORE RECENT");
    expect(out).toContain("source of truth");
    expect(out).toContain("[1] Redis docs — https://redis.io/docs");
    expect(out).toContain("Redis is an in-memory store.");
    expect(out).toContain("[2] Lua scripting — https://redis.io/lua");
  });

  it("BOUNDS a single oversized result instead of admitting it whole", () => {
    // The first block is always admitted so a lone result is never dropped, but
    // unbounded it blew straight through the cap and that text goes verbatim
    // into every panel member's prompt.
    const out = formatWebContext([{ title: "t", url: "https://x", content: "z".repeat(50_000) }], 200);
    expect(out).not.toBeNull();
    // Blocks are capped; the fixed preamble is deliberately outside the budget.
    const blocksOnly = out!.slice(out!.indexOf("[1] "));
    // The ellipsis counts against the budget — never one char over what was asked.
    expect(blocksOnly.length).toBeLessThanOrEqual(200);
    expect(blocksOnly).toContain("…");
  });

  it("holds the budget and surrogate-safety across every budget 0..300", () => {
    // Boundary sweep rather than a spot check: the cap interacts with an
    // ellipsis that counts against the budget AND with surrogate pairs, and an
    // off-by-one in either shows up only at specific widths.
    const content = "a😀b😀".repeat(200);
    for (let budget = 0; budget <= 300; budget++) {
      const out = formatWebContext([{ title: "t", url: "https://x", content }], budget);
      if (out === null) continue;
      const blocks = out.slice(out.indexOf("[1] "));
      expect(blocks.length, `budget=${budget}`).toBeLessThanOrEqual(Math.max(budget, 1));
      for (let i = 0; i < blocks.length; i++) {
        const c = blocks.charCodeAt(i);
        if (c >= 0xd800 && c <= 0xdbff) {
          const next = blocks.charCodeAt(i + 1);
          expect(next >= 0xdc00 && next <= 0xdfff, `lone high surrogate at ${i}, budget=${budget}`).toBe(true);
          i++;
        } else {
          expect(c >= 0xdc00 && c <= 0xdfff, `lone low surrogate at ${i}, budget=${budget}`).toBe(false);
        }
      }
    }
  });

  it("caps the block to maxContextChars, dropping later results", () => {
    const results: WebSearchResult[] = [
      { title: "a", url: "https://a", content: "x".repeat(2000) },
      { title: "b", url: "https://b", content: "y".repeat(2000) },
      { title: "c", url: "https://c", content: "z".repeat(2000) },
    ];
    const out = formatWebContext(results, 2500);
    expect(out).not.toBeNull();
    expect(out!.length).toBeLessThanOrEqual(3000); // head + one body, not all three
    expect(out).toContain("https://a");
    expect(out).not.toContain("https://c");
  });
});

describe("web grounding — tavilySearch", () => {
  it("returns parsed results on a 200 with a results array", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () =>
          tavilyResponse([
            { title: "T", url: "https://example.com", content: "hello world" },
          ]),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: true, results: [{ title: "T", url: "https://example.com", content: "hello world" }] });
  });

  // The reason matters: a dead key must be distinguishable from an empty search
  // in the caller's log, otherwise grounding rots invisibly (see fusion.ts).
  it("reports the STATUS on a non-2xx (degrades, never throws)", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ error: "bad key" }, 401),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "http_status", status: 401 } });
  });

  it("reports bad_body when the body is not a Tavily shape", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ ok: false }),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "bad_body" } });
  });

  it("reports the network error message when fetch throws", async () => {
    const fetchFn: FetchFn = async () => {
      throw new Error("network down");
    };
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "network", detail: "network down" } });
  });

  it("reports bad_body when results exist but none is usable (shape drift, not an empty search)", async () => {
    // If Tavily renames or drops `url`, every entry is filtered out. Calling that
    // "found nothing" would let grounding die permanently behind a reassuring log.
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ results: [{ title: "t", content: "c" }, { title: "t2", content: "c2" }] }),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "bad_body" } });
  });

  it("reports bad_body when the entries are not objects at all", async () => {
    // Narrowing drops non-record entries, so the surviving array is empty — but
    // the response DID carry results, so this is shape drift, not a null search.
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ results: [null, "nope", 42] }),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "bad_body" } });
  });

  it("reports no_results (not an error) on an empty 200 result set", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => tavilyResponse([]),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "no_results" } });
  });
});

describe("web grounding — buildWebContext", () => {
  it("returns a context block on a successful search", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () =>
          tavilyResponse([
            { title: "T", url: "https://example.com", content: "fresh fact" },
          ]),
      },
    ]);
    const out = await buildWebContext("query", cfg({ fetch: fetchFn }));
    expect(out.ok).toBe(true);
    expect(out.ok && out.context).toContain("fresh fact");
  });

  it("reports no_results when the search yields nothing (no injection)", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => tavilyResponse([]),
      },
    ]);
    const out = await buildWebContext("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "no_results" } });
  });

  it("propagates the search failure reason instead of flattening it to 'nothing'", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ error: "rate limited" }, 429),
      },
    ]);
    const out = await buildWebContext("query", cfg({ fetch: fetchFn }));
    expect(out).toEqual({ ok: false, failure: { reason: "http_status", status: 429 } });
  });

  it("sends redirect:'error' so the Tavily API key cannot leak via 307/308", async () => {
    let capturedRedirect: RequestInit["redirect"] | undefined;
    const fetchFn: FetchFn = async (_input, init) => {
      capturedRedirect = init?.redirect;
      return tavilyResponse([{ title: "T", url: "https://example.com", content: "hello world" }]);
    };
    await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(capturedRedirect).toBe("error");
  });
});