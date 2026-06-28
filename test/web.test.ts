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
    expect(out).toEqual([{ title: "T", url: "https://example.com", content: "hello world" }]);
  });

  it("returns null on a non-2xx (degrades, never throws)", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ error: "bad key" }, 401),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toBeNull();
  });

  it("returns null when the body is not a Tavily shape", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => jsonResponse({ ok: false }),
      },
    ]);
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toBeNull();
  });

  it("returns null when the network throws", async () => {
    const fetchFn: FetchFn = async () => {
      throw new Error("network down");
    };
    const out = await tavilySearch("query", cfg({ fetch: fetchFn }));
    expect(out).toBeNull();
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
    expect(out).not.toBeNull();
    expect(out).toContain("fresh fact");
  });

  it("returns null when the search yields nothing (no injection)", async () => {
    const fetchFn = mockFetch([
      {
        match: (url) => url === "https://api.tavily.com/search",
        respond: () => tavilyResponse([]),
      },
    ]);
    const out = await buildWebContext("query", cfg({ fetch: fetchFn }));
    expect(out).toBeNull();
  });
});