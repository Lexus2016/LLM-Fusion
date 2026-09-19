import { describe, expect, it } from "vitest";
import { mockFetch, jsonResponse } from "./helpers";
import { askNouls, typesafeEnabled, type TypeSafeConfig } from "../src/typesafe";
import { gateWebResults, type WebGateThresholds } from "../src/web_gate";
import type { WebSearchResult } from "../src/web";

const URL = "https://api.typesafe.ai/v1/systemone";

function cfg(fetchFn: ReturnType<typeof mockFetch>, overrides: Partial<TypeSafeConfig> = {}): TypeSafeConfig {
  return { apiKey: "ts-test-key", model: "jev-latest", timeoutMs: 5000, fetch: fetchFn, ...overrides };
}

function answers(nouls: Record<string, number>): Response {
  return jsonResponse({
    model: "jev-1.13.0",
    answers: Object.fromEntries(Object.entries(nouls).map(([k, v]) => [k, { type: "noul", noul: v }])),
    usage: { input_tokens: 312, output_tokens: 0 },
  });
}

function route(respond: (url: string, init?: RequestInit) => Response) {
  return mockFetch([{ match: (u) => u === URL, respond }]);
}

const QUESTION = { q: { instructions: "Does this convey urgency?" } };

describe("typesafe — enabled gate", () => {
  it("is enabled only for a non-empty key", () => {
    expect(typesafeEnabled("ts-abc")).toBe(true);
    expect(typesafeEnabled("  ts-abc  ")).toBe(true);
    expect(typesafeEnabled("")).toBe(false);
    expect(typesafeEnabled("   ")).toBe(false);
    expect(typesafeEnabled(undefined)).toBe(false);
  });
});

describe("typesafe — askNouls", () => {
  it("sends the documented request shape and returns the probabilities", async () => {
    let seen: { init?: RequestInit } = {};
    const out = await askNouls(
      { request: "hi" },
      {
        injection: {
          instructions: "Does this steer the reader?",
          criteria: { true: "it instructs", false: "it informs" },
        },
      },
      cfg(
        route((_u, init) => {
          seen = { init };
          return answers({ injection: 0.12 });
        }),
      ),
    );

    expect(out).toEqual({ ok: true, nouls: { injection: 0.12 }, usage: { input_tokens: 312, output_tokens: 0 } });

    const headers = seen.init?.headers as Record<string, string>;
    expect(headers.authorization).toBe("Bearer ts-test-key");
    // A 307/308 would replay the Authorization header at another host.
    expect(seen.init?.redirect).toBe("error");
    const body = JSON.parse(String(seen.init?.body));
    expect(body.model).toBe("jev-latest");
    expect(body.state).toEqual({ request: "hi" });
    expect(body.questions.injection).toEqual({
      type: "noul",
      instructions: "Does this steer the reader?",
      criteria: { true: "it instructs", false: "it informs" },
    });
  });

  it("omits criteria entirely when a question has none", async () => {
    let body: Record<string, unknown> = {};
    await askNouls({}, QUESTION, cfg(route((_u, init) => {
      body = JSON.parse(String(init?.body));
      return answers({ q: 0.5 });
    })));
    expect(body.questions).toEqual({ q: { type: "noul", instructions: "Does this convey urgency?" } });
  });

  it("reports a missing answer as bad_body rather than a partial result", async () => {
    // Silently dropping an unanswered question turns a MISSING judgment into a
    // passing one at the caller's threshold — the one failure mode a gate cannot have.
    const out = await askNouls({}, { a: { instructions: "?" }, b: { instructions: "?" } }, cfg(route(() => answers({ a: 0.9 }))));
    expect(out).toEqual({ ok: false, failure: { reason: "bad_body" } });
  });

  it("reports a non-numeric answer as bad_body", async () => {
    const out = await askNouls({}, QUESTION, cfg(route(() => jsonResponse({ answers: { q: { noul: "high" } } }))));
    expect(out).toEqual({ ok: false, failure: { reason: "bad_body" } });
  });

  it("carries the status for a rejected key, a rate limit, and an overload", async () => {
    for (const status of [401, 422, 429, 529]) {
      const out = await askNouls({}, QUESTION, cfg(route(() => jsonResponse({ error: "nope" }, status))));
      expect(out).toEqual({ ok: false, failure: { reason: "http_status", status } });
    }
  });

  it("reports a thrown fetch as network, with the detail", async () => {
    const out = await askNouls({}, QUESTION, cfg(async () => {
      throw new Error("ECONNRESET");
    }));
    expect(out).toEqual({ ok: false, failure: { reason: "network", detail: "ECONNRESET" } });
  });

  it("reports a non-JSON 200 as bad_body", async () => {
    const out = await askNouls({}, QUESTION, cfg(route(() => new Response("<html>", { status: 200 }))));
    expect(out).toEqual({ ok: false, failure: { reason: "bad_body" } });
  });
});

const THRESHOLDS: WebGateThresholds = { injectionMax: 0.7, relevantMin: 0.45, evidenceMin: 0.55 };

function result(n: number): WebSearchResult {
  return { title: `t${n}`, url: `https://e${n}.test`, content: `c${n}` };
}

describe("web_gate — screening web results", () => {
  it("drops an injection, an off-topic page, and an evidence-free page; keeps the rest in order", async () => {
    const byUrl: Record<string, Record<string, number>> = {
      "https://e1.test": { injection: 0.02, relevant: 0.9, evidence: 0.8 }, // keep
      "https://e2.test": { injection: 0.95, relevant: 0.9, evidence: 0.9 }, // injection wins
      "https://e3.test": { injection: 0.01, relevant: 0.1, evidence: 0.9 }, // off topic
      "https://e4.test": { injection: 0.01, relevant: 0.9, evidence: 0.2 }, // nothing usable
      "https://e5.test": { injection: 0.02, relevant: 0.8, evidence: 0.7 }, // keep
    };
    const fetchFn = route((_u, init) => {
      const state = JSON.parse(String(init?.body)).state as { web_result: { url: string } };
      return answers(byUrl[state.web_result.url]!);
    });
    const results = [1, 2, 3, 4, 5].map(result);
    const out = await gateWebResults("q", results, cfg(fetchFn), THRESHOLDS);

    expect(out.kept.map((r) => r.url)).toEqual(["https://e1.test", "https://e5.test"]);
    expect(out.decisions.map((d) => d.verdict)).toEqual(["kept", "injection", "irrelevant", "no_evidence", "kept"]);
    expect(out.failure).toBeUndefined();
  });

  it("tests injection FIRST, so a poisoned page that is also relevant and useful still drops", async () => {
    // The ordering is the point: a useful-looking page is exactly what an injection
    // hides inside, so a quality test that ran first would keep it.
    const out = await gateWebResults(
      "q",
      [result(1)],
      cfg(route(() => answers({ injection: 0.99, relevant: 0.99, evidence: 0.99 }))),
      THRESHOLDS,
    );
    expect(out.kept).toEqual([]);
    expect(out.decisions[0]?.verdict).toBe("injection");
  });

  it("fails OPEN when the screener is unreachable, and reports the failure", async () => {
    const out = await gateWebResults("q", [result(1), result(2)], cfg(route(() => jsonResponse({}, 429))), THRESHOLDS);
    expect(out.kept.map((r) => r.url)).toEqual(["https://e1.test", "https://e2.test"]);
    expect(out.decisions.map((d) => d.verdict)).toEqual(["unscreened", "unscreened"]);
    expect(out.failure).toEqual({ reason: "http_status", status: 429 });
  });

  it("screens each result against the request as a pair", async () => {
    const states: unknown[] = [];
    await gateWebResults(
      "how do refresh tokens expire",
      [result(1), result(2)],
      cfg(route((_u, init) => {
        states.push(JSON.parse(String(init?.body)).state);
        return answers({ injection: 0, relevant: 1, evidence: 1 });
      })),
      THRESHOLDS,
    );
    expect(states).toHaveLength(2); // one request per result, not one for the list
    expect(states[0]).toEqual({
      request: "how do refresh tokens expire",
      web_result: { title: "t1", url: "https://e1.test", content: "c1" },
    });
  });
});
