import { describe, it, expect } from "vitest";
import { UsageAccumulator } from "../src/usage";
import type { ChatCompletionResult, Usage } from "../src/types";

/**
 * H-1 regression: a request that streams MORE THAN ONE upstream call must fold
 * EVERY streamed call's tokens into the finalized aggregate. The accumulator
 * previously kept a single `pendingStream` slot that each new streamed call
 * overwrote, silently dropping all but the last stream's usage.
 */

function streamResult(usage: Usage): ChatCompletionResult {
  return {
    kind: "stream",
    status: 200,
    body: null,
    contentType: "text/event-stream",
    usage: Promise.resolve(usage),
  };
}

function jsonResult(usage: Usage): ChatCompletionResult {
  return { kind: "json", status: 200, data: {}, usage };
}

describe("UsageAccumulator — multiple streamed calls", () => {
  it("sums tokens across two streamed calls in finalize()", async () => {
    const acc = new UsageAccumulator();
    acc.record("m1", streamResult({ promptTokens: 10, completionTokens: 5, totalTokens: 15 }));
    acc.record("m2", streamResult({ promptTokens: 20, completionTokens: 7, totalTokens: 27 }));

    const agg = await acc.finalize();
    expect(agg.upstreamCalls).toBe(2);
    expect(agg.promptTokens).toBe(30);
    expect(agg.completionTokens).toBe(12);
    expect(agg.totalTokens).toBe(42); // before the fix this was only 27 (last stream)
  });

  it("folds streamed and JSON calls together", async () => {
    const acc = new UsageAccumulator();
    acc.record("j1", jsonResult({ promptTokens: 1, completionTokens: 2, totalTokens: 3 }));
    acc.record("s1", streamResult({ promptTokens: 4, completionTokens: 8, totalTokens: 12 }));
    acc.record("s2", streamResult({ promptTokens: 100, completionTokens: 1, totalTokens: 101 }));

    expect(acc.hasPendingStream).toBe(true);
    const agg = await acc.finalize();
    expect(agg.upstreamCalls).toBe(3);
    expect(agg.totalTokens).toBe(116);
  });

  it("snapshot excludes pending streams; finalize includes all of them", async () => {
    const acc = new UsageAccumulator();
    acc.record("s1", streamResult({ promptTokens: 5, completionTokens: 5, totalTokens: 10 }));
    acc.record("s2", streamResult({ promptTokens: 5, completionTokens: 5, totalTokens: 10 }));

    expect(acc.snapshot().totalTokens).toBe(0); // pending streams not yet drained
    expect((await acc.finalize()).totalTokens).toBe(20);
  });
});
