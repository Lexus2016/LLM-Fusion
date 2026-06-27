import { describe, it, expect } from "vitest";
import {
  CircuitBreaker,
  backoffDelay,
  createLimiter,
} from "../src/concurrency";
import { OllamaClient } from "../src/upstream/ollama";
import type { FetchFn } from "../src/types";

describe("backoff helper", () => {
  it("produces increasing delays capped at maxMs (no jitter)", () => {
    const opts = { baseMs: 100, factor: 2, maxMs: 1000, jitter: 0, rng: () => 0 };
    const delays = [0, 1, 2, 3, 4, 5].map((n) => backoffDelay(n, opts));
    expect(delays).toEqual([100, 200, 400, 800, 1000, 1000]);
    // non-decreasing, rising until the cap then flat — and always within bounds
    let prev = -1;
    for (const d of delays) {
      expect(d).toBeGreaterThanOrEqual(prev);
      expect(d).toBeLessThanOrEqual(1000);
      prev = d;
    }
  });

  it("keeps every jittered delay within [raw*(1-jitter), raw]", () => {
    const base = 200;
    const jitter = 0.25;
    for (const r of [0, 0.3, 0.5, 0.9, 0.999]) {
      const d = backoffDelay(2, { baseMs: base, factor: 2, maxMs: 10_000, jitter, rng: () => r });
      const raw = base * 4; // 200 * 2^2
      expect(d).toBeLessThanOrEqual(raw);
      expect(d).toBeGreaterThanOrEqual(Math.round(raw * (1 - jitter)));
    }  });
});

describe("circuit breaker", () => {
  it("opens after N consecutive failures, fast-fails while open, half-opens after cooldown, closes on success", () => {
    let now = 1_000_000;
    const breaker = new CircuitBreaker({ failureThreshold: 3, cooldownMs: 30_000, now: () => now });
    const model = "glm-5.2";

    // closed: allowed
    expect(breaker.canAttempt(model)).toBe(true);
    expect(breaker.getState(model)).toBe("closed");

    // 3 consecutive failures -> open
    breaker.recordFailure(model);
    breaker.recordFailure(model);
    expect(breaker.getState(model)).toBe("closed"); // 2 < threshold
    breaker.recordFailure(model);
    expect(breaker.getState(model)).toBe("open");

    // fast-fail while open (cooldown not elapsed)
    expect(breaker.canAttempt(model)).toBe(false);
    now += 29_999;
    expect(breaker.canAttempt(model)).toBe(false);

    // cooldown elapsed -> half-open allows exactly ONE probe
    now += 1; // total +30_000
    expect(breaker.getState(model)).toBe("half-open");
    expect(breaker.canAttempt(model)).toBe(true); // probe reserved
    expect(breaker.canAttempt(model)).toBe(false); // concurrent probe denied

    // probe succeeds -> closed again, failure count reset
    breaker.recordSuccess(model);
    expect(breaker.getState(model)).toBe("closed");
    expect(breaker.canAttempt(model)).toBe(true);
  });

  it("re-opens (restarting cooldown) when the half-open probe fails", () => {
    let now = 0;
    const breaker = new CircuitBreaker({ failureThreshold: 1, cooldownMs: 10_000, now: () => now });
    const model = "m";

    breaker.recordFailure(model); // threshold 1 -> open
    expect(breaker.getState(model)).toBe("open");

    now += 10_000; // cooldown elapsed -> half-open
    expect(breaker.canAttempt(model)).toBe(true); // probe
    breaker.recordFailure(model); // probe fails -> re-open
    expect(breaker.getState(model)).toBe("open");
    expect(breaker.canAttempt(model)).toBe(false);

    now += 10_000; // cooldown elapsed again
    expect(breaker.getState(model)).toBe("half-open");
  });
});

describe("concurrency limiter", () => {
  it("never runs more than max_concurrency upstream calls at once", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const fetchFn: FetchFn = async () => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight -= 1;
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    const client = new OllamaClient({ baseUrl: "https://mock.test", fetchFn });
    const limiter = createLimiter(2);

    const tasks = Array.from({ length: 8 }, () =>
      limiter(() => client.chatCompletions({ model: "m" }, { stream: false })),
    );
    const results = await Promise.all(tasks);

    expect(results).toHaveLength(8);
    expect(maxInFlight).toBeGreaterThanOrEqual(2);
    expect(maxInFlight).toBeLessThanOrEqual(2);
  });
});
