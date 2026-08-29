import { describe, it, expect } from "vitest";
import {
  CircuitBreaker,
  backoffDelay,
  createLimiter,
  createResilience,
  throttledGateWaitLogger,
} from "../src/concurrency";
import type { Logger } from "pino";
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

  it("recordProbeAbandoned frees a reserved half-open probe without opening the breaker (client disconnect)", () => {
    let now = 0;
    const breaker = new CircuitBreaker({ failureThreshold: 1, cooldownMs: 10_000, now: () => now });
    const model = "m";

    breaker.recordFailure(model); // open
    now += 10_000; // half-open
    expect(breaker.canAttempt(model)).toBe(true); // probe reserved (probeInFlight=true)
    expect(breaker.canAttempt(model)).toBe(false); // concurrent probe denied

    // Client disconnects before the probe settles: release without recording a failure.
    breaker.recordProbeAbandoned(model);
    expect(breaker.getState(model)).toBe("half-open");
    // The probe slot is free again, so a new probe is allowed.
    expect(breaker.canAttempt(model)).toBe(true);
    // Still half-open (not re-opened): a subsequent success closes normally.
    breaker.recordSuccess(model);
    expect(breaker.getState(model)).toBe("closed");
  });

  it("recordProbeAbandoned is a no-op outside half-open (does not corrupt closed/open state)", () => {
    const breaker = new CircuitBreaker({ failureThreshold: 2, cooldownMs: 10_000, now: () => 0 });
    const model = "m";
    expect(breaker.canAttempt(model)).toBe(true); // closed
    breaker.recordProbeAbandoned(model);
    expect(breaker.getState(model)).toBe("closed");
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

describe("per-model keyed limiter (limiterFor)", () => {
  /** A tracked job factory: records per-model and global in-flight peaks. */
  function makeProbe() {
    const inFlight = new Map<string, number>();
    const peak = new Map<string, number>();
    let globalInFlight = 0;
    let globalPeak = 0;
    const job = (model: string, ms = 10) => async () => {
      inFlight.set(model, (inFlight.get(model) ?? 0) + 1);
      peak.set(model, Math.max(peak.get(model) ?? 0, inFlight.get(model)!));
      globalInFlight += 1;
      globalPeak = Math.max(globalPeak, globalInFlight);
      await new Promise((r) => setTimeout(r, ms));
      inFlight.set(model, inFlight.get(model)! - 1);
      globalInFlight -= 1;
    };
    return { job, peak: (m: string) => peak.get(m) ?? 0, globalPeak: () => globalPeak };
  }

  it("caps an overridden model at its own budget while global slots remain free", async () => {
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { "deepseek-v4-pro": 2 } },
    });
    const probe = makeProbe();
    await Promise.all(
      Array.from({ length: 6 }, () => r.limiterFor("deepseek-v4-pro")(probe.job("deepseek-v4-pro"))),
    );
    expect(probe.peak("deepseek-v4-pro")).toBe(2);
  });

  it("binds a family-name budget to every tagged id of that family", async () => {
    // Ollama ids carry a tag (`deepseek-v4-flash:0731-cloud`) and every call site
    // passes the full tagged id, but operators write the family in the config.
    // Before the family fallback this key gated nothing and the model silently
    // ran at the global cap.
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { "deepseek-v4-flash": 2 } },
    });
    const probe = makeProbe();
    const id = "deepseek-v4-flash:0731-cloud";
    await Promise.all(Array.from({ length: 6 }, () => r.limiterFor(id)(probe.job(id))));
    expect(probe.peak(id)).toBe(2);
  });

  it("prefers an exact tagged key over its family key", async () => {
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { "glm-5.3": 1, "glm-5.3:cloud": 3 } },
    });
    const probe = makeProbe();
    await Promise.all(
      Array.from({ length: 6 }, () => r.limiterFor("glm-5.3:cloud")(probe.job("glm-5.3:cloud"))),
    );
    expect(probe.peak("glm-5.3:cloud")).toBe(3);
  });

  it("does not let a family key leak across different families", async () => {
    const r = createResilience({
      maxConcurrency: 6,
      perModel: { overrides: { "deepseek-v4-flash": 1 } },
    });
    const probe = makeProbe();
    // Same prefix STRING, different family — must not inherit the budget.
    const other = "deepseek-v4-flash-turbo:cloud";
    await Promise.all(Array.from({ length: 4 }, () => r.limiterFor(other)(probe.job(other))));
    expect(probe.peak(other)).toBe(4);
  });

  it("a saturated model queues at its own gate and does not block another model", async () => {
    const r = createResilience({ maxConcurrency: 8, perModel: { overrides: { slow: 1 } } });
    let release!: () => void;
    const blocked = new Promise<void>((res) => (release = res));
    // Saturate `slow`: one call running (holding a global slot), many queued at
    // ITS OWN gate — those queued calls must hold no global slots.
    const slowJobs = [
      r.limiterFor("slow")(() => blocked),
      ...Array.from({ length: 20 }, () => r.limiterFor("slow")(() => Promise.resolve())),
    ];
    let fastRan = false;
    await r.limiterFor("fast")(async () => {
      fastRan = true;
    });
    expect(fastRan).toBe(true);
    release();
    await Promise.all(slowJobs);
  });

  it("clamps a per-model budget configured ABOVE the global cap", async () => {
    // The schema accepts `per_model_concurrency: { m: 10 }` with
    // `max_concurrency: 4`. Unclamped, the model gate admits 10 and six calls
    // pile up in the GLOBAL queue — the exact head-of-line blocking this gate
    // exists to prevent — and the saturation line stays silent while they wait,
    // then reports a depth measured at the wrong gate.
    const seen: { scope: string; budget: number; queued: number }[] = [];
    const r = createResilience({
      maxConcurrency: 4,
      perModel: { overrides: { m: 10 } },
      onGateWait: ({ scope, budget, queued }) => seen.push({ scope, budget, queued }),
    });
    let release!: () => void;
    const blocked = new Promise<void>((res) => (release = res));
    const jobs = Array.from({ length: 6 }, () => r.limiterFor("m")(() => blocked));
    // Calls 5 and 6 queue at a gate that is 4 wide, not 10, and say so.
    expect(seen).toEqual([
      { scope: "global", budget: 4, queued: 1 },
      { scope: "global", budget: 4, queued: 2 },
    ]);
    release();
    await Promise.all(jobs);
  });

  it("defaults to the global budget when unconfigured (behavior unchanged)", async () => {
    const r = createResilience({ maxConcurrency: 3 });
    const probe = makeProbe();
    await Promise.all(Array.from({ length: 9 }, () => r.limiterFor("m")(probe.job("m"))));
    expect(probe.peak("m")).toBe(3); // bounded by the global cap, not tighter
  });

  it("the global cap still bounds the SUM across models", async () => {
    const r = createResilience({ maxConcurrency: 3 });
    const probe = makeProbe();
    await Promise.all(
      ["a", "b", "c"].flatMap((m) => Array.from({ length: 3 }, () => r.limiterFor(m)(probe.job(m)))),
    );
    expect(probe.globalPeak()).toBeLessThanOrEqual(3);
  });

  it("applies perModel.defaultPerModel to models without an explicit override", async () => {
    const r = createResilience({ maxConcurrency: 8, perModel: { defaultPerModel: 2 } });
    const probe = makeProbe();
    await Promise.all(
      Array.from({ length: 6 }, () => r.limiterFor("any-model")(probe.job("any-model"))),
    );
    expect(probe.peak("any-model")).toBe(2);
  });
});

describe("per-model gate saturation telemetry", () => {
  it("reports a wait ONLY once the model's budget is spent, naming model/budget/queue", async () => {
    const seen: { model: string; budget: number; queued: number }[] = [];
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { hot: 2 } },
      onGateWait: (info) => seen.push(info),
    });
    let release!: () => void;
    const blocked = new Promise<void>((res) => (release = res));
    // Two calls fill the budget without waiting; the next three must queue.
    const jobs = Array.from({ length: 5 }, () => r.limiterFor("hot")(() => blocked));
    expect(seen).toHaveLength(3);
    expect(seen[0]).toEqual({ model: "hot", budget: 2, queued: 1, scope: "per_model" });
    expect(seen[2]).toEqual({ model: "hot", budget: 2, queued: 3, scope: "per_model" });
    release();
    await Promise.all(jobs);
  });

  it("stays silent for a model that never saturates (no log spam on the happy path)", async () => {
    const seen: string[] = [];
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { cool: 4 } },
      onGateWait: (info) => seen.push(info.model),
    });
    await Promise.all(Array.from({ length: 3 }, () => r.limiterFor("cool")(() => Promise.resolve())));
    expect(seen).toEqual([]);
  });

  it("throttles the log to one line per model per interval, but never merges two models", () => {
    const lines: { obj: unknown; msg: string }[] = [];
    const logger = {
      warn: (obj: unknown, msg: string) => lines.push({ obj, msg }),
    } as unknown as Parameters<typeof throttledGateWaitLogger>[0];
    let clock = 1_000;
    const observe = throttledGateWaitLogger(logger, { intervalMs: 30_000, now: () => clock });

    observe({ model: "a", budget: 2, queued: 1, scope: "per_model" });
    observe({ model: "a", budget: 2, queued: 2, scope: "per_model" }); // same model, inside the window
    observe({ model: "b", budget: 2, queued: 1, scope: "per_model" }); // different model, own window
    expect(lines).toHaveLength(2);

    clock += 29_999;
    observe({ model: "a", budget: 2, queued: 9, scope: "per_model" }); // still inside a's window
    expect(lines).toHaveLength(2);

    clock += 1;
    observe({ model: "a", budget: 2, queued: 9, scope: "per_model" }); // window elapsed
    expect(lines).toHaveLength(3);
    expect(lines[2]!.msg).toContain("per-model concurrency budget saturated");
  });

  it("names the GLOBAL cap when the model has no per-model budget of its own", async () => {
    // sizeFor falls back to maxConcurrency, so an unconfigured model's gate can
    // never bind before the global limiter does. Reporting that as a per-model
    // problem sends the operator to `per_model_concurrency`, where raising the
    // value changes nothing — the real knob is `max_concurrency`.
    const seen: { scope: string; budget: number }[] = [];
    const r = createResilience({
      maxConcurrency: 2,
      perModel: {}, // nothing configured
      onGateWait: ({ scope, budget }) => seen.push({ scope, budget }),
    });
    let release!: () => void;
    const blocked = new Promise<void>((res) => (release = res));
    const jobs = [
      r.limiterFor("m")(() => blocked),
      r.limiterFor("m")(() => blocked),
      r.limiterFor("m")(() => blocked), // third call queues
    ];
    expect(seen).toEqual([{ scope: "global", budget: 2 }]);
    release();
    await Promise.all(jobs);
  });

  it("still names the per-model budget when one is actually configured", async () => {
    const seen: string[] = [];
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { m: 2 } },
      onGateWait: ({ scope }) => seen.push(scope),
    });
    let release!: () => void;
    const blocked = new Promise<void>((res) => (release = res));
    const jobs = [r.limiterFor("m")(() => blocked), r.limiterFor("m")(() => blocked), r.limiterFor("m")(() => blocked)];
    expect(seen).toEqual(["per_model"]);
    release();
    await Promise.all(jobs);
  });

  it("the logger tells the operator WHICH knob to reach for", () => {
    const lines: { obj: Record<string, unknown>; msg: string }[] = [];
    const logger = {
      warn: (obj: Record<string, unknown>, msg: string) => lines.push({ obj, msg }),
    } as unknown as Parameters<typeof throttledGateWaitLogger>[0];
    let clock = 0;
    const observe = throttledGateWaitLogger(logger, { intervalMs: 10, now: () => clock });
    observe({ model: "a", budget: 2, queued: 1, scope: "per_model" });
    clock += 100;
    observe({ model: "b", budget: 8, queued: 1, scope: "global" });
    expect(lines[0]!.msg).toContain("per_model_concurrency");
    expect(lines[1]!.msg).toContain("max_concurrency");
    expect(lines[1]!.msg).not.toContain("per-model concurrency budget saturated");
  });

  it("holds its invariants across 80 randomised interleavings (budgets, caps, failures)", async () => {
    // Spot tests pin the cases you thought of. The counters here are mutated from
    // two places (submit and settle) across two scopes, which is exactly the
    // shape where an unimagined interleaving hides — so sweep instead.
    let seed = 12345; // deterministic: a failure is reproducible
    const rnd = (n: number) => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) % n);

    for (let trial = 0; trial < 80; trial++) {
      const maxConcurrency = 1 + rnd(4);
      const models = ["a", "b", "c"];
      const overrides: Record<string, number> = {};
      for (const m of models) if (rnd(2) === 0) overrides[m] = 1 + rnd(4);

      const waits: { model: string; scope: string; queued: number }[] = [];
      const r = createResilience({
        maxConcurrency,
        perModel: { overrides },
        onGateWait: (i) => waits.push({ model: i.model, scope: i.scope, queued: i.queued }),
      });

      const jobs: Promise<unknown>[] = [];
      for (let i = 0, n = 1 + rnd(10); i < n; i++) {
        const m = models[rnd(models.length)]!;
        const fail = rnd(5) === 0; // rejections must release their slots too
        jobs.push(
          r
            .limiterFor(m)(async () => {
              await new Promise((res) => setTimeout(res, rnd(2)));
              if (fail) throw new Error("boom");
              return 1;
            })
            .catch(() => undefined),
        );
      }
      await Promise.all(jobs);

      // A reported depth of zero or less means the counter drifted from its budget.
      for (const w of waits) expect(w.queued, `trial=${trial} ${JSON.stringify(w)}`).toBeGreaterThan(0);

      // Everything settled: a fresh single call on an idle limiter must be silent.
      waits.length = 0;
      await r.limiterFor("a")(() => Promise.resolve());
      await r.limiterFor("b")(() => Promise.resolve());
      expect(waits, `trial=${trial} leaked with ${JSON.stringify({ maxConcurrency, overrides })}`).toEqual([]);
    }
  });

  it("a THROWING observer neither breaks the call nor leaks a gate slot", () => {
    // Telemetry sits between the `outstanding` increment and the limiter call, so
    // an observer that throws used to escape synchronously out of
    // `limiterFor(m)(fn)` — the upstream call was never even submitted — and left
    // the counter permanently inflated, so the gate reported saturation forever.
    const seen: string[] = [];
    const r = createResilience({
      maxConcurrency: 8,
      perModel: { overrides: { m: 1 } },
      onGateWait: () => {
        seen.push("fired");
        throw new Error("logger blew up");
      },
    });
    let release!: () => void;
    const blocked = new Promise<void>((res) => (release = res));
    const first = r.limiterFor("m")(() => blocked);
    // Saturates the gate -> observer fires -> throws. Must not escape.
    const second = r.limiterFor("m")(() => Promise.resolve("ok"));
    expect(seen).toEqual(["fired"]);
    release();
    return Promise.all([first, second]).then(async () => {
      // Counter recovered: with both calls settled the gate is empty again, so a
      // fresh single call must NOT be reported as queued.
      seen.length = 0;
      await r.limiterFor("m")(() => Promise.resolve("ok"));
      expect(seen).toEqual([]);
    });
  });
});
