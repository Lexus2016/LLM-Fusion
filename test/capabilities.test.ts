import { describe, it, expect } from "vitest";
import { CapabilityService, parseShow } from "../src/capabilities";
import { createLogger } from "../src/logging";

const logger = createLogger({ level: "silent" });

describe("capabilities", () => {
  it("parses vision + tools + generic context_length", () => {
    const cap = parseShow({
      capabilities: ["completion", "vision"],
      model_info: { "qwen3.context_length": 32768, "general.architecture": "qwen3" },
    });
    expect(cap.vision).toBe(true);
    expect(cap.tools).toBe(false);
    expect(cap.context).toBe(32768);
  });

  it("matches a differently-prefixed context_length key (generic match)", () => {
    const cap = parseShow({
      capabilities: ["completion", "tools"],
      model_info: { "llama.context_length": 8192 },
    });
    expect(cap.tools).toBe(true);
    expect(cap.vision).toBe(false);
    expect(cap.context).toBe(8192);
  });

  it("discovers via client.show and caches the result", async () => {
    let calls = 0;
    const svc = new CapabilityService({
      client: {
        show: async () => {
          calls += 1;
          return { capabilities: ["vision"], model_info: { "glm.context_length": 4096 } };
        },
      },
      getOverrides: () => ({}),
      logger,
    });
    const a = await svc.discover("glm-5.2");
    const b = await svc.discover("glm-5.2");
    expect(a.source).toBe("discovered");
    expect(a.capability.vision).toBe(true);
    expect(a.capability.context).toBe(4096);
    expect(b.capability.context).toBe(4096);
    expect(calls).toBe(1);
  });

  it("degrades to an override on show failure", async () => {
    const svc = new CapabilityService({
      client: {
        show: async () => {
          throw new Error("no /api/show");
        },
      },
      getOverrides: () => ({ "kimi-k2.7-code": { vision: true, tools: false, context: 128000 } }),
      logger,
    });
    const r = await svc.discover("kimi-k2.7-code");
    expect(r.source).toBe("override");
    expect(r.capability.vision).toBe(true);
    expect(r.capability.tools).toBe(false);
    expect(r.capability.context).toBe(128000);
  });

  it("degrades to conservative defaults when there is no override", async () => {
    const svc = new CapabilityService({
      client: {
        show: async () => {
          throw new Error("boom");
        },
      },
      getOverrides: () => ({}),
      logger,
    });
    const r = await svc.discover("unknown-model");
    expect(r.source).toBe("default");
    expect(r.capability).toEqual({ vision: false, tools: true, context: null });
  });

  it("does NOT cache a degraded `default` result; retries discovery and caches the later success (C-3)", async () => {
    let calls = 0;
    const svc = new CapabilityService({
      client: {
        show: async () => {
          calls += 1;
          if (calls === 1) throw new Error("transient blip"); // first call fails -> default
          return { capabilities: ["vision"], model_info: { "glm.context_length": 4096 } };
        },
      },
      getOverrides: () => ({}),
      logger,
    });

    const a = await svc.discover("glm-5.2");
    expect(a.source).toBe("default"); // failure -> conservative default, NOT cached
    expect(a.capability.vision).toBe(false);

    const b = await svc.discover("glm-5.2"); // retries: transient failure gone
    expect(b.source).toBe("discovered");
    expect(b.capability.vision).toBe(true);
    expect(b.capability.context).toBe(4096);

    const c = await svc.discover("glm-5.2"); // discovered result IS cached
    expect(c.source).toBe("discovered");
    expect(calls).toBe(2); // one failed + one success; the third call hit the cache
  });

  it("coalesces concurrent discovery of the same cold model onto ONE client.show", async () => {
    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const svc = new CapabilityService({
      client: {
        show: async () => {
          calls += 1;
          await gate;
          return { capabilities: ["vision", "tools"], model_info: { "glm.context_length": 4096 } };
        },
      },
      getOverrides: () => ({}),
      logger,
    });

    const a = svc.discover("glm-5.2");
    const b = svc.discover("glm-5.2");
    const c = svc.discover("glm-5.2");
    expect(calls).toBe(1); // all three joined the single in-flight request
    release();
    const [ra, rb, rc] = await Promise.all([a, b, c]);
    expect(calls).toBe(1);
    expect(ra).toBe(rb); // same object, not just an equal one
    expect(rb).toBe(rc);
    expect(ra.source).toBe("discovered");
    expect(ra.capability.context).toBe(4096);

    // The in-flight entry is released after settling, and the result is cached.
    await svc.discover("glm-5.2");
    expect(calls).toBe(1);
  });

  it("coalescing does NOT cache a `default` result; the next call retries", async () => {
    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const svc = new CapabilityService({
      client: {
        show: async () => {
          calls += 1;
          if (calls === 1) {
            await gate;
            throw new Error("transient blip");
          }
          return { capabilities: ["vision"], model_info: { "glm.context_length": 4096 } };
        },
      },
      getOverrides: () => ({}),
      logger,
    });

    const a = svc.discover("glm-5.2");
    const b = svc.discover("glm-5.2");
    release();
    const [ra, rb] = await Promise.all([a, b]);
    expect(calls).toBe(1); // both concurrent callers shared the failing call
    expect(ra.source).toBe("default");
    expect(rb.source).toBe("default");

    const retry = await svc.discover("glm-5.2"); // not cached -> re-discovered
    expect(retry.source).toBe("discovered");
    expect(retry.capability.context).toBe(4096);
    expect(calls).toBe(2);
  });

  it("clear() during an in-flight discover leaves no pre-reload data in the cache", async () => {
    const shown: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let context = 4096;
    const svc = new CapabilityService({
      client: {
        show: async (model: string) => {
          shown.push(model);
          const observed = context; // snapshot at call time, before blocking
          if (shown.length === 1) await gate; // the pre-reload call blocks
          return { capabilities: ["vision"], model_info: { "glm.context_length": observed } };
        },
      },
      getOverrides: () => ({}),
      logger,
    });

    const inFlight = svc.discover("glm-5.2");
    svc.clear(); // config hot-reload lands mid-discovery
    context = 8192; // the reloaded upstream reports a different context window

    release();
    const stale = await inFlight;
    expect(stale.capability.context).toBe(4096); // the awaiting caller still gets a result

    const fresh = await svc.discover("glm-5.2");
    expect(fresh.capability.context).toBe(8192); // NOT served from the stale cache
    expect(shown).toEqual(["glm-5.2", "glm-5.2"]); // pre-reload result was not reused

    const cached = await svc.discover("glm-5.2"); // post-reload result IS cached
    expect(cached.capability.context).toBe(8192);
    expect(shown.length).toBe(2);
  });
});
