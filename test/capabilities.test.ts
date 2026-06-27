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
});
