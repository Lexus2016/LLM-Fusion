import { describe, it, expect } from "vitest";
import { parseConfig } from "../src/config";

const minimal = {
  upstream: { base_url: "https://ollama.com", api_key_env: "OLLAMA_API_KEY" },
  server: { bind: "127.0.0.1", port: 8080 },
  models: { "fast-glm": { strategy: "single", target: "glm-5.2" } },
};

describe("config", () => {
  it("parses a valid minimal config and applies defaults", () => {
    const cfg = parseConfig(minimal);
    expect(cfg.upstream.api_mode).toBe("auto");
    expect(cfg.upstream.max_concurrency).toBe(4);
    expect(cfg.upstream.request_timeout_s).toBe(170);
    expect(cfg.defaults.panel_member_timeout_s).toBe(90);
    expect(cfg.defaults.judge_timeout_s).toBe(60);
    expect(cfg.defaults.min_panel_success).toBe(1);
    const m = cfg.models["fast-glm"];
    expect(m?.strategy).toBe("single");
  });

  it("rejects an unknown strategy", () => {
    expect(() =>
      parseConfig({ ...minimal, models: { x: { strategy: "banana", target: "y" } } }),
    ).toThrow();
  });

  it("rejects a single model missing target", () => {
    expect(() => parseConfig({ ...minimal, models: { x: { strategy: "single" } } })).toThrow();
  });

  it("rejects a fusion model missing judge/synth", () => {
    expect(() =>
      parseConfig({ ...minimal, models: { x: { strategy: "fusion", panel: ["a"] } } }),
    ).toThrow();
  });

  it("rejects request_timeout_s >= 182", () => {
    expect(() =>
      parseConfig({ ...minimal, upstream: { ...minimal.upstream, request_timeout_s: 182 } }),
    ).toThrow();
  });

  it("rejects unknown top-level keys", () => {
    expect(() => parseConfig({ ...minimal, surprise: true })).toThrow();
  });

  it("accepts all four strategy variants incl. smart with inline + referenced sub-strategies", () => {
    const cfg = parseConfig({
      ...minimal,
      models: {
        "fast-1": { strategy: "single", target: "deepseek-v4-pro" },
        "resilient-1": { strategy: "failover", chain: ["a", "b"] },
        "fusion-1": { strategy: "fusion", panel: ["a", "b"], judge: "a", synth: "b" },
        "smart-1": {
          strategy: "smart",
          router: "glm-5.2",
          simple: "fast-1",
          fusion: { panel: ["a"], judge: "a", synth: "b" },
        },
      },
    });
    const smart = cfg.models["smart-1"];
    expect(smart?.strategy).toBe("smart");
    if (smart?.strategy === "smart") {
      expect(smart.default).toBe("simple");
    }
  });

  it("rejects a smart model referencing an unknown sub-model name", () => {
    expect(() =>
      parseConfig({
        ...minimal,
        models: {
          "smart-1": {
            strategy: "smart",
            router: "r",
            simple: "does-not-exist",
            fusion: { panel: ["a"], judge: "a", synth: "b" },
          },
        },
      }),
    ).toThrow();
  });
});
