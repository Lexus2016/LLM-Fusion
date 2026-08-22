// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import {
  mountPanel,
  openModelForm,
  saveForm,
  clickToggle,
  input,
  type Panel,
} from "./panel_dom";

let panel: Panel | null = null;
afterEach(() => {
  if (panel) panel.unmount();
  panel = null;
});

const CFG = {
  providers: {
    "ollama-cloud": { type: "ollama", base_url: "https://ollama.com", accounts: [{ id: "a", api_key_env: "K" }] },
  },
  models: {
    "smart-named": {
      strategy: "smart",
      provider: "ollama-cloud",
      router: "minimax-m3",
      default: "simple",
      escalate_on_tool_error: false,
      simple: "fast-glm",
      fusion: "fusion-coder",
    },
    "smart-inline": {
      strategy: "smart",
      provider: "ollama-cloud",
      router: "minimax-m3",
      default: "simple",
      simple: { target: "glm-5.2" },
      fusion: { panel: ["glm-5.2"], judge: "minimax-m3", synth: "glm-5.2" },
    },
    "fast-glm": { strategy: "single", provider: "ollama-cloud", target: "glm-5.2" },
    "fusion-coder": { strategy: "fusion", provider: "ollama-cloud", panel: ["glm-5.2"], judge: "m", synth: "s" },
  },
  server: { bind: "127.0.0.1", port: 8081 },
  upstream: { api_mode: "auto", max_concurrency: 4, request_timeout_s: 170 },
  defaults: {},
  pricing: {},
  overrides: {},
  envKnown: {},
};

function lastPut(p: Panel): { body: unknown } {
  const put = p.sent.filter((r) => r.method === "PUT");
  const l = put[put.length - 1];
  if (!l) throw new Error("no PUT was sent");
  return l;
}

describe("smart form", () => {
  it("round-trips escalate_on_tool_error: false", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "smart-named");
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).toMatchObject({ escalate_on_tool_error: false });
  });

  it("omits escalate_on_tool_error when it is on (the schema default)", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "smart-named");
    clickToggle("Escalate on tool error");
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).not.toHaveProperty("escalate_on_tool_error");
  });

  it("preserves inline simple/fusion blocks and disables their inputs", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "smart-inline");
    expect(input("Simple route").disabled).toBe(true);
    expect(input("Fusion route").disabled).toBe(true);
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).toMatchObject({
      simple: { target: "glm-5.2" },
      fusion: { panel: ["glm-5.2"], judge: "minimax-m3", synth: "glm-5.2" },
    });
  });
});
