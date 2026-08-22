// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import {
  mountPanel,
  openModelForm,
  saveForm,
  setSelect,
  setInput,
  clickToggle,
  clickTab,
  hasField,
  input,
  formError,
  type Panel,
} from "./panel_dom";

let panel: Panel | null = null;
afterEach(() => {
  if (panel) panel.unmount();
  panel = null;
});

/** A config whose fusion model carries keys the form does not render. */
const CFG = {
  providers: {
    "ollama-cloud": {
      type: "ollama",
      base_url: "https://ollama.com",
      accounts: [{ id: "ollama-1", api_key_env: "OLLAMA_API_KEY" }],
    },
  },
  models: {
    "fusion-coder": {
      strategy: "fusion",
      provider: "ollama-cloud",
      panel: ["glm-5.2", "deepseek-v4-pro"],
      judge: "minimax-m3",
      synth: "glm-5.2",
      tool_mode: "deliberate",
      image_describe: { enabled: true, model: "minimax-m3", max_chars: 12000, timeout_s: 60 },
    },
  },
  server: { bind: "127.0.0.1", port: 8081 },
  upstream: { api_mode: "auto", max_concurrency: 4, request_timeout_s: 170 },
  defaults: {},
  pricing: {},
  overrides: {},
  envKnown: { OLLAMA_API_KEY: true },
};

function lastPut(p: Panel): { method: string; path: string; body: unknown } {
  const put = p.sent.filter((r) => r.method === "PUT");
  const l = put[put.length - 1];
  if (!l) throw new Error("no PUT was sent (sent: " + JSON.stringify(p.sent) + ")");
  return l;
}

describe("model form round-trip", () => {
  it("keeps a key the form does not render", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "fusion-coder");
    saveForm();
    await panel.flush();

    const body = lastPut(panel).body;
    expect(body).toMatchObject({
      strategy: "fusion",
      provider: "ollama-cloud",
      image_describe: { enabled: true, model: "minimax-m3", max_chars: 12000, timeout_s: 60 },
    });
  });

  it("starts clean when the strategy changes", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "fusion-coder");
    setSelect("Strategy", "single");
    setInput("Target model", "glm-5.2");
    saveForm();
    await panel.flush();

    const body = lastPut(panel).body;
    expect(body).toEqual({ strategy: "single", provider: "ollama-cloud", target: "glm-5.2" });
  });

  it("deletes a key when its toggle is switched off", async () => {
    const withSearch = JSON.parse(JSON.stringify(CFG));
    withSearch.models["fusion-coder"].web_search = { enabled: true, max_results: 3 };
    panel = await mountPanel(withSearch);
    await openModelForm(panel, "fusion-coder");
    clickToggle("Web search grounding");
    saveForm();
    await panel.flush();

    const body = lastPut(panel).body;
    expect(body).not.toHaveProperty("web_search");
    expect(body).toHaveProperty("judge", "minimax-m3");
  });

  it("does not render request_overrides on a fusion model", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "fusion-coder");
    // FusionModelSchema is .strict() and has no request_overrides — rendering it
    // can only produce a 400. Fusion's real control is synth_request_overrides.
    expect(hasField("Request overrides (optional)")).toBe(false);
    expect(hasField("Synth request overrides (optional)")).toBe(true);
  });

  it("still renders request_overrides on a single model", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "fusion-coder");
    setSelect("Strategy", "single");
    expect(hasField("Request overrides (optional)")).toBe(true);
  });

  it("edits image_describe through the form", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "fusion-coder");
    expect(input("Describer model").value).toBe("minimax-m3");
    setInput("Max description chars", "8000");
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).toMatchObject({
      image_describe: { enabled: true, model: "minimax-m3", max_chars: 8000, timeout_s: 60 },
    });
  });

  it("deletes image_describe when the toggle goes off", async () => {
    panel = await mountPanel(CFG);
    await openModelForm(panel, "fusion-coder");
    clickToggle("Image description (vision pre-stage)");
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).not.toHaveProperty("image_describe");
  });

  it("requires a describer model when enabled", async () => {
    const noDesc = JSON.parse(JSON.stringify(CFG));
    delete noDesc.models["fusion-coder"].image_describe;
    panel = await mountPanel(noDesc);
    await openModelForm(panel, "fusion-coder");
    clickToggle("Image description (vision pre-stage)");
    saveForm();
    await panel.flush();

    expect(formError()).toContain("describer model is required");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("badges the extras on the model card", async () => {
    panel = await mountPanel(CFG);
    clickTab("models");
    await panel.flush();
    const card = Array.from(document.querySelectorAll("#models-editor .ecard")).find((c) => {
      const n = c.querySelector(".ename");
      return n !== null && n.textContent === "fusion-coder";
    });
    if (!card) throw new Error("no card");
    const badges = Array.from(card.querySelectorAll(".badge")).map((b) => b.textContent);
    expect(badges).toContain("vision pre-stage");
  });
});
