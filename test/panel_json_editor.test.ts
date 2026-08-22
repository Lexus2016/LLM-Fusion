// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import { mountPanel, clickTab, cardButton, saveForm, field, formError, type Panel } from "./panel_dom";

let panel: Panel | null = null;
afterEach(() => {
  if (panel) panel.unmount();
  panel = null;
});

const CFG = {
  providers: {
    "ollama-cloud": {
      type: "ollama",
      base_url: "https://ollama.com",
      accounts: [{ id: "a", api_key_env: "K", extra_headers: { "x-title": "•••" } }],
    },
  },
  models: {
    "fusion-coder": {
      strategy: "fusion",
      provider: "ollama-cloud",
      panel: ["glm-5.2"],
      judge: "m",
      synth: "s",
      bineval: { enabled: true, dimensions: ["is it correct?"] },
    },
  },
  server: { bind: "127.0.0.1", port: 8081 },
  upstream: { api_mode: "auto", max_concurrency: 4, request_timeout_s: 170 },
  defaults: {},
  pricing: {},
  overrides: {},
  envKnown: { K: true },
};

function textarea(label: string): HTMLTextAreaElement {
  const t = field(label).querySelector("textarea");
  if (!t) throw new Error("field '" + label + "' has no textarea");
  return t;
}

function lastPut(p: Panel): { path: string; body: unknown } {
  const put = p.sent.filter((r) => r.method === "PUT");
  const l = put[put.length - 1];
  if (!l) throw new Error("no PUT was sent");
  return l;
}

describe("JSON editor", () => {
  it("round-trips a model verbatim", async () => {
    panel = await mountPanel(CFG);
    clickTab("models");
    await panel.flush();
    cardButton("models-editor", "fusion-coder", "JSON").click();

    expect(JSON.parse(textarea("JSON").value)).toEqual(CFG.models["fusion-coder"]);
    saveForm();
    await panel.flush();

    const put = lastPut(panel);
    expect(put.path).toBe("admin/config/models/fusion-coder");
    expect(put.body).toEqual(CFG.models["fusion-coder"]);
  });

  it("reports invalid JSON instead of sending it", async () => {
    panel = await mountPanel(CFG);
    clickTab("models");
    await panel.flush();
    cardButton("models-editor", "fusion-coder", "JSON").click();
    textarea("JSON").value = "{ nope";
    saveForm();
    await panel.flush();

    expect(formError()).toContain("Invalid JSON");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("rejects a non-object top level", async () => {
    panel = await mountPanel(CFG);
    clickTab("models");
    await panel.flush();
    cardButton("models-editor", "fusion-coder", "JSON").click();
    textarea("JSON").value = "[1,2]";
    saveForm();
    await panel.flush();

    expect(formError()).toContain("JSON object");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("passes the extra_headers mask through untouched", async () => {
    panel = await mountPanel(CFG);
    clickTab("providers");
    await panel.flush();
    cardButton("providers-editor", "ollama-cloud", "JSON").click();
    saveForm();
    await panel.flush();

    // The panel never sees real header values; the server restores them on write.
    expect(lastPut(panel).body).toEqual(CFG.providers["ollama-cloud"]);
  });
});
