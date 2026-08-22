// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import { mountPanel, openSettingsCard, saveForm, setCell, addRow, rows, formError, setRowSelect, type Panel } from "./panel_dom";

let panel: Panel | null = null;
afterEach(() => {
  if (panel) panel.unmount();
  panel = null;
});

const CFG = {
  providers: {},
  models: {},
  server: { bind: "127.0.0.1", port: 8081 },
  upstream: {
    api_mode: "auto",
    max_concurrency: 4,
    request_timeout_s: 180,
    connector_cooldown_s: 60,
    connector_down_recheck_s: 900,
    per_model_concurrency: { "deepseek-v4-pro": 5 },
  },
  defaults: {},
  pricing: {},
  overrides: {},
  envKnown: {},
};

function lastPut(p: Panel): { path: string; body: unknown } {
  const put = p.sent.filter((r) => r.method === "PUT");
  const l = put[put.length - 1];
  if (!l) throw new Error("no PUT was sent");
  return l;
}

describe("upstream settings", () => {
  it("edits per_model_concurrency as numbers", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Upstream");
    addRow("Per-model concurrency (optional)");
    setCell("Per-model concurrency (optional)", 1, 0, "deepseek-v4-flash");
    setCell("Per-model concurrency (optional)", 1, 1, "3");
    saveForm();
    await panel.flush();

    const put = lastPut(panel);
    expect(put.path).toBe("admin/config/settings/upstream");
    expect(put.body).toMatchObject({
      per_model_concurrency: { "deepseek-v4-pro": 5, "deepseek-v4-flash": 3 },
    });
  });

  it("rejects a non-numeric per-model budget", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Upstream");
    setCell("Per-model concurrency (optional)", 0, 1, "lots");
    saveForm();
    await panel.flush();

    expect(formError()).toContain("deepseek-v4-pro");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("rejects a zero per-model budget", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Upstream");
    setCell("Per-model concurrency (optional)", 0, 1, "0");
    saveForm();
    await panel.flush();

    expect(formError()).toContain("deepseek-v4-pro");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("rejects a negative per-model budget", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Upstream");
    setCell("Per-model concurrency (optional)", 0, 1, "-3");
    saveForm();
    await panel.flush();

    expect(formError()).toContain("deepseek-v4-pro");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("rejects a fractional per-model budget", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Upstream");
    setCell("Per-model concurrency (optional)", 0, 1, "2.5");
    saveForm();
    await panel.flush();

    expect(formError()).toContain("deepseek-v4-pro");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });

  it("clearing the only row drops per_model_concurrency entirely", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Upstream");
    const row0 = rows("Per-model concurrency (optional)")[0];
    if (!row0) throw new Error("no row 0 in field 'Per-model concurrency (optional)'");
    const removeBtn = row0.querySelector("button");
    if (!removeBtn) throw new Error("no remove button on row 0");
    removeBtn.click();
    saveForm();
    await panel.flush();

    const put = lastPut(panel);
    expect(put.path).toBe("admin/config/settings/upstream");
    expect(put.body).not.toHaveProperty("per_model_concurrency");
  });
});

describe("pricing settings", () => {
  it("creates a pricing entry", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Model pricing");
    addRow("Prices");
    setCell("Prices", 0, 0, "glm-5.2");
    setCell("Prices", 0, 1, "0.6");
    setCell("Prices", 0, 2, "2.2");
    saveForm();
    await panel.flush();

    const put = lastPut(panel);
    expect(put.path).toBe("admin/config/settings/pricing");
    expect(put.body).toEqual({ "glm-5.2": { input_per_mtok: 0.6, output_per_mtok: 2.2 } });
  });

  it("round-trips an existing entry unchanged", async () => {
    const priced = JSON.parse(JSON.stringify(CFG));
    priced.pricing = { "minimax-m3": { input_per_mtok: 0.3, output_per_mtok: 1.1 } };
    panel = await mountPanel(priced);
    await openSettingsCard(panel, "Model pricing");
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).toEqual({ "minimax-m3": { input_per_mtok: 0.3, output_per_mtok: 1.1 } });
  });

  it("rejects a blank price", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Model pricing");
    addRow("Prices");
    setCell("Prices", 0, 0, "glm-5.2");
    setCell("Prices", 0, 1, "0.6");
    saveForm();
    await panel.flush();

    expect(formError()).toContain("Output price");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });
});

describe("capability overrides settings", () => {
  it("creates a tri-state override", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Capability overrides");
    addRow("Overrides");
    setCell("Overrides", 0, 0, "glm-5.2");
    setRowSelect("Overrides", 0, 0, "on"); // tools
    setRowSelect("Overrides", 0, 1, "off"); // vision
    setCell("Overrides", 0, 1, "128000"); // context
    saveForm();
    await panel.flush();

    const put = lastPut(panel);
    expect(put.path).toBe("admin/config/settings/overrides");
    expect(put.body).toEqual({ "glm-5.2": { tools: true, vision: false, context: 128000 } });
  });

  it("omits a flag left unset", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Capability overrides");
    addRow("Overrides");
    setCell("Overrides", 0, 0, "glm-5.2");
    setRowSelect("Overrides", 0, 0, "on"); // tools only
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).toEqual({ "glm-5.2": { tools: true } });
  });

  it("round-trips an existing override unchanged", async () => {
    const withOv = JSON.parse(JSON.stringify(CFG));
    withOv.overrides = { "minimax-m3": { vision: true, context: 200000 } };
    panel = await mountPanel(withOv);
    await openSettingsCard(panel, "Capability overrides");
    saveForm();
    await panel.flush();

    expect(lastPut(panel).body).toEqual({ "minimax-m3": { vision: true, context: 200000 } });
  });

  it("rejects a fractional context", async () => {
    panel = await mountPanel(CFG);
    await openSettingsCard(panel, "Capability overrides");
    addRow("Overrides");
    setCell("Overrides", 0, 0, "glm-5.2");
    setCell("Overrides", 0, 1, "1000.5"); // context
    saveForm();
    await panel.flush();

    expect(formError()).toContain("must be a whole number of tokens");
    expect(panel.sent.filter((r) => r.method === "PUT")).toHaveLength(0);
  });
});
