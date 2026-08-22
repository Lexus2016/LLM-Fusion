// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import { mountPanel, openSettingsCard, saveForm, setCell, addRow, formError, type Panel } from "./panel_dom";

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
});
