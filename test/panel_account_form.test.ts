// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import { mountPanel, mountPanelWithFailure, openAccountForm, saveForm, setInput, clickTab, type Panel } from "./panel_dom";

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
      accounts: [
        { id: "ollama-1", api_key_env: "OLLAMA_API_KEY", request_timeout_s: 150 },
        { id: "ollama-lexus", api_key_env: "OLLAMA_API_KEY_2" },
      ],
    },
  },
  models: {},
  server: { bind: "127.0.0.1", port: 8081 },
  upstream: { api_mode: "auto", max_concurrency: 4, request_timeout_s: 170 },
  defaults: {},
  pricing: {},
  overrides: {},
  envKnown: { OLLAMA_API_KEY: true, OLLAMA_API_KEY_2: true },
};

function accountsFromLastPut(p: Panel): Array<Record<string, unknown>> {
  const put = p.sent.filter((r) => r.method === "PUT");
  const l = put[put.length - 1];
  if (!l) throw new Error("no PUT was sent");
  const body = l.body;
  if (!body || typeof body !== "object" || !("accounts" in body)) throw new Error("no accounts in body");
  const accs = body.accounts;
  if (!Array.isArray(accs)) throw new Error("accounts is not an array");
  return accs;
}

describe("account form round-trip", () => {
  it("keeps request_timeout_s on an unrelated edit", async () => {
    panel = await mountPanel(CFG);
    await openAccountForm(panel, "ollama-cloud", "ollama-1");
    saveForm();
    await panel.flush();

    const accs = accountsFromLastPut(panel);
    expect(accs[0]).toMatchObject({ id: "ollama-1", request_timeout_s: 150 });
    // The account we did NOT edit must come through untouched.
    expect(accs[1]).toEqual({ id: "ollama-lexus", api_key_env: "OLLAMA_API_KEY_2" });
  });

  it("edits request_timeout_s", async () => {
    panel = await mountPanel(CFG);
    await openAccountForm(panel, "ollama-cloud", "ollama-1");
    setInput("Request timeout override (s)", "120");
    saveForm();
    await panel.flush();

    expect(accountsFromLastPut(panel)[0]).toMatchObject({ request_timeout_s: 120 });
  });

  it("clears request_timeout_s when blanked", async () => {
    panel = await mountPanel(CFG);
    await openAccountForm(panel, "ollama-cloud", "ollama-1");
    setInput("Request timeout override (s)", "");
    saveForm();
    await panel.flush();

    expect(accountsFromLastPut(panel)[0]).not.toHaveProperty("request_timeout_s");
  });
});

describe("account delete error handling", () => {
  it("toasts the server error when deleting an account is rejected", async () => {
    panel = await mountPanelWithFailure(CFG, (method, path) =>
      method === "PUT" && path.indexOf("admin/config/providers/") === 0
        ? { body: { error: "a provider with base_url must keep an account" } }
        : null,
    );
    clickTab("providers");
    await panel.flush();

    const row = Array.from(document.querySelectorAll("#providers-editor .accrow")).find(
      (r) => r.querySelector(".aname")?.textContent === "ollama-1",
    );
    if (!row) throw new Error("no account row 'ollama-1'");
    const delBtn = Array.from(row.querySelectorAll("button")).find((b) => b.textContent === "Delete");
    if (!delBtn) throw new Error("no Delete button on account row");
    delBtn.click();

    const confirmYes = document.getElementById("ovl-ok");
    if (!confirmYes) throw new Error("no confirm-yes button");
    confirmYes.click();
    await panel.flush();

    const errToasts = Array.from(document.querySelectorAll("#toasts .toast.err")).map((t) => t.textContent);
    expect(errToasts).toContain("a provider with base_url must keep an account");
  });
});
