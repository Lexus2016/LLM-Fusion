# Panel Config Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `ConfigSchema` field reachable from the panel and stop panel saves from silently deleting config keys the form does not render.

**Architecture:** The panel is one self-contained HTML+JS string (`PANEL_HTML` in `src/panel/page.ts`). The root defect is that its save handlers *rebuild* the object from form fields instead of *editing the object they loaded*, so unrendered keys are dropped. We flip that to merge-on-save, add the missing controls, and add a raw-JSON escape hatch so anything the form cannot express stays editable. A new jsdom test harness runs the shipped panel JS in a real DOM and asserts on captured PUT bodies, so the next schema addition fails loudly.

**Tech Stack:** TypeScript 5.9 (`tsc --noEmit`, no build step — the server is run through `tsx`), Hono, zod 3, `yaml` Document API, vitest 2.1, jsdom (new devDependency).

**Spec:** `docs/superpowers/specs/2026-08-22-panel-config-coverage.md`

## Global Constraints

- **The panel JS must not use template literals.** `PANEL_HTML` is itself a template literal; a backtick inside it terminates the string. Use `"a"+b+"c"` concatenation. Same for `${`.
- **The panel JS is ES5-flavoured** by house style: `var`, `function`, no arrow functions, no `const`/`let`, no `Object.entries`, no optional chaining. Match the surrounding code exactly.
- **No `as` casts in TypeScript** (user standard). Narrow with `instanceof` or tag-specific `querySelector` overloads instead.
- **`.strict()` schemas.** `ModelSchema` is a `z.discriminatedUnion("strategy", …)` of `.strict()` objects — an unknown or foreign-strategy key is a hard 400, not a warning.
- **`FusionModelSchema` has no `request_overrides`.** It has `synth_request_overrides`. Do not confuse them.
- **`request_timeout_s` / `timeout_s` upper bound is `< 182`** everywhere (the Ollama Cloud ceiling).
- **Never render a secret.** The config stores env-var *names*. `extra_headers` values arrive from `GET /admin/config` masked as `•••` and `restoreExtraHeaders` in `src/panel/config_editor.ts` puts the real values back on write — pass the mask through untouched, never strip it.
- **No server changes are needed.** `EDITABLE_SECTIONS` in `src/panel/config_editor.ts:` already contains `server`, `upstream`, `defaults`, `pricing`, `overrides`.
- **Every task ends green:** `npm run test` and `npm run typecheck` both pass before the commit.

---

### Task 1: jsdom harness + merge-on-save for models

The mechanism fix, plus the test infrastructure that proves it. These ship together because the fix is untestable without the harness and the harness is pointless without a first assertion.

**Files:**
- Modify: `package.json` (add `jsdom` devDependency)
- Modify: `tsconfig.json` (add `lib`)
- Create: `test/panel_dom.ts` (harness, not a test file)
- Create: `test/panel_model_form.test.ts`
- Modify: `src/panel/page.ts` (the model form's save handler, ~line 684)

**Interfaces:**
- Consumes: `PANEL_HTML` from `src/panel/page.ts`
- Produces (used by Tasks 2–9):
  - `mountPanel(cfg: unknown): Promise<Panel>` where `Panel = { sent: SentRequest[]; flush(): Promise<void>; unmount(): void }`
  - `SentRequest = { method: string; path: string; body: unknown }`
  - `clickTab(name: string): void`
  - `cardButton(container: string, name: string, label: string): HTMLButtonElement`
  - `field(label: string): Element`, `input(label: string): HTMLInputElement`, `select(label: string): HTMLSelectElement`
  - `setInput(label: string, value: string): void`, `setSelect(label: string, value: string): void`
  - `clickToggle(label: string): void`
  - `saveForm(): void`, `formError(): string`

- [ ] **Step 1: Install jsdom and widen the TS lib**

```bash
npm install --save-dev jsdom
```

Then edit `tsconfig.json` — add a `lib` line right after `"target"`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable", "DOM.AsyncIterable"],
    "module": "ESNext",
```

`DOM` alone is not enough and *breaks* the existing build: `src/anthropic.ts` calls `Headers.entries()` (declared in `DOM.Iterable`) and `test/reasoning.test.ts` async-iterates a `ReadableStream` (declared in `DOM.AsyncIterable`). All three entries are required. Do not add `@types/jsdom` — the harness never imports jsdom directly, it uses the globals vitest's jsdom environment installs.

- [ ] **Step 2: Verify the tsconfig change is neutral**

Run: `npm run typecheck`
Expected: PASS, no output. (If you see `Property 'entries' does not exist on type 'Headers'`, you missed `DOM.Iterable`.)

- [ ] **Step 3: Write the harness**

Create `test/panel_dom.ts`:

```ts
import { vi } from "vitest";
import { PANEL_HTML } from "../src/panel/page";

/** A non-GET request the panel issued, captured by the fetch stub. */
export interface SentRequest {
  method: string;
  path: string;
  body: unknown;
}

export interface Panel {
  /** Every non-GET request, in order. GETs are answered but not recorded. */
  sent: SentRequest[];
  /** Drain the microtask queue so a fetch chain settles. */
  flush(): Promise<void>;
  unmount(): void;
}

const BODY_RE = /<body>([\s\S]*)<\/body>/;
const SCRIPT_RE = /<script>([\s\S]*?)<\/script>/;

function extract(re: RegExp, what: string): string {
  const m = PANEL_HTML.match(re);
  if (!m || m[1] === undefined) throw new Error("could not extract " + what + " from PANEL_HTML");
  return m[1];
}

/**
 * Mount the SHIPPED panel markup + script into jsdom and stub the network.
 *
 * The point of running the real string (rather than a copy of the logic) is that
 * a schema field the form forgets to round-trip must fail HERE, in a test, not
 * in fusion.yaml. Keep it black-box: drive the panel through DOM clicks only.
 */
export async function mountPanel(cfg: unknown): Promise<Panel> {
  // Only setInterval is faked: the 3s monitor poll would otherwise fire for the
  // whole test run. setTimeout stays REAL so `flush()` and the panel's own
  // reloadConfigSoon(400ms) behave normally.
  vi.useFakeTimers({ toFake: ["setInterval"] });

  const markup = extract(BODY_RE, "<body>");
  const script = extract(SCRIPT_RE, "<script>");
  document.body.innerHTML = markup.replace(SCRIPT_RE, "");

  const sent: SentRequest[] = [];
  vi.stubGlobal("fetch", (path: string, opt?: { method?: string; body?: string }) => {
    const method = (opt && opt.method) || "GET";
    if (method !== "GET") {
      sent.push({ method, path, body: opt && opt.body ? JSON.parse(opt.body) : undefined });
      return Promise.resolve(fakeResponse({ ok: true }));
    }
    if (path === "admin/config") return Promise.resolve(fakeResponse(cfg));
    if (path === "admin/providers") return Promise.resolve(fakeResponse([]));
    if (path.indexOf("/models") >= 0) return Promise.resolve(fakeResponse({ models: [] }));
    return Promise.resolve(fakeResponse({}));
  });

  // The panel script is an IIFE that wires itself to the DOM on evaluation.
  new Function(script)();
  await flush();
  return { sent, flush, unmount };
}

/** Minimal Response stand-in — the panel only reads .ok, .status and .json(). */
function fakeResponse(data: unknown): { ok: boolean; status: number; json(): Promise<unknown> } {
  return { ok: true, status: 200, json: () => Promise.resolve(data) };
}

function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function unmount(): void {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.body.innerHTML = "";
}

function asHtml(n: Element | null, what: string): HTMLElement {
  if (!(n instanceof HTMLElement)) throw new Error("not an HTMLElement: " + what);
  return n;
}

function byId(id: string): HTMLElement {
  return asHtml(document.getElementById(id), "#" + id);
}

export function clickTab(name: string): void {
  byId("tabbtn-" + name).click();
}

/**
 * The Nth `.ecard` in `container` whose `.ename` is `name`, then the action
 * button labelled `label`. Cards are rebuilt on every render, so never hold on
 * to a node across a save.
 */
export function cardButton(container: string, name: string, label: string): HTMLButtonElement {
  const cards = Array.from(byId(container).querySelectorAll(".ecard"));
  const card = cards.find((c) => {
    const n = c.querySelector(".ename");
    return n !== null && n.textContent === name;
  });
  if (!card) {
    const have = cards.map((c) => (c.querySelector(".ename") || {}).textContent).join(", ");
    throw new Error("no card '" + name + "' in #" + container + " (have: " + have + ")");
  }
  const btn = Array.from(card.querySelectorAll("button")).find((b) => b.textContent === label);
  if (!btn) throw new Error("no '" + label + "' button on card '" + name + "'");
  return btn;
}

export function field(label: string): Element {
  const all = Array.from(document.querySelectorAll("#fovl-body .fld"));
  const found = all.find((f) => {
    const l = f.querySelector("label");
    return l !== null && l.textContent === label;
  });
  if (!found) {
    const have = all.map((f) => (f.querySelector("label") || {}).textContent).join(" | ");
    throw new Error("no field '" + label + "' (have: " + have + ")");
  }
  return found;
}

/** True when the form renders NO field with this label. */
export function hasField(label: string): boolean {
  try {
    field(label);
    return true;
  } catch {
    return false;
  }
}

export function input(label: string): HTMLInputElement {
  const i = field(label).querySelector("input");
  if (!i) throw new Error("field '" + label + "' has no input");
  return i;
}

export function select(label: string): HTMLSelectElement {
  const s = field(label).querySelector("select");
  if (!s) throw new Error("field '" + label + "' has no select");
  return s;
}

export function setInput(label: string, value: string): void {
  const i = input(label);
  i.value = value;
  i.dispatchEvent(new Event("input"));
}

export function setSelect(label: string, value: string): void {
  const s = select(label);
  s.value = value;
  s.dispatchEvent(new Event("change"));
}

export function clickToggle(label: string): void {
  asHtml(field(label), "toggle " + label).click();
}

export function saveForm(): void {
  byId("fovl-save").click();
}

export function formError(): string {
  return byId("fovl-err").textContent || "";
}

/** Load the config, open the Models tab, click Edit on `name`. */
export async function openModelForm(panel: Panel, name: string): Promise<void> {
  clickTab("models");
  await panel.flush();
  cardButton("models-editor", name, "Edit").click();
}

/** Load the config, open the Providers tab, click Edit on account `accId`. */
export async function openAccountForm(panel: Panel, provId: string, accId: string): Promise<void> {
  clickTab("providers");
  await panel.flush();
  const cards = Array.from(document.querySelectorAll("#providers-editor .ecard"));
  const card = cards.find((c) => {
    const n = c.querySelector(".ename");
    return n !== null && n.textContent === provId;
  });
  if (!card) throw new Error("no provider card '" + provId + "'");
  const rows = Array.from(card.querySelectorAll(".accrow"));
  const row = rows.find((r) => {
    const n = r.querySelector(".aname");
    return n !== null && n.textContent === accId;
  });
  if (!row) throw new Error("no account row '" + accId + "'");
  const btn = Array.from(row.querySelectorAll("button")).find((b) => b.textContent === "Edit");
  if (!btn) throw new Error("no Edit button on account '" + accId + "'");
  btn.click();
}
```

- [ ] **Step 4: Write the failing test**

Create `test/panel_model_form.test.ts`:

```ts
// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import {
  mountPanel,
  openModelForm,
  saveForm,
  setSelect,
  setInput,
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
});
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `npx vitest run test/panel_model_form.test.ts`
Expected: the first test FAILS — the PUT body has no `image_describe` key. The second test should already PASS (it documents the behaviour we must not break).

- [ ] **Step 6: Implement merge-on-save**

In `src/panel/page.ts`, in `modelForm`'s save callback, replace:

```js
      var strat=fStrat._get(); var obj={ strategy:strat }; var prov=fProv._get(); if(prov) obj.provider=prov;
      // Collect the per-model promote override (tri-state) + request_overrides onto obj.
      function applyCommon(o){ if(dyn.promote){ var pv=dyn.promote._get(); if(pv==="on") o.promote_reasoning_to_content=true; else if(pv==="off") o.promote_reasoning_to_content=false; }
        if(dyn.overrides){ var ov=dyn.overrides._get(); if(Object.keys(ov).length) o.request_overrides=ov; } }
```

with:

```js
      var strat=fStrat._get();
      // MERGE-ON-SAVE: start from the model we LOADED, so a key this form does
      // not render (image_describe, or whatever the schema grows next) rides
      // along instead of being silently deleted from fusion.yaml. Rebuilding
      // from scratch is what lost web_search/bineval in v0.1.32 — same class.
      // A strategy SWITCH starts clean: ModelSchema is a .strict() discriminated
      // union, so the previous strategy's keys would be rejected outright.
      // The trade: every optional control must now DELETE its key explicitly.
      var obj=(existing&&existing.strategy===strat)?JSON.parse(JSON.stringify(existing)):{};
      obj.strategy=strat;
      var prov=fProv._get(); if(prov) obj.provider=prov; else delete obj.provider;
      // Collect the per-model promote override (tri-state) + request_overrides onto obj.
      function applyCommon(o){ if(dyn.promote){ var pv=dyn.promote._get(); if(pv==="on") o.promote_reasoning_to_content=true; else if(pv==="off") o.promote_reasoning_to_content=false; else delete o.promote_reasoning_to_content; }
        if(dyn.overrides){ var ov=dyn.overrides._get(); if(Object.keys(ov).length) o.request_overrides=ov; else delete o.request_overrides; } }
```

Then, in the same save callback, make each optional fusion control delete its key. Replace:

```js
        obj.panel=panel; obj.judge=judge; obj.synth=synth; var adv=dyn.adv._get(); if(adv) obj.adversarial=adv;
        obj.tool_mode=dyn.tool._get(); if(dyn.planOnly._get()) obj.fusion_planning_turn_only=true;
```

with:

```js
        obj.panel=panel; obj.judge=judge; obj.synth=synth; var adv=dyn.adv._get(); if(adv) obj.adversarial=adv; else delete obj.adversarial;
        obj.tool_mode=dyn.tool._get(); if(dyn.planOnly._get()) obj.fusion_planning_turn_only=true; else delete obj.fusion_planning_turn_only;
```

Replace `          obj.web_search=ws; }` with:

```js
          obj.web_search=ws; }
        else delete obj.web_search;
```

Replace `          obj.bineval=be; }` with:

```js
          obj.bineval=be; }
        else delete obj.bineval;
```

Replace:

```js
        if(dyn.synthOverrides){ var so=dyn.synthOverrides._get(); if(Object.keys(so).length) obj.synth_request_overrides=so; }
```

with:

```js
        if(dyn.synthOverrides){ var so=dyn.synthOverrides._get(); if(Object.keys(so).length) obj.synth_request_overrides=so; else delete obj.synth_request_overrides; }
```

- [ ] **Step 7: Run the tests**

Run: `npx vitest run test/panel_model_form.test.ts`
Expected: both PASS.

- [ ] **Step 8: Add the turn-it-off test**

Append to the `describe` block in `test/panel_model_form.test.ts`:

```ts
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
```

Add `clickToggle` to the import list at the top of the file.

- [ ] **Step 9: Run the full suite**

Run: `npm run test && npm run typecheck`
Expected: both PASS.

- [ ] **Step 10: Commit**

```bash
git add package.json package-lock.json tsconfig.json test/panel_dom.ts test/panel_model_form.test.ts src/panel/page.ts
git commit -m "fix(panel): merge model edits into the loaded object instead of rebuilding

Saving a model rebuilt the object from the rendered fields only, so every
config key the form does not know about (image_describe, and whatever the
schema grows next) was deleted from fusion.yaml on an unrelated edit.

Adds a jsdom harness that runs the shipped PANEL_HTML script, so the next
unrendered field fails a test instead of a production config."
```

---

### Task 2: merge-on-save for accounts + `request_timeout_s`

**Files:**
- Modify: `src/panel/page.ts` (`accountForm`, ~lines 565–588)
- Create: `test/panel_account_form.test.ts`

**Interfaces:**
- Consumes: `mountPanel`, `openAccountForm`, `saveForm`, `setInput`, `Panel` from `test/panel_dom.ts` (Task 1)
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

Create `test/panel_account_form.test.ts`:

```ts
// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import { mountPanel, openAccountForm, saveForm, setInput, type Panel } from "./panel_dom";

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_account_form.test.ts`
Expected: all three FAIL — the first because `request_timeout_s` is dropped, the other two because there is no such field.

- [ ] **Step 3: Add the field**

In `src/panel/page.ts`, in `accountForm`, change the declaration line:

```js
    var p=cfg.providers[provId]; var fId,fEnv,fBase,fMap,f403,fQuota;
```

to:

```js
    var p=cfg.providers[provId]; var fId,fEnv,fBase,fTimeout,fMap,f403,fQuota;
```

Then, after the `fBase=fText(...)` line, insert:

```js
      fTimeout=fNum("Request timeout override (s)","Per-request deadline for THIS account only. Blank = inherit the global upstream request timeout. Must stay below ~182s (the Ollama Cloud ceiling).", existing?existing.request_timeout_s:undefined);
```

and add it to the append chain:

```js
      body.appendChild(fId); body.appendChild(fEnv); body.appendChild(fBase); body.appendChild(fTimeout); body.appendChild(fMap); body.appendChild(f403); body.appendChild(fQuota);
```

- [ ] **Step 4: Implement merge-on-save**

Replace:

```js
      var acc={ id:nid, api_key_env:env }; var b=fBase._get(); if(b) acc.base_url=b;
      var m=fMap._get(); if(Object.keys(m).length) acc.model_map=m; var q=fQuota._get(); if(q.length) acc.quota_markers=q;
      var t=f403._get(); if(t!=="passthrough") acc.treat_403_as=t;
```

with:

```js
      // MERGE-ON-SAVE (see modelForm): edit the account we LOADED so unrendered
      // keys survive. extra_headers rides through with its values still masked
      // as ••• — restoreExtraHeaders() on the server swaps the real ones back in.
      var acc=existing?JSON.parse(JSON.stringify(existing)):{};
      acc.id=nid; acc.api_key_env=env;
      var b=fBase._get(); if(b) acc.base_url=b; else delete acc.base_url;
      var rt=fTimeout._get();
      if(rt===undefined) delete acc.request_timeout_s;
      else if(isNaN(rt)){ formError("Request timeout override must be a number."); return; }
      else acc.request_timeout_s=rt;
      var m=fMap._get(); if(Object.keys(m).length) acc.model_map=m; else delete acc.model_map;
      var q=fQuota._get(); if(q.length) acc.quota_markers=q; else delete acc.quota_markers;
      var t=f403._get(); if(t!=="passthrough") acc.treat_403_as=t; else delete acc.treat_403_as;
```

- [ ] **Step 5: Run the tests**

Run: `npx vitest run test/panel_account_form.test.ts`
Expected: all three PASS.

- [ ] **Step 6: Commit**

```bash
git add src/panel/page.ts test/panel_account_form.test.ts
git commit -m "fix(panel): merge account edits + expose accounts[].request_timeout_s"
```

---

### Task 3: remove the unsavable `request_overrides` field from the fusion form

`FusionModelSchema` is `.strict()` and has no `request_overrides` — the field the fusion form renders can only ever produce a 400. Its real counterpart, `synth_request_overrides`, is already rendered below it.

**Files:**
- Modify: `src/panel/page.ts` (one call site, ~line 662)
- Modify: `test/panel_model_form.test.ts`

**Interfaces:**
- Consumes: `hasField` from `test/panel_dom.ts` (Task 1)
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

Append to the `describe` block in `test/panel_model_form.test.ts`:

```ts
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
```

Add `hasField` to the import list.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_model_form.test.ts -t "request_overrides"`
Expected: the fusion test FAILS (`expected true to be false`); the single test passes.

- [ ] **Step 3: Drop the call**

In `src/panel/page.ts`, in the `strat==="fusion"` branch, replace:

```js
          addPromote(h, ex); addOverrides(h, ex);
```

with:

```js
          // NO addOverrides() here: FusionModelSchema is .strict() and has no
          // request_overrides — the fusion strategy ignores it. The synth-only
          // control below is the real one.
          addPromote(h, ex);
```

Leave the `strat==="single"` branch untouched — `SingleModelSchema` does have `request_overrides`.

- [ ] **Step 4: Run the tests**

Run: `npx vitest run test/panel_model_form.test.ts`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/panel/page.ts test/panel_model_form.test.ts
git commit -m "fix(panel): drop request_overrides from the fusion form (schema rejects it)"
```

---

### Task 4: `image_describe` controls on the fusion form

The vision pre-stage the user reported missing. Mirrors the existing `web_search` / `bineval` toggle + `subGroup()` + `bindReveal()` pattern exactly.

**Files:**
- Modify: `src/panel/page.ts` (fusion branch of `buildStrategyFields`, and the fusion save branch)
- Modify: `test/panel_model_form.test.ts`

**Interfaces:**
- Consumes: `mountPanel`, `openModelForm`, `clickToggle`, `setInput`, `saveForm`, `formError`, `hasField`
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

Append to the `describe` block in `test/panel_model_form.test.ts`:

```ts
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
```

Add `input` and `formError` to the import list.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_model_form.test.ts -t "image_describe"`
Expected: FAIL — `no field 'Describer model'`.

- [ ] **Step 3: Add the controls**

In `src/panel/page.ts`, in the fusion branch, immediately after the BinEval block's `h.appendChild(bsub); bindReveal(dyn.bineval, bsub);` line and before `addPromote(h, ex);`, insert:

```js
          // Vision pre-stage. Each image_url block is described ONCE by a
          // multimodal model and replaced in place with that text, so no panel
          // member needs vision at all. All-or-nothing: any describer failure
          // falls the whole request back to the legacy per-member vision gate.
          var idc=(ex&&ex.image_describe)||{};
          dyn.imgDesc=fToggle("Image description (vision pre-stage)","Describe every image once with a multimodal model and splice that text into the prompt, so text-only panel members can still answer. On any describer failure the whole request falls back to the per-member vision gate.", !!idc.enabled); h.appendChild(dyn.imgDesc);
          var isub=subGroup();
          dyn.idModel=fText("Describer model","A multimodal model that turns each image into text (e.g. minimax-m3). Required while this is on.", idc.model, true, up); isub.appendChild(dyn.idModel);
          dyn.idChars=fNum("Max description chars","Cap on one image's description before it is truncated. Default 12000.", idc.max_chars); isub.appendChild(dyn.idChars);
          dyn.idTimeout=fNum("Describe timeout (s)","Per-image deadline; must stay below 182. Default 60.", idc.timeout_s); isub.appendChild(dyn.idTimeout);
          h.appendChild(isub); bindReveal(dyn.imgDesc, isub);
```

- [ ] **Step 4: Add the save branch**

In the fusion save branch, immediately after the `else delete obj.bineval;` line added in Task 1 and before `applyCommon(obj);`, insert:

```js
        if(dyn.imgDesc._get()){ var idm=dyn.idModel._get(); if(!idm){ formError("A describer model is required when image description is on."); return; }
          var idobj={ enabled:true, model:idm };
          var idc2=dyn.idChars._get(); if(idc2!==undefined){ if(isNaN(idc2)){ formError("Max description chars must be a number."); return; } idobj.max_chars=idc2; }
          var idt=dyn.idTimeout._get(); if(idt!==undefined){ if(isNaN(idt)){ formError("Describe timeout must be a number."); return; } idobj.timeout_s=idt; }
          obj.image_describe=idobj; }
        else delete obj.image_describe;
```

- [ ] **Step 5: Run the tests**

Run: `npm run test && npm run typecheck`
Expected: both PASS. Task 1's "keeps a key the form does not render" test now passes for a *different* reason — the field is rendered. That is fine; it still guards the merge for future keys.

- [ ] **Step 6: Commit**

```bash
git add src/panel/page.ts test/panel_model_form.test.ts
git commit -m "feat(panel): edit image_describe (vision pre-stage) from the fusion form"
```

---

### Task 5: smart — `escalate_on_tool_error` + inline `simple`/`fusion` blocks

`SmartModelSchema.simple` and `.fusion` accept either a model NAME or an inline block. The form only handles the string form and demands a name, so opening a smart model with inline blocks and saving either wipes them or blocks the save outright.

**Files:**
- Modify: `src/panel/page.ts` (smart branch of `buildStrategyFields`, and the smart save branch)
- Create: `test/panel_smart_form.test.ts`

**Interfaces:**
- Consumes: harness exports from Task 1
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

Create `test/panel_smart_form.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_smart_form.test.ts`
Expected: all three FAIL — no `Escalate on tool error` field, and the inline inputs are enabled + blank.

- [ ] **Step 3: Add the controls**

In `src/panel/page.ts`, replace the whole `else if(strat==="smart"){ … }` block in `buildStrategyFields` with:

```js
        else if(strat==="smart"){
          dyn.router=fText("Router model","A fast model that classifies each request as simple vs deep (needs reliable JSON). Pick from the provider's list.", ex&&ex.router, true, up); h.appendChild(dyn.router);
          dyn.def=fSelect("Default route","Used when the router is unsure or errors.", ex?ex.default:"simple",[{label:"simple",value:"simple"},{label:"fusion",value:"fusion"}]); h.appendChild(dyn.def);
          dyn.escalate=fToggle("Escalate on tool error","On: if the simple route's tool call fails, retry that step on the fusion route. Off: surface the failure to the client.", ex?(ex.escalate_on_tool_error==null?true:!!ex.escalate_on_tool_error):true); h.appendChild(dyn.escalate);
          // simple/fusion accept either a model NAME or an INLINE block. The form
          // only edits names; an inline block is shown read-only and rides through
          // the save untouched (merge-on-save already carries it on `obj`).
          var sInline=!!(ex&&ex.simple&&typeof ex.simple==="object");
          var fInline=!!(ex&&ex.fusion&&typeof ex.fusion==="object");
          var INLINE_HINT=' Defined INLINE in fusion.yaml — open the model card\'s "JSON" button to edit it. Saving this form leaves the block as-is.';
          dyn.simple=fText("Simple route", sInline?INLINE_HINT:"Name of a single/failover model to use for cheap steps (a model from the Models list).", sInline?"":(ex&&typeof ex.simple==="string"?ex.simple:""), true, virt);
          if(sInline){ var si=dyn.simple.querySelector("input"); si.disabled=true; si.placeholder="(inline block)"; }
          h.appendChild(dyn.simple);
          dyn.fusion=fText("Fusion route", fInline?INLINE_HINT:"Name of a fusion model to use for deep steps (must be in the same provider group).", fInline?"":(ex&&typeof ex.fusion==="string"?ex.fusion:""), true, virt);
          if(fInline){ var fi=dyn.fusion.querySelector("input"); fi.disabled=true; fi.placeholder="(inline block)"; }
          h.appendChild(dyn.fusion);
        }
```

- [ ] **Step 4: Rewrite the smart save branch**

Replace:

```js
      else if(strat==="smart"){ var router=dyn.router._get(); if(!router){ formError("Router model is required."); return; }
        obj.router=router; obj.default=dyn.def._get(); var s=dyn.simple._get(), fu=dyn.fusion._get();
        if(!s||!fu){ formError("Simple and Fusion route model names are required."); return; } obj.simple=s; obj.fusion=fu;
      }
```

with:

```js
      else if(strat==="smart"){ var router=dyn.router._get(); if(!router){ formError("Router model is required."); return; }
        obj.router=router; obj.default=dyn.def._get();
        // `true` is the schema default — omit it so the YAML stays minimal, and
        // only write the key when the operator turns escalation OFF.
        if(dyn.escalate._get()) delete obj.escalate_on_tool_error; else obj.escalate_on_tool_error=false;
        // A blank input is only acceptable when `obj` already carries an INLINE
        // block for that route (merge-on-save kept it); otherwise a name is required.
        var s=dyn.simple._get(); if(s) obj.simple=s;
        else if(!obj.simple||typeof obj.simple!=="object"){ formError("Simple route model name is required."); return; }
        var fu=dyn.fusion._get(); if(fu) obj.fusion=fu;
        else if(!obj.fusion||typeof obj.fusion!=="object"){ formError("Fusion route model name is required."); return; }
      }
```

- [ ] **Step 5: Run the tests**

Run: `npx vitest run test/panel_smart_form.test.ts && npm run typecheck`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/panel/page.ts test/panel_smart_form.test.ts
git commit -m "feat(panel): smart form gains escalate_on_tool_error and preserves inline routes"
```

---

### Task 6: `upstream.per_model_concurrency` in the Upstream form

`upstreamForm` already does `Object.assign({}, up)` so the key survives — it just cannot be *edited*. `fKV` yields strings; `per_model_concurrency` is `z.record(z.string(), z.number().int().positive())`, so the values must be parsed.

**Files:**
- Modify: `src/panel/page.ts` (`upstreamForm`)
- Create: `test/panel_settings_form.test.ts`

**Interfaces:**
- Consumes: harness exports from Task 1
- Produces: `openSettingsCard(panel, title)` — added to `test/panel_dom.ts` in Step 1 below, used by Tasks 7 and 8

- [ ] **Step 1: Add a settings-card opener to the harness**

Append to `test/panel_dom.ts`:

```ts
/** Load the config, open the Settings tab, click Edit on the named card. */
export async function openSettingsCard(panel: Panel, title: string): Promise<void> {
  clickTab("settings");
  await panel.flush();
  cardButton("settings-editor", title, "Edit").click();
}

/** The Nth row of a multi-column rows editor inside the open form. */
export function rows(label: string): Element[] {
  return Array.from(field(label).querySelectorAll(".kv"));
}

/** Click the "+ Add …" button of a rows editor. */
export function addRow(label: string): void {
  const btns = Array.from(field(label).querySelectorAll("button"));
  const add = btns.find((b) => (b.textContent || "").indexOf("+ Add") === 0);
  if (!add) throw new Error("no '+ Add' button in field '" + label + "'");
  add.click();
}

/** Set the Nth input of the Nth row of a rows editor. */
export function setCell(label: string, row: number, col: number, value: string): void {
  const r = rows(label)[row];
  if (!r) throw new Error("no row " + row + " in field '" + label + "'");
  const inputs = Array.from(r.querySelectorAll("input"));
  const cell = inputs[col];
  if (!cell) throw new Error("no input column " + col + " in row " + row);
  cell.value = value;
  cell.dispatchEvent(new Event("input"));
}

/** Set the Nth select of the Nth row of a rows editor. */
export function setRowSelect(label: string, row: number, col: number, value: string): void {
  const r = rows(label)[row];
  if (!r) throw new Error("no row " + row + " in field '" + label + "'");
  const selects = Array.from(r.querySelectorAll("select"));
  const cell = selects[col];
  if (!cell) throw new Error("no select column " + col + " in row " + row);
  cell.value = value;
  cell.dispatchEvent(new Event("change"));
}
```

- [ ] **Step 2: Write the failing test**

Create `test/panel_settings_form.test.ts`:

```ts
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run test/panel_settings_form.test.ts`
Expected: FAIL — `no field 'Per-model concurrency (optional)'`.

- [ ] **Step 4: Add the field**

In `src/panel/page.ts`, in `upstreamForm`, change the declaration:

```js
  function upstreamForm(up){ var fMode,fConc,fTimeout,fCool,fRecheck,fPerDef;
```

to:

```js
  function upstreamForm(up){ var fMode,fConc,fTimeout,fCool,fRecheck,fPerDef,fPerModel;
```

After the `fPerDef=fNum(...)` line, insert:

```js
      fPerModel=fKV("Per-model concurrency (optional)","A per-model budget that overrides the default above, e.g. deepseek-v4-pro → 5. Each value must be a whole number above 0. Models not listed use the default.", up.per_model_concurrency||{});
```

and extend the append chain:

```js
      body.appendChild(fMode); body.appendChild(fConc); body.appendChild(fTimeout); body.appendChild(fCool); body.appendChild(fRecheck); body.appendChild(fPerDef); body.appendChild(fPerModel);
```

- [ ] **Step 5: Parse it on save**

In `upstreamForm`'s save callback, immediately before the `saveSettings("upstream", …)` line, insert:

```js
      // fKV yields strings; the schema wants positive integers.
      var pmRaw=fPerModel._get(); var pm={}; var pmBad=null;
      Object.keys(pmRaw).forEach(function(k){ var n=Number(pmRaw[k]); if(pmRaw[k]===""||!isFinite(n)) pmBad=k; else pm[k]=n; });
      if(pmBad){ formError("Per-model concurrency for '"+pmBad+"' must be a number."); return; }
      if(Object.keys(pm).length) obj.per_model_concurrency=pm; else delete obj.per_model_concurrency;
```

Also update the `Object.assign` comment two lines up, since the form now edits that key:

```js
      var obj=Object.assign({},up); // preserve base_url/api_key_env the form doesn't edit
```

- [ ] **Step 6: Run the tests**

Run: `npx vitest run test/panel_settings_form.test.ts && npm run typecheck`
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add src/panel/page.ts test/panel_dom.ts test/panel_settings_form.test.ts
git commit -m "feat(panel): edit upstream.per_model_concurrency from the Upstream form"
```

---

### Task 7: Pricing settings card

`GET /admin/config` already returns `pricing`, and `PUT /admin/config/settings/pricing` already works — only the UI is missing. `PricingEntrySchema` is `{ input_per_mtok: number>=0, output_per_mtok: number>=0 }`, so this needs a three-column rows editor; `fKV` is key→one value.

**Files:**
- Modify: `src/panel/page.ts` (new `fPricing` builder, new `pricingForm`, one line in `renderSettings`, one CSS line)
- Modify: `test/panel_settings_form.test.ts`

**Interfaces:**
- Consumes: `fld`, `el`, `openForm`, `closeForm`, `formError`, `saveSettings`, `settingsCard` (all existing in `page.ts`); `openSettingsCard`, `setCell`, `addRow` from Task 6
- Produces: `fPricing(label, hint, obj)` — a field whose `_get()` returns `Array<[key, inStr, outStr]>`

- [ ] **Step 1: Write the failing test**

Append to `test/panel_settings_form.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_settings_form.test.ts -t pricing`
Expected: FAIL — `no card 'Model pricing' in #settings-editor`.

- [ ] **Step 3: Let a `.kv` row hold selects too**

In the CSS block of `PANEL_HTML`, change:

```css
  .kv{display:flex; gap:6px} .kv input{flex:1}
```

to:

```css
  .kv{display:flex; gap:6px} .kv input, .kv select{flex:1} .kv .k1{flex:2}
```

(`.k1` gives the model-id column double width in the 3- and 4-column editors; `.kv` is flex, so extra children need no grid changes.)

- [ ] **Step 4: Add the `fPricing` builder**

In `src/panel/page.ts`, immediately after the `fKV` function, insert:

```js
  // model id -> {input_per_mtok, output_per_mtok}. A three-column cousin of fKV;
  // pricing is the only place that needs two values per key. _get() returns the
  // RAW string rows so the caller can report which cell is bad by name.
  function fPricing(label, hint, obj){ var f=fld(label,hint); var wrap=el("div","rows");
    var rows=Object.keys(obj||{}).map(function(k){ var e=obj[k]||{};
      return [k, e.input_per_mtok==null?"":String(e.input_per_mtok), e.output_per_mtok==null?"":String(e.output_per_mtok)]; });
    function cell(p, idx, ph, aria, cls){ var n=el("input"); n.type="text"; n.className=cls; n.placeholder=ph; n.setAttribute("aria-label",aria); n.value=p[idx];
      n.oninput=function(){ p[idx]=n.value; }; return n; }
    function draw(){ wrap.textContent="";
      rows.forEach(function(p,i){ var row=el("div","kv");
        row.appendChild(cell(p,0,"upstream model id","Priced model","mono k1"));
        row.appendChild(cell(p,1,"in $/Mtok","Input price per million tokens","mono num"));
        row.appendChild(cell(p,2,"out $/Mtok","Output price per million tokens","mono num"));
        var x=el("button","act",""); x.type="button"; x.textContent="×"; x.setAttribute("aria-label","Remove pricing row"); x.onclick=function(){ rows.splice(i,1); draw(); }; row.appendChild(x);
        wrap.appendChild(row); });
      var add=el("button","act",""); add.type="button"; add.textContent="+ Add model"; add.onclick=function(){ rows.push(["","",""]); draw(); }; wrap.appendChild(add); }
    draw(); f.appendChild(wrap); f._get=function(){ return rows.slice(); }; return f; }
```

- [ ] **Step 5: Add the form and the card**

In `src/panel/page.ts`, immediately after `defaultsForm`, insert:

```js
  function pricingForm(pr){ var fRows;
    openForm("Edit model pricing", function(body){
      var note=el("div","hint"); note.style.margin="0 2px 14px";
      note.textContent="Keyed by the REAL upstream model id (glm-5.2), not the virtual model name (fusion-coder). Prices are USD per million tokens and only feed the cost figures in logs and response headers — they never affect routing.";
      body.appendChild(note);
      fRows=fPricing("Prices","One row per model. Both columns are required; use 0 for a free model.", pr);
      body.appendChild(fRows);
    }, function(){
      var out={}, rs=fRows._get();
      for(var i=0;i<rs.length;i++){ var r=rs[i], k=r[0].trim(); if(!k) continue;
        var vin=Number(r[1]), vout=Number(r[2]);
        if(r[1].trim()===""||!isFinite(vin)||vin<0){ formError("Input price for '"+k+"' must be a number of 0 or more."); return; }
        if(r[2].trim()===""||!isFinite(vout)||vout<0){ formError("Output price for '"+k+"' must be a number of 0 or more."); return; }
        out[k]={ input_per_mtok:vin, output_per_mtok:vout }; }
      saveSettings("pricing", out, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
```

Then in `renderSettings`, change the opening line to read `pricing` off `cfg` and append the card after "Fusion defaults":

```js
  function renderSettings(){ var box=document.getElementById("settings-editor"); box.textContent=""; if(!cfg) return;
    var sv=cfg.server||{}, up=cfg.upstream||{}, df=cfg.defaults||{}, pr=cfg.pricing||{}, ov=cfg.overrides||{};
```

and after the "Fusion defaults" `box.appendChild(...)` call, insert:

```js
    var prCount=Object.keys(pr).length;
    box.appendChild(settingsCard("Model pricing",
      prCount?(prCount+" model(s) priced · "+Object.keys(pr).slice(0,3).map(esc).join(", ")+(prCount>3?", …":"")):"no prices set — cost reporting shows 0",
      function(){ pricingForm(pr); }, false));
```

(`ov` is unused until Task 8 — add it now so the two tasks do not fight over the same line.)

- [ ] **Step 6: Run the tests**

Run: `npx vitest run test/panel_settings_form.test.ts && npm run typecheck`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/panel/page.ts test/panel_settings_form.test.ts
git commit -m "feat(panel): add a Model pricing settings card"
```

---

### Task 8: Capability overrides settings card

`overrides` is `Record<string, {tools?: boolean, vision?: boolean, context?: number|null}>`. Each flag is **tri-state** — omitted means "no opinion" — so plain toggles cannot express it; use three-option selects.

The hint copy must be precise. Per `src/capabilities.ts:97-109`, an override is consulted **only inside `degrade()`, i.e. when `/api/show` discovery fails**. It is a fallback, not a force. Wording like "force vision on" would be wrong.

**Files:**
- Modify: `src/panel/page.ts` (new `fOverrides` builder, new `overridesForm`, one line in `renderSettings`)
- Modify: `test/panel_settings_form.test.ts`

**Interfaces:**
- Consumes: `fld`, `el`, `openForm`, `closeForm`, `formError`, `saveSettings`, `settingsCard`; `openSettingsCard`, `setCell`, `setRowSelect`, `addRow` from Task 6
- Produces: `fOverrides(label, hint, obj)` — `_get()` returns `Array<[key, toolsStr, visionStr, contextStr]>` where the flag strings are `""` | `"on"` | `"off"`

- [ ] **Step 1: Write the failing test**

Append to `test/panel_settings_form.test.ts`:

```ts
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
});
```

Add `setRowSelect` to the import list at the top of the file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_settings_form.test.ts -t "Capability overrides"`
Expected: FAIL — `no card 'Capability overrides' in #settings-editor`.

- [ ] **Step 3: Add the `fOverrides` builder**

In `src/panel/page.ts`, immediately after `fPricing`, insert:

```js
  // model id -> {tools?, vision?, context?}. Each flag is TRI-STATE: "unset"
  // omits the key entirely. This map is only consulted when /api/show discovery
  // FAILS (see degrade() in src/capabilities.ts) — it is a fallback, not a force.
  function fOverrides(label, hint, obj){ var f=fld(label,hint); var wrap=el("div","rows");
    var TRI=[{label:"unset",value:""},{label:"yes",value:"on"},{label:"no",value:"off"}];
    var rows=Object.keys(obj||{}).map(function(k){ var e=obj[k]||{};
      return [k, e.tools==null?"":(e.tools?"on":"off"), e.vision==null?"":(e.vision?"on":"off"), e.context==null?"":String(e.context)]; });
    function draw(){ wrap.textContent="";
      rows.forEach(function(p,i){ var row=el("div","kv");
        var k=el("input"); k.type="text"; k.className="mono k1"; k.placeholder="upstream model id"; k.setAttribute("aria-label","Overridden model"); k.value=p[0];
        k.oninput=function(){ p[0]=k.value; }; row.appendChild(k);
        [[1,"tools"],[2,"vision"]].forEach(function(d){ var s=el("select"); s.setAttribute("aria-label",d[1]+" override");
          TRI.forEach(function(o){ var op=el("option",null,d[1]+": "+o.label); op.value=o.value; if(o.value===p[d[0]]) op.selected=true; s.appendChild(op); });
          s.onchange=function(){ p[d[0]]=s.value; }; row.appendChild(s); });
        var c=el("input"); c.type="text"; c.className="mono num"; c.placeholder="context"; c.setAttribute("aria-label","Context window override"); c.value=p[3];
        c.oninput=function(){ p[3]=c.value; }; row.appendChild(c);
        var x=el("button","act",""); x.type="button"; x.textContent="×"; x.setAttribute("aria-label","Remove override row"); x.onclick=function(){ rows.splice(i,1); draw(); }; row.appendChild(x);
        wrap.appendChild(row); });
      var add=el("button","act",""); add.type="button"; add.textContent="+ Add override"; add.onclick=function(){ rows.push(["","","",""]); draw(); }; wrap.appendChild(add); }
    draw(); f.appendChild(wrap); f._get=function(){ return rows.slice(); }; return f; }
```

- [ ] **Step 4: Add the form and the card**

In `src/panel/page.ts`, immediately after `pricingForm`, insert:

```js
  function overridesForm(ov){ var fRows;
    openForm("Edit capability overrides", function(body){
      var note=el("div","hint"); note.style.margin="0 2px 14px";
      note.textContent="A FALLBACK, not a force: these values are used only when capability discovery (/api/show) fails for that model. While discovery works, the discovered capabilities win. Keyed by the REAL upstream model id. Leave a flag on 'unset' to fall through to the conservative default (tools: yes, vision: no, context: unknown).";
      body.appendChild(note);
      fRows=fOverrides("Overrides","One row per model. Context is the window in tokens; leave it blank to keep it unknown.", ov);
      body.appendChild(fRows);
    }, function(){
      var out={}, rs=fRows._get();
      for(var i=0;i<rs.length;i++){ var r=rs[i], k=r[0].trim(); if(!k) continue; var e={};
        if(r[1]) e.tools=(r[1]==="on");
        if(r[2]) e.vision=(r[2]==="on");
        var cv=r[3].trim();
        if(cv!==""){ var n=Number(cv); if(!isFinite(n)){ formError("Context for '"+k+"' must be a number."); return; } e.context=n; }
        out[k]=e; }
      saveSettings("overrides", out, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
```

Then in `renderSettings`, after the "Model pricing" card, insert:

```js
    var ovCount=Object.keys(ov).length;
    box.appendChild(settingsCard("Capability overrides",
      ovCount?(ovCount+" model(s) with a discovery fallback · "+Object.keys(ov).slice(0,3).map(esc).join(", ")+(ovCount>3?", …":"")):"none — discovery failures fall back to tools: yes, vision: no",
      function(){ overridesForm(ov); }, false));
```

- [ ] **Step 5: Run the tests**

Run: `npm run test && npm run typecheck`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/panel/page.ts test/panel_settings_form.test.ts
git commit -m "feat(panel): add a Capability overrides settings card"
```

---

### Task 9: "JSON" escape hatch on model and provider cards

This is what structurally guarantees full coverage: anything the forms cannot express (an inline smart block, `accounts[].extra_headers`, a `bineval.dimensions` list) stays editable. Two clearly separated modes — never a hybrid form/JSON view, which would raise "which one wins?" on every save.

**Files:**
- Modify: `src/panel/page.ts` (new `fJson` builder, new `jsonForm`, two card action buttons)
- Create: `test/panel_json_editor.test.ts`

**Interfaces:**
- Consumes: `fld`, `el`, `openForm`, `closeForm`, `formError`, `saveModel`, `saveProvider`, `mkBtn`; harness exports from Task 1
- Produces: `fJson(label, hint, value)` — `_get()` returns the raw textarea string; `jsonForm(title, hint, value, onParsed)`

- [ ] **Step 1: Write the failing test**

Create `test/panel_json_editor.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_json_editor.test.ts`
Expected: FAIL — `no 'JSON' button on card 'fusion-coder'`.

- [ ] **Step 3: Add the `fJson` builder and `jsonForm`**

In `src/panel/page.ts`, immediately after `fOverrides`, insert:

```js
  // Raw-JSON escape hatch. Deliberately NOT merged with the form fields: two
  // clearly separated modes, so there is never a "which one wins?" question.
  function fJson(label, hint, value){ var f=fld(label,hint); var t=el("textarea"); t.className="mono"; t.rows=20; t.spellcheck=false;
    t.id="fi"+(++dlSeq); if(f._label)f._label.htmlFor=t.id; if(f._hintId)t.setAttribute("aria-describedby",f._hintId);
    t.value=JSON.stringify(value,null,2); f.appendChild(t); f._get=function(){ return t.value; }; return f; }
  // Parse-and-hand-off. Whole-config zod validation still gates the result
  // server-side, so this only rejects what is not even a JSON object.
  function jsonForm(title, hint, value, onParsed){ var fJ;
    openForm(title, function(body){ fJ=fJson("JSON", hint, value); body.appendChild(fJ); },
      function(){ var parsed;
        try { parsed=JSON.parse(fJ._get()); } catch(e){ formError("Invalid JSON: "+String(e.message||e)); return; }
        if(!parsed||typeof parsed!=="object"||Array.isArray(parsed)){ formError("The top level must be a JSON object."); return; }
        onParsed(parsed); });
  }
```

- [ ] **Step 4: Add the button to the model card**

In `renderModels`, replace:

```js
      var acts=el("div","eacts"); acts.appendChild(mkBtn("Edit","act",false,function(){ modelForm(name,m); }));
```

with:

```js
      var acts=el("div","eacts"); acts.appendChild(mkBtn("Edit","act",false,function(){ modelForm(name,m); }));
      acts.appendChild(mkBtn("JSON","act",false,function(){ jsonForm("Edit model "+name+" as JSON",
        "The whole model object. Use this for anything the form cannot express — an inline smart simple/fusion block, custom bineval questions. The config schema still validates it on save.", m,
        function(o){ saveModel(name, o, function(ok,err){ if(ok) closeForm(); else formError(err); }); }); }));
```

- [ ] **Step 5: Add the button to the provider card**

In `renderProviders`, replace:

```js
      acts.appendChild(mkBtn("Edit","act",false,function(){ providerForm(id,p); }));
```

with:

```js
      acts.appendChild(mkBtn("Edit","act",false,function(){ providerForm(id,p); }));
      acts.appendChild(mkBtn("JSON","act",false,function(){ jsonForm("Edit provider "+id+" as JSON",
        "The whole provider group including every account. Use this for anything the forms cannot express — accounts[].extra_headers. Header VALUES are shown masked as ••• and the server swaps the real ones back in on save; leave a mask alone to keep the stored value.", p,
        function(o){ saveProvider(id, o, function(ok,err){ if(ok) closeForm(); else formError(err); }); }); }));
```

- [ ] **Step 6: Run the tests**

Run: `npm run test && npm run typecheck`
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add src/panel/page.ts test/panel_json_editor.test.ts
git commit -m "feat(panel): add a raw-JSON editor for models and providers"
```

---

### Task 10: model-card badges + docs

The model list shows only `strategySummary`, so an operator cannot tell from the Models tab which models have the vision pre-stage, web search, BinEval, an adversarial member, or escalation disabled. This is what let `image_describe` go unnoticed.

**Files:**
- Modify: `src/panel/page.ts` (`renderModels`)
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `test/panel_model_form.test.ts`

**Interfaces:**
- Consumes: harness exports from Task 1
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

Append to `test/panel_model_form.test.ts`:

```ts
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
```

Add `clickTab` to the import list.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run test/panel_model_form.test.ts -t badges`
Expected: FAIL — the only badge is the provider name.

- [ ] **Step 3: Add the badges**

In `src/panel/page.ts`, immediately before `renderModels`, insert:

```js
  // Extras that are easy to forget exist. They live inside nested blocks, so
  // strategySummary can't show them and an operator cannot tell from the list
  // that a model has (say) a vision pre-stage configured.
  function modelBadges(m){ var out=[];
    if(m.image_describe&&m.image_describe.enabled) out.push("vision pre-stage");
    if(m.web_search&&m.web_search.enabled) out.push("web search");
    if(m.bineval&&m.bineval.enabled) out.push("bineval");
    if(m.adversarial) out.push("adversarial");
    if(m.strategy==="smart"&&m.escalate_on_tool_error===false) out.push("no escalation");
    return out; }
```

Then in `renderModels`, replace:

```js
      r.appendChild(el("span","ename",name)); r.appendChild(el("span","ptype",m.strategy)); if(m.provider) r.appendChild(el("span","badge",m.provider));
```

with:

```js
      r.appendChild(el("span","ename",name)); r.appendChild(el("span","ptype",m.strategy)); if(m.provider) r.appendChild(el("span","badge",m.provider));
      modelBadges(m).forEach(function(b){ r.appendChild(el("span","badge",b)); });
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run test/panel_model_form.test.ts && npm run typecheck`
Expected: both PASS.

- [ ] **Step 5: Document the limitation and the new surface**

In `README.md`, in the section that describes the web panel, add:

```markdown
The Settings tab covers server, upstream, fusion defaults, model pricing and
capability overrides. Every model and provider card also has a **JSON** button
that opens the raw object — use it for anything the forms do not render, such as
an inline `simple`/`fusion` block on a `smart` model or `accounts[].extra_headers`.

**Known limitation:** saving from the panel replaces the edited node in
`fusion.yaml`, so comments written *inside* that model or provider block are
lost (comments on sibling entries survive). Every write is preceded by a
timestamped backup — the last 10 are kept next to the config file.
```

In `CHANGELOG.md`, add a new entry at the top under the next version heading:

```markdown
### Fixed
- **Panel saves no longer delete config keys the form does not render.** The
  model and account forms now edit the object they loaded instead of rebuilding
  it from the rendered fields. Previously an unrelated edit silently dropped
  `image_describe`, `escalate_on_tool_error`, `accounts[].request_timeout_s` and
  anything else the form did not know about. Covered by a new jsdom test suite
  that runs the shipped panel script.
- The fusion form no longer renders a `request_overrides` field, which
  `FusionModelSchema` rejects outright. Its real counterpart,
  `synth_request_overrides`, is unchanged.

### Added
- Panel: `image_describe` (vision pre-stage) controls on the fusion form.
- Panel: `escalate_on_tool_error` on the smart form; an inline `simple`/`fusion`
  block is now shown read-only and preserved instead of being wiped.
- Panel: `accounts[].request_timeout_s` and `upstream.per_model_concurrency`.
- Panel: **Model pricing** and **Capability overrides** settings cards.
- Panel: a **JSON** button on every model and provider card for raw editing.
- Panel: model cards badge the vision pre-stage, web search, BinEval,
  adversarial member and disabled escalation.
```

- [ ] **Step 6: Run everything one last time**

Run: `npm run test && npm run typecheck`
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add src/panel/page.ts test/panel_model_form.test.ts README.md CHANGELOG.md
git commit -m "feat(panel): badge model extras; document the JSON editor and the comment limitation"
```

---

## Self-review

**Spec coverage.** Every row of the spec's findings table maps to a task: A1→4, A2→5, A3→2, A4→1+2, B1→3, C1→7, C2→8, C3→6, C4→9 (JSON editor) and 2 (merge keeps it), D1→5, D2→9, E1→10. Spec design sections 1–6 map to tasks 1–2, 2/4/5/6, 3, 7–8, 9, 1 respectively. All eight acceptance criteria have a test: `image_describe` round-trip (Task 1 + 4), `escalate_on_tool_error` (5), `request_timeout_s` (2), strategy switch clean (1), toggle-off deletes (1), no fusion `request_overrides` (3), pricing/overrides editable (7, 8), full reachability via JSON (9), suite green (every task's last step).

**Non-goals honoured.** No field-by-field YAML merge — the comment loss is documented in Task 10 Step 5 instead. No optimistic concurrency. No framework rewrite.

**Type/name consistency.** Harness exports used by later tasks (`mountPanel`, `Panel`, `SentRequest`, `clickTab`, `cardButton`, `field`, `hasField`, `input`, `select`, `setInput`, `setSelect`, `clickToggle`, `saveForm`, `formError`, `openModelForm`, `openAccountForm`) are all defined in Task 1; `openSettingsCard`, `rows`, `addRow`, `setCell`, `setRowSelect` are added in Task 6 before Tasks 7–8 use them. Panel-side builders: `fPricing` (Task 7) and `fOverrides` (Task 8) both `_get()` raw string rows, matching their save handlers; `fJson`/`jsonForm` (Task 9) are defined before their two call sites in the same task.

**Ordering hazards.** Task 6 Step 5 and Tasks 7/8 all touch `renderSettings`'s first line — Task 7 rewrites it once to add both `pr` and `ov`, so Task 8 only appends a card. Task 4's save code is inserted at the `else delete obj.bineval;` line that Task 1 creates, so Task 1 must land first. Task 5's save branch assumes merge-on-save (`obj` already carries the inline block) — also Task 1.
