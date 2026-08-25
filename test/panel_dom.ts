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
  /** Settle the panel's pending fetches and its 400ms reload debounce. */
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
 * Opt-in failure hook for `mountPanelWithFailure`: given the request's method
 * and path, return the error body to fail the request with, or
 * `undefined`/`null` to let it succeed normally. Lets a test exercise the
 * `.catch → formError/toast` path of a save/delete handler, which the
 * default stub never triggers.
 */
export type FailRequest = (method: string, path: string) => { status?: number; body?: unknown } | undefined | null;

/**
 * Mount the SHIPPED panel markup + script into jsdom and stub the network.
 *
 * The point of running the real string (rather than a copy of the logic) is that
 * a schema field the form forgets to round-trip must fail HERE, in a test, not
 * in fusion.yaml. Keep it black-box: drive the panel through DOM clicks only.
 */
export async function mountPanel(cfg: unknown): Promise<Panel> {
  return doMount(cfg);
}

/**
 * Same as `mountPanel`, but a non-GET request `failRequest` flags fails
 * instead of succeeding with `{ok:true}` — for testing a save/delete
 * handler's error path. Kept as a separate export so `mountPanel`'s
 * signature never changes.
 */
export async function mountPanelWithFailure(cfg: unknown, failRequest: FailRequest): Promise<Panel> {
  return doMount(cfg, failRequest);
}

async function doMount(cfg: unknown, failRequest?: FailRequest): Promise<Panel> {
  // BOTH timer kinds are faked. setInterval: the 3s monitor poll would otherwise
  // fire for the whole test run. setTimeout: the panel debounces a config reload
  // by 400 ms after every save (`reloadConfigSoon`), and with a REAL timer that
  // callback outlives the test — it fires after `unmount()` has unstubbed fetch
  // and cleared the body, so it hits the real fetch with the relative URL
  // "admin/config" (unhandled rejection) and renders into the NEXT test's DOM.
  // On an idle machine it landed harmlessly between tests; under CPU contention
  // it landed inside one, which is the whole story behind the panel suite's
  // intermittent "no PUT was sent" reds. Faking it makes `flush()` deterministic
  // and lets `useRealTimers()` in `unmount` discard whatever is still pending.
  vi.useFakeTimers({ toFake: ["setInterval", "setTimeout"] });

  const markup = extract(BODY_RE, "<body>");
  const script = extract(SCRIPT_RE, "<script>");
  document.body.innerHTML = markup.replace(SCRIPT_RE, "");

  const sent: SentRequest[] = [];
  vi.stubGlobal("fetch", (path: string, opt?: { method?: string; body?: string }) => {
    const method = (opt && opt.method) || "GET";
    if (method !== "GET") {
      sent.push({ method, path, body: opt && opt.body ? JSON.parse(opt.body) : undefined });
      const failure = failRequest ? failRequest(method, path) : null;
      if (failure) {
        return Promise.resolve(
          fakeResponse(failure.body !== undefined ? failure.body : { error: "simulated failure" }, false, failure.status || 500),
        );
      }
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
function fakeResponse(data: unknown, ok = true, status = 200): { ok: boolean; status: number; json(): Promise<unknown> } {
  return { ok, status, json: () => Promise.resolve(data) };
}

/**
 * Settle everything the panel has in flight: the fetch promise chains AND its
 * own 400 ms `reloadConfigSoon` debounce. `advanceTimersByTimeAsync` drains the
 * microtask queue between timer firings, so one call covers a
 * `fetch().then().then()` chain that crosses several timer boundaries — which a
 * single real `setTimeout(0)` tick only ever did by luck on an idle machine.
 * Stays under the 3 s monitor poll interval.
 */
async function flush(): Promise<void> {
  await vi.advanceTimersByTimeAsync(500);
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
