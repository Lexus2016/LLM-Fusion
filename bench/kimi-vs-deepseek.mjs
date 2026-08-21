#!/usr/bin/env node
/**
 * kimi-vs-deepseek — a FOCUSED head-to-head between the two `single`-strategy
 * aliases that a Claude Code session actually rides on for cheap/fast turns:
 *
 *   fast-kimi     -> kimi-k2.7-code               (currently ANTHROPIC_DEFAULT_HAIKU_MODEL)
 *   fast-deepseek -> deepseek-v4-flash:0731-cloud (currently the smart router's `simple:` route)
 *
 * Two axes, because "good at coding" and "survives an agent loop" are different
 * failure modes and the project already conflates them:
 *
 *  A. AGENTIC (machine-graded, no LLM scorer) — a real OpenAI-tools loop against
 *     a mock filesystem. Deterministic pass/fail per scenario plus turn count,
 *     so an answer that is right for the wrong reason (guessed the content
 *     without reading the file) still fails. Scenarios probe the six ways an
 *     agent loop actually breaks: wrong args, no chaining, no parallelism,
 *     tool-happiness on a no-tool question, giving up on a tool error, and
 *     enum/schema drift.
 *
 *  B. CODING (blind, two independent scorers) — the coding/debugging subset of
 *     bench/tasks.json, reusing that harness's rubric and per-line JSON parse.
 *     Scorers are gpt-oss:120b and minimax-m3: both sit OUTSIDE every condition
 *     here, so neither grades its own family.
 *
 * Both conditions go THROUGH THE PROXY, not direct to Ollama — the question is
 * which alias to configure, so request_overrides and the connector path are
 * part of what is under test.
 *
 * Usage: OLLAMA_API_KEY=... node bench/kimi-vs-deepseek.mjs [--agentic-only] [--coding-only]
 */
import { readFileSync, writeFileSync } from "node:fs";

const PROXY = process.env.FUSION_PROXY_URL ?? "http://127.0.0.1:8081/v1/chat/completions";
const OLLAMA = "https://ollama.com/v1/chat/completions";
const KEY = process.env.OLLAMA_API_KEY;
if (!KEY) { console.error("OLLAMA_API_KEY required"); process.exit(1); }

const CONDITIONS = ["fast-kimi", "fast-deepseek"];
const OUT = "bench/results-kimi-vs-deepseek.json";
const AGENTIC_ONLY = process.argv.includes("--agentic-only");
const CODING_ONLY = process.argv.includes("--coding-only");

// --- plumbing ---------------------------------------------------------------
/**
 * Non-streaming here ON PURPOSE, unlike fusion-bench: tool_calls arrive as
 * fragmented deltas whose index/id reassembly is its own source of harness bugs,
 * and every call in this bench is short (agentic turns are tiny; coding answers
 * are capped well under the ~180s wall that forced streaming there).
 */
async function chat(url, body, tries = 3) {
  const isProxy = url === PROXY;
  // The proxy's connector circuit breaker has a 60s cooldown — a 3s retry lands
  // inside it and turns one transient 503 into a scored zero. Wait it out.
  const backoff = (a) => (isProxy ? [5000, 65000, 65000][a] ?? 65000 : 3000 * (a + 1));
  for (let a = 0; ; a++) {
    try {
      const t0 = Date.now();
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
        body: JSON.stringify({ ...body, stream: false }),
      });
      if (res.status === 429 || res.status >= 500) throw new Error(`HTTP ${res.status}`);
      if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`), { fatal: true });
      const j = await res.json();
      const m = j.choices?.[0]?.message ?? {};
      return {
        message: m,
        text: (typeof m.content === "string" ? m.content : "").replace(/<think>[\s\S]*?<\/think>/gi, "").trim(),
        toolCalls: Array.isArray(m.tool_calls) ? m.tool_calls : [],
        finish: j.choices?.[0]?.finish_reason ?? null,
        ms: Date.now() - t0,
        usage: j.usage ?? null,
      };
    } catch (e) {
      if (e.fatal || a >= tries - 1) throw e;
      await new Promise((r) => setTimeout(r, backoff(a)));
    }
  }
}

// ===========================================================================
// AXIS A — agentic tool loop
// ===========================================================================

/** Mock repo the tools read from. Contents are chosen so a guessed answer is wrong. */
const FS = {
  "src/config.ts": `export interface Config { timeout: number; retries: number }\nexport function parseConfig(raw: string): Config {\n  const o = JSON.parse(raw);\n  return { timeout: o.timeout ?? 47, retries: o.retries ?? 3 };\n}\n`,
  "src/util.ts": "export const noop = () => {};\nexport const id = <T>(x: T) => x;\nexport function sum(a: number, b: number) {\n  return a + b;\n}\n",
  "src/a.ts": "export function alpha() {}\nexport function beta() {}\nconst hidden = 1;\n",
  "src/b.ts": "export function gamma() {}\n",
  "src/c.ts": "export function delta() {}\nexport function epsilon() {}\nexport function zeta() {}\n",
};
const DIRS = { "src": ["a.ts", "b.ts", "c.ts", "config.ts", "util.ts"] };

const TOOLS = [
  {
    type: "function",
    function: {
      name: "list_files",
      description: "List the file names in a directory of the repository.",
      parameters: { type: "object", properties: { dir: { type: "string", description: "Directory path, e.g. 'src'" } }, required: ["dir"] },
    },
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read the full text content of one file.",
      parameters: { type: "object", properties: { path: { type: "string", description: "Repo-relative file path, e.g. 'src/util.ts'" } }, required: ["path"] },
    },
  },
  {
    type: "function",
    function: {
      name: "run_tests",
      description: "Run a test suite and return its result.",
      parameters: {
        type: "object",
        properties: { suite: { type: "string", enum: ["unit", "e2e"], description: "Which suite to run" } },
        required: ["suite"],
      },
    },
  },
];

/** Executes a tool call against the mock repo. Returns the string the model sees. */
function execTool(name, args) {
  if (name === "list_files") {
    const d = String(args?.dir ?? "").replace(/^\.?\/*/, "").replace(/\/$/, "");
    return DIRS[d] ? DIRS[d].join("\n") : `ERROR: no such directory '${args?.dir}'. Known directories: src`;
  }
  if (name === "read_file") {
    const p = String(args?.path ?? "").replace(/^\.?\/*/, "");
    if (FS[p]) return FS[p];
    // Deliberate recovery affordance: the error NAMES the fix. A model that
    // gives up or fabricates content here fails S5.
    return `ERROR: no such file '${args?.path}'. Did you mean 'src/config.ts'?`;
  }
  if (name === "run_tests") {
    if (args?.suite === "e2e") return "e2e: 12 passed, 0 failed";
    if (args?.suite === "unit") return "unit: 40 passed, 1 failed";
    return `ERROR: invalid suite '${args?.suite}'. Allowed: unit, e2e`;
  }
  return `ERROR: unknown tool '${name}'`;
}

const SYS = "You are a coding agent working in a repository. Use the provided tools to inspect real files instead of guessing. When you have the answer, reply with it in plain text.";

const SCENARIOS = [
  {
    id: "S1-basic",
    what: "one correct tool call with correct args",
    // First revision asked for "lines of code" and BOTH models answered 4 for a
    // 5-line file — "lines of code" legitimately excludes a bare closing brace,
    // so the scenario was measuring my ambiguity, not their accuracy. Replaced
    // with a question that has exactly one defensible answer.
    prompt: "What is the name of the third exported symbol declared in src/util.ts? Answer with just the identifier.",
    grade: ({ calls, finalText }) => {
      const read = calls.find((c) => c.name === "read_file");
      const ok = !!read && String(read.args?.path ?? "").endsWith("src/util.ts");
      const answered = /\bsum\b/.test(finalText);
      return { pass: ok && answered, why: !ok ? "did not read src/util.ts" : answered ? "" : `wrong symbol: ${finalText.slice(0, 60)}` };
    },
  },
  {
    id: "S2-chain",
    what: "multi-step chaining: discover then read",
    prompt: "Somewhere in the src/ directory there is a function named parseConfig. Find it and tell me the default timeout value it falls back to.",
    grade: ({ calls, finalText }) => {
      const listed = calls.some((c) => c.name === "list_files");
      const readCfg = calls.some((c) => c.name === "read_file" && String(c.args?.path ?? "").includes("config.ts"));
      const right = /\b47\b/.test(finalText);
      return { pass: listed && readCfg && right, why: !listed ? "never listed the dir" : !readCfg ? "never read config.ts" : right ? "" : "wrong timeout value" };
    },
  },
  {
    id: "S3-parallel",
    what: "batches independent reads instead of serializing",
    prompt: "Read src/a.ts, src/b.ts and src/c.ts, then tell me the TOTAL number of exported functions across all three. Answer with just the number.",
    grade: ({ calls, finalText, turns, maxCallsInOneTurn }) => {
      const files = new Set(calls.filter((c) => c.name === "read_file").map((c) => String(c.args?.path ?? "").replace(/^\.?\/*/, "")));
      const allThree = ["src/a.ts", "src/b.ts", "src/c.ts"].every((f) => files.has(f));
      const right = /\b6\b/.test(finalText);
      return {
        pass: allThree && right,
        why: !allThree ? "did not read all three files" : right ? "" : "wrong total",
        extra: { parallel: maxCallsInOneTurn >= 2, turns },
      };
    },
  },
  {
    id: "S4-restraint",
    what: "does NOT reach for a tool when none is needed",
    prompt: "Without using any tools, what is 17 * 23? Answer with just the number.",
    grade: ({ calls, finalText }) => {
      const right = /\b391\b/.test(finalText);
      return { pass: calls.length === 0 && right, why: calls.length ? `called ${calls.length} tool(s) anyway` : right ? "" : "wrong arithmetic" };
    },
  },
  {
    id: "S5-recovery",
    what: "recovers from a tool error instead of giving up or fabricating",
    prompt: "Read the file src/configuration.ts and tell me the default number of retries. Answer with just the number.",
    grade: ({ calls, finalText }) => {
      const retried = calls.filter((c) => c.name === "read_file").length >= 2;
      const hitCorrect = calls.some((c) => c.name === "read_file" && String(c.args?.path ?? "").includes("src/config.ts"));
      const right = /\b3\b/.test(finalText);
      return { pass: retried && hitCorrect && right, why: !retried ? "gave up after the first error" : !hitCorrect ? "never tried src/config.ts" : right ? "" : "wrong retries value" };
    },
  },
  {
    id: "S6-schema",
    what: "honors an enum parameter exactly",
    prompt: "Run the end-to-end test suite and report how many tests passed.",
    grade: ({ calls, finalText }) => {
      const rt = calls.find((c) => c.name === "run_tests");
      const ok = rt && rt.args?.suite === "e2e";
      const right = /\b12\b/.test(finalText);
      return { pass: !!ok && right, why: !rt ? "never called run_tests" : !ok ? `bad enum value: ${JSON.stringify(rt.args?.suite)}` : right ? "" : "wrong pass count" };
    },
  },
];

const MAX_TURNS = 8;

async function runAgentic(model, scenario) {
  const messages = [{ role: "system", content: SYS }, { role: "user", content: scenario.prompt }];
  const calls = [];
  let turns = 0, ms = 0, maxCallsInOneTurn = 0, finalText = "", error = null, badJson = 0, hallucinated = 0;
  try {
    for (; turns < MAX_TURNS; turns++) {
      const r = await chat(PROXY, { model, messages, tools: TOOLS, max_tokens: 4096, temperature: 0 });
      ms += r.ms;
      if (r.toolCalls.length === 0) { finalText = r.text; break; }
      maxCallsInOneTurn = Math.max(maxCallsInOneTurn, r.toolCalls.length);
      // Push the assistant turn verbatim so ids line up with the tool replies.
      messages.push({ role: "assistant", content: r.message.content ?? "", tool_calls: r.toolCalls });
      for (const tc of r.toolCalls) {
        const name = tc.function?.name ?? "";
        let args = {};
        try { args = JSON.parse(tc.function?.arguments || "{}"); }
        catch { badJson++; args = {}; }
        if (!TOOLS.some((t) => t.function.name === name)) hallucinated++;
        calls.push({ name, args });
        messages.push({ role: "tool", tool_call_id: tc.id, content: execTool(name, args) });
      }
    }
  } catch (e) { error = String(e.message ?? e).slice(0, 200); }
  const runawayLoop = turns >= MAX_TURNS && !finalText;
  const g = error || runawayLoop
    ? { pass: false, why: error ? `call error: ${error}` : `runaway loop: hit ${MAX_TURNS} turns without answering` }
    : scenario.grade({ calls, finalText, turns, maxCallsInOneTurn });
  return { scenario: scenario.id, what: scenario.what, model, ...g, turns: turns + 1, toolCalls: calls.length, maxCallsInOneTurn, badJson, hallucinated, ms, finalText: finalText.slice(0, 300), callTrace: calls.map((c) => `${c.name}(${JSON.stringify(c.args).slice(0, 60)})`) };
}

// ===========================================================================
// AXIS B — coding quality, blind, two scorers
// ===========================================================================
/**
 * `--coding-tasks T01,T06` narrows the set — needed because a task can come back
 * unscored through no fault of the model: on the first run BOTH scorers failed to
 * emit a parseable line for fast-kimi's T01 label (an 80s, very long answer), so
 * that cell was `n/a`. Re-running just that task is cheaper than the whole axis.
 */
const argT = process.argv.indexOf("--coding-tasks");
const CODING_TASK_IDS = argT > -1
  ? process.argv[argT + 1].split(",")
  : ["T01", "T02", "T06", "T08", "T10", "T14"];
const SCORERS = ["gpt-oss:120b", "minimax-m3"];
const GEN_TOKENS = 16384;

function extractJson(s) {
  const start = s.indexOf("{");
  if (start === -1) throw new Error("no JSON object");
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (esc) { esc = false; continue; }
    if (c === "\\") { esc = true; continue; }
    if (c === '"') inStr = !inStr;
    if (inStr) continue;
    if (c === "{") depth++; else if (c === "}" && --depth === 0) return s.slice(start, i + 1);
  }
  throw new Error("unbalanced JSON");
}

async function scoreTask(task, answers) {
  const scorable = answers.filter((a) => a.text.trim().length > 0);
  if (scorable.length < 2) return [];
  const shuffled = [...scorable].sort(() => Math.random() - 0.5);
  const labels = shuffled.map((a, i) => ({ label: `S${i + 1}`, ...a }));
  const MAX = 8000;
  const block = labels.map((l) => `### ${l.label}\n${l.text.length > MAX ? `${l.text.slice(0, MAX)}\n...[truncated for scoring]` : l.text}`).join("\n\n");
  const rubric =
    "You are a strict, impartial evaluator. Score EACH answer against the TASK and the EXPECTED CRITERIA on three axes, " +
    "integers 0-10: accuracy (technically correct, criteria satisfied), completeness (covers all required aspects), " +
    "truthfulness (no fabricated APIs or facts; confident fabrication is 0-3). Grade HARSHLY and use the full scale: " +
    "10 means flawless expert work; deduct for every missed criterion; a typical good answer lands 6-8. " +
    `Respond with EXACTLY ${labels.length} lines and nothing else — one line per answer, each line a single ` +
    'standalone JSON object: {"label":"S1","accuracy":n,"completeness":n,"truthfulness":n,"note":"<=15 words"}. ' +
    "No surrounding array, no code fences, no blank lines, no prose.";
  const messages = [
    { role: "system", content: rubric },
    { role: "user", content: `TASK:\n${task.prompt}\n\nEXPECTED CRITERIA:\n${task.criteria}\n\nANSWERS:\n${block}` },
  ];
  const parse = (raw) => {
    const out = new Map();
    for (const line of raw.replace(/```(?:json)?/gi, "").split("\n")) {
      const t = line.trim();
      if (!t.startsWith("{")) continue;
      try { const o = JSON.parse(extractJson(t)); if (typeof o.label === "string") out.set(o.label, o); } catch { /* skip line */ }
    }
    return out;
  };
  const perScorer = await Promise.all(SCORERS.map(async (m) => {
    try {
      const r = await chat(OLLAMA, { model: m, max_tokens: 2048, temperature: 0, messages });
      return { scorer: m, map: parse(r.text) };
    } catch (e) { console.error(`  scorer ${m} failed: ${e.message}`); return { scorer: m, map: new Map() }; }
  }));
  return labels.map((l) => {
    const raws = perScorer.map((p) => ({ scorer: p.scorer, s: p.map.get(l.label) })).filter((x) => x.s);
    if (!raws.length) return { cond: l.cond, total: null, byScorer: [] };
    const totals = raws.map((x) => (Number(x.s.accuracy) || 0) + (Number(x.s.completeness) || 0) + (Number(x.s.truthfulness) || 0));
    return {
      cond: l.cond,
      total: totals.reduce((a, b) => a + b, 0) / totals.length,
      byScorer: raws.map((x, i) => ({ scorer: x.scorer, total: totals[i], note: x.s.note })),
    };
  });
}

// ===========================================================================
const results = { agentic: [], coding: [], meta: { conditions: CONDITIONS, proxy: PROXY, scorers: SCORERS } };

if (!CODING_ONLY) {
  console.error("=== AXIS A: agentic tool loop ===");
  for (const sc of SCENARIOS) {
    // Conditions run in parallel per scenario: same scenario, same moment, so a
    // provider hiccup hits both or neither.
    const rs = await Promise.all(CONDITIONS.map((m) => runAgentic(m, sc)));
    for (const r of rs) {
      console.error(`  ${sc.id.padEnd(13)} ${r.model.padEnd(14)} ${r.pass ? "PASS" : "FAIL"}  turns=${r.turns} calls=${r.toolCalls} ${Math.round(r.ms / 100) / 10}s ${r.why ? `— ${r.why}` : ""}`);
      if (!r.pass) console.error(`      trace: ${r.callTrace.join(" -> ") || "(no tool calls)"} | final: ${r.finalText.slice(0, 120).replace(/\n/g, " ")}`);
    }
    results.agentic.push(...rs);
    writeFileSync(OUT, JSON.stringify(results, null, 2));
  }
}

if (!AGENTIC_ONLY) {
  console.error("\n=== AXIS B: coding quality (blind, 2 scorers) ===");
  const allTasks = JSON.parse(readFileSync(new URL("./tasks.json", import.meta.url), "utf8"));
  const tasks = allTasks.filter((t) => CODING_TASK_IDS.includes(t.id));
  for (const t of tasks) {
    const answers = await Promise.all(CONDITIONS.map(async (m) => {
      try {
        const r = await chat(PROXY, { model: m, max_tokens: GEN_TOKENS, messages: [{ role: "user", content: t.prompt }] });
        return { cond: m, text: r.text, ms: r.ms, finish: r.finish };
      } catch (e) { console.error(`  ${t.id} ${m} call error: ${e.message}`); return { cond: m, text: "", ms: 0, finish: "error" }; }
    }));
    const scores = await scoreTask(t, answers);
    console.error(`  ${t.id} (${t.cat}): ${scores.map((s) => `${s.cond}=${s.total === null ? "n/a" : s.total.toFixed(1)}/30`).join("  ")}`);
    // Answer texts are persisted so an unscored cell can be re-scored without
    // paying for generation again, and so any surprising score stays auditable.
    results.coding.push({
      id: t.id, cat: t.cat, scores,
      latency: Object.fromEntries(answers.map((a) => [a.cond, a.ms])),
      answers: Object.fromEntries(answers.map((a) => [a.cond, a.text])),
    });
    writeFileSync(OUT, JSON.stringify(results, null, 2));
  }
}

// --- summary ---------------------------------------------------------------
console.error("\n=== SUMMARY ===");
for (const m of CONDITIONS) {
  const ag = results.agentic.filter((r) => r.model === m);
  const cd = results.coding.flatMap((t) => t.scores.filter((s) => s.cond === m && s.total !== null).map((s) => s.total));
  const lat = results.coding.map((t) => t.latency[m]).filter(Boolean);
  const agLat = ag.map((r) => r.ms).filter(Boolean);
  console.error(
    `${m.padEnd(14)} agentic ${ag.filter((r) => r.pass).length}/${ag.length}` +
    `  coding ${cd.length ? (cd.reduce((a, b) => a + b, 0) / cd.length).toFixed(1) : "n/a"}/30` +
    `  agentic-latency ${agLat.length ? Math.round(agLat.reduce((a, b) => a + b, 0) / agLat.length / 100) / 10 : "n/a"}s/scenario` +
    `  coding-latency ${lat.length ? Math.round(lat.reduce((a, b) => a + b, 0) / lat.length / 100) / 10 : "n/a"}s/task` +
    `  badJson=${ag.reduce((a, r) => a + r.badJson, 0)} hallucinatedTools=${ag.reduce((a, r) => a + r.hallucinated, 0)}`,
  );
}
writeFileSync(OUT, JSON.stringify(results, null, 2));
console.error(`\nwrote ${OUT}`);
