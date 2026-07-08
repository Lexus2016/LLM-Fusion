#!/usr/bin/env node
/**
 * fusion-bench — measures whether the fused virtual model beats every
 * individual panel member (the project's mission statement), and whether
 * self-fusion (same model ×3) approaches the mixed panel (OpenRouter's
 * synthesis>diversity hypothesis, R0).
 *
 * fusion-coder panel as of this run (4 models, replacing the earlier
 * glm+kimi+mistral 3-model panel — re-run from scratch, not resumed, since
 * the panel composition changed and old fusion/fusion-agents scores no
 * longer describe the current pipeline):
 *   solo-glm         : glm-5.2 direct                (judge + synth reference — NOT on panel)
 *   solo-kimi        : kimi-k2.7-code direct         (panel member; self-fusion baseline)
 *   solo-deepseek-flash: deepseek-v4-flash direct    (panel member)
 *   solo-gemini-flash: gemini-3-flash-preview direct (panel member — NOTE: previously removed
 *                                                     from this panel for a "missing
 *                                                     thought_signature" 400 on foreign
 *                                                     tool-call history in agent loops; this
 *                                                     bench sends single-turn prompts with NO
 *                                                     tool history, so it CANNOT reproduce or
 *                                                     rule out that regression — only measures
 *                                                     plain-prompt answer quality)
 *   solo-qwen-coder  : qwen3-coder-next direct       (panel member)
 *   fusion           : fusion-coder via the LOCAL PROXY (the real product path)
 *   self-kimi        : kimi ×3 samples -> judge glm -> synth kimi, replicated in-script
 *                with the product's judge/synth prompt texts (config forbids
 *                duplicate panel members, so this cannot run through the proxy)
 *
 * Scoring: blind, by deepseek-v4-pro (member of NO condition), three axes
 * 0-10 (accuracy, completeness, truthfulness) against per-task criteria;
 * answers are shuffled and anonymized per task.
 *
 * Usage: node bench/fusion-bench.mjs [--tasks N] [--out bench/results.json]
 */
import { readFileSync, writeFileSync } from "node:fs";

const OLLAMA = "https://ollama.com/v1/chat/completions";
const PROXY = "http://127.0.0.1:8080/v1/chat/completions";
const KEY = process.env.OLLAMA_API_KEY;
if (!KEY) { console.error("OLLAMA_API_KEY required"); process.exit(1); }

const argN = process.argv.indexOf("--tasks");
const N = argN > -1 ? Number(process.argv[argN + 1]) : Infinity;
const argO = process.argv.indexOf("--out");
const OUT = argO > -1 ? process.argv[argO + 1] : "bench/results.json";

const allTasks = JSON.parse(readFileSync(new URL("./tasks.json", import.meta.url), "utf8")).slice(0, N);

// --resume: skip tasks already present (with scores) in an existing OUT file —
// lets a killed/crashed run continue without re-paying for completed tasks.
let resumed = [];
if (process.argv.includes("--resume")) {
  try {
    // r.scores could be a non-null EMPTY array if every label was unrecoverable
    // (R2) — that's still "not really done", so require at least one real score.
    resumed = JSON.parse(readFileSync(OUT, "utf8")).filter((r) => r.scores && r.scores.length > 0);
    console.error(`--resume: ${resumed.length} task(s) already done in ${OUT}, skipping them`);
  } catch { /* no existing file — plain fresh run */ }
}
const doneIds = new Set(resumed.map((r) => r.id));
const tasks = allTasks.filter((t) => !doneIds.has(t.id));

// --- plumbing ---------------------------------------------------------------
/**
 * All GENERATION calls stream and accumulate. Non-stream calls die at Ollama's
 * ~180s wall and come back as silently TRUNCATED content (pilot: two answers
 * cut mid-word at 456/325 chars, both on 160-211s calls) — while the real
 * agent path always streams, where progressing generations run for minutes.
 * Streaming keeps the bench on the product path AND removes the artifact.
 */
async function chat(url, key, body, tries = 3) {
  for (let a = 0; ; a++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
        body: JSON.stringify(body.stream === false ? body : { ...body, stream: true }),
      });
      if (res.status === 429 || res.status >= 500) throw new Error(`HTTP ${res.status}`);
      if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`), { fatal: true });
      let content = "", reasoning = "", finish = null;
      if (body.stream === false) {
        const j = await res.json();
        const m = j.choices?.[0]?.message ?? {};
        content = typeof m.content === "string" ? m.content : "";
        reasoning = typeof m.reasoning === "string" ? m.reasoning : "";
        finish = j.choices?.[0]?.finish_reason ?? null;
      } else {
        const dec = new TextDecoder();
        let buf = "";
        for await (const raw of res.body) {
          buf += dec.decode(raw, { stream: true });
          let i;
          while ((i = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, i).trim();
            buf = buf.slice(i + 1);
            if (!line.startsWith("data:")) continue;
            const p = line.slice(5).trim();
            if (p === "[DONE]") continue;
            try {
              const c = JSON.parse(p).choices?.[0];
              if (typeof c?.delta?.content === "string") content += c.delta.content;
              if (typeof c?.delta?.reasoning === "string") reasoning += c.delta.reasoning;
              if (c?.finish_reason) finish = c.finish_reason;
            } catch { /* partial line */ }
          }
        }
      }
      const text = (content.trim().length > 0 ? content : reasoning)
        .replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
      return { text, finish };
    } catch (e) {
      if (e.fatal || a >= tries - 1) throw e;
      await new Promise((r) => setTimeout(r, 3000 * (a + 1)));
    }
  }
}
const solo = (model, prompt) => chat(OLLAMA, KEY, { model, max_tokens: 4096, messages: [{ role: "user", content: prompt }] });

// --- self-fusion pipeline (product prompts, replicated) ---------------------
const JUDGE_PROMPT =
  "You are an impartial judge. You are given the user's ORIGINAL REQUEST followed by several independent " +
  "expert answers to it. Assess the answers AGAINST THE REQUEST and respond with ONLY a JSON object with these keys: " +
  '"consensus", "disagreements", "unique_insights", "blind_spots", "partial_coverage", "hallucination_flags", ' +
  '"confidence" ("high"/"medium"/"low"), "fragile_claims". Judge factual correctness and how well each answer serves ' +
  "the request; do not reward verbosity. Output JSON only — no prose, no code fences.";

async function selfFusion(prompt) {
  const samples = (await Promise.all([0, 1, 2].map(() => solo("kimi-k2.7-code", prompt)))).map((r) => r.text);
  const experts = samples.map((s, i) => `EXPERT ${i + 1}:\n${s}`).join("\n\n---\n\n");
  let analysis = "{}";
  try {
    analysis = (await chat(OLLAMA, KEY, {
      model: "glm-5.2", max_tokens: 2048, temperature: 0,
      messages: [
        { role: "system", content: JUDGE_PROMPT },
        { role: "user", content: `ORIGINAL REQUEST:\n${prompt}\n\n${experts}` },
      ],
    })).text;
  } catch { /* judge failure degrades to raw answers, mirroring the product */ }
  const synthCtx =
    "A panel of expert models answered the user's request, and an impartial judge produced a structured " +
    "analysis of their answers. Write the single best final answer: take the actual content (code, formulas, " +
    "exact text) from the expert answers, and use the judge analysis to resolve disagreements, cover blind " +
    "spots, and weight the consensus. Where the judge listed partial_coverage, take each such aspect from the " +
    "expert that covered it — do not average partially covered aspects away. Do not drop detail that only one " +
    "expert provided unless it is wrong. If the judge flagged hallucination_flags, treat those items as suspect.\n\n" +
    `JUDGE ANALYSIS (JSON):\n${analysis}\n\nEXPERT ANSWERS:\n${experts}`;
  return chat(OLLAMA, KEY, {
    model: "kimi-k2.7-code", max_tokens: 4096,
    messages: [{ role: "system", content: synthCtx }, { role: "user", content: prompt }],
  });
}

// --- blind scoring -----------------------------------------------------------
/** First balanced {...} object in a string (greedy regex breaks on prose+code). */
function extractJson(s) {
  const start = s.indexOf("{");
  if (start === -1) throw new Error(`no JSON object in scorer reply: ${s.slice(0, 120)}`);
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (esc) { esc = false; continue; }
    if (c === "\\") { esc = true; continue; }
    if (c === '"') inStr = !inStr;
    if (inStr) continue;
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return s.slice(start, i + 1);
  }
  throw new Error(`unbalanced JSON in scorer reply: ${s.slice(0, 120)}`);
}

/** First balanced [...] array in a string, wherever it sits (prose before/after tolerated). */
function extractJsonArray(s) {
  const start = s.indexOf("[");
  if (start === -1) throw new Error(`no JSON array in scorer reply: ${s.slice(0, 120)}`);
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (esc) { esc = false; continue; }
    if (c === "\\") { esc = true; continue; }
    if (c === '"') inStr = !inStr;
    if (inStr) continue;
    if (c === "[") depth++;
    else if (c === "]" && --depth === 0) return s.slice(start, i + 1);
  }
  throw new Error(`unbalanced JSON array in scorer reply: ${s.slice(0, 120)}`);
}

async function score(task, answers /* {cond, text}[] */) {
  const shuffled = [...answers].sort(() => Math.random() - 0.5);
  const labels = shuffled.map((a, i) => ({ label: `S${i + 1}`, ...a }));
  // Cap per-answer length before joining: an unusually verbose answer (seen up
  // to ~19k chars) blows up the joint scorer prompt across 7 conditions and the
  // scorer's JSON reply comes back truncated/malformed (harness lesson, R0).
  const MAX_ANSWER_CHARS = 8000;
  const block = labels
    .map((l) => {
      const text = l.text || "(empty answer)";
      const capped = text.length > MAX_ANSWER_CHARS
        ? `${text.slice(0, MAX_ANSWER_CHARS)}\n...[truncated at ${MAX_ANSWER_CHARS} chars for scoring — score only what is shown]`
        : text;
      return `### ${l.label}\n${capped}`;
    })
    .join("\n\n");
  const rubric =
    "You are a strict, impartial evaluator. Score EACH answer against the TASK and the EXPECTED CRITERIA on three axes, " +
    "integers 0-10: accuracy (technically correct, criteria satisfied), completeness (covers all required aspects), " +
    "truthfulness (no fabricated APIs, flags, RFCs, or facts; hedged uncertainty is fine, confident fabrication is 0-3). " +
    "Grade HARSHLY and use the full scale: 10 means flawless expert work; deduct for every missed criterion; a typical " +
    "good answer lands 6-8. An answer that is cut off mid-text is scored only on what is present. " +
    `Respond with EXACTLY ${labels.length} lines and nothing else — one line per answer, each line a single ` +
    'standalone JSON object: {"label":"S1","accuracy":n,"completeness":n,"truthfulness":n,"note":"<=15 words"}. ' +
    "No surrounding array, no code fences, no blank lines, no prose before or after.";
  // No response_format here: json_object mode forces ONE top-level JSON value for
  // the whole reply, which is incompatible with N independent per-line objects.
  const ask = (messages) => chat(OLLAMA, KEY, { model: "deepseek-v4-pro", max_tokens: 4000, temperature: 0, messages });
  const messages = [
    { role: "system", content: rubric },
    { role: "user", content: `TASK:\n${task.prompt}\n\nEXPECTED CRITERIA:\n${task.criteria}\n\nANSWERS:\n${block}` },
  ];

  /**
   * One JSON object per line, fault-isolated: a line that fails to parse (bad
   * syntax, wrong shape, stray prose) is skipped rather than sinking the whole
   * task's score — the exact failure mode a single joint {"scores":[...]} object
   * had (SyntaxError on one bad character, or a missing "scores" key, killed
   * every condition's score at once, R0/R1).
   *
   * Tolerates the model reverting to its default habit of wrapping everything
   * in one array or a {"scores":[...]} object DESPITE the one-per-line
   * instruction (observed live, R2: every line started with "[" so the
   * per-line pass matched zero lines) — after the per-line pass, two whole-
   * document fallbacks run in order: a bare/prose-wrapped [...] array anywhere
   * in the text, then a {"scores":[...]}-shaped object anywhere in the text.
   * Both tolerate leading/trailing prose (offline-tested: a naive
   * `text.trim().startsWith("[")` check missed any preamble before the
   * bracket — fixed by searching for "[" wherever it occurs instead).
   */
  function parseLines(raw) {
    const out = new Map();
    const cleaned = raw.replace(/```(?:json)?/gi, "");
    for (const line of cleaned.split("\n")) {
      const t = line.trim();
      if (!t.startsWith("{")) continue;
      try {
        const obj = JSON.parse(extractJson(t));
        if (typeof obj.label === "string") out.set(obj.label, obj);
      } catch { /* this line is unrecoverable — the rest of the response is unaffected */ }
    }
    if (out.size === 0) {
      try {
        const arr = JSON.parse(extractJsonArray(cleaned));
        if (Array.isArray(arr)) for (const obj of arr) if (obj && typeof obj.label === "string") out.set(obj.label, obj);
      } catch { /* no bare array recovered — try the {"scores":[...]} shape next */ }
    }
    if (out.size === 0) {
      try {
        const whole = JSON.parse(extractJson(cleaned));
        const arr = Array.isArray(whole.scores) ? whole.scores : Array.isArray(whole) ? whole : [];
        for (const obj of arr) if (obj && typeof obj.label === "string") out.set(obj.label, obj);
      } catch { /* neither shape recovered anything — out stays empty, handled by the caller */ }
    }
    return out;
  }

  const parsed = parseLines((await ask(messages)).text);
  const missing = labels.filter((l) => !parsed.has(l.label));
  if (missing.length > 0) {
    // One retry, scoped to exactly the missing labels — labels that already
    // parsed cleanly are not re-litigated.
    const retryRaw = (await ask([
      ...messages,
      {
        role: "user",
        content: `Missing or unparseable lines for: ${missing.map((l) => l.label).join(", ")}. ` +
          "Output ONLY the missing line(s), same one-JSON-object-per-line format, nothing else.",
      },
    ])).text;
    for (const [k, v] of parseLines(retryRaw)) parsed.set(k, v);
  }

  // Labels that STILL didn't recover are dropped, not defaulted to a fake 0 —
  // a 0/0/0 "score" would silently corrupt the aggregate as if every model
  // failed the task, when the truth is just "the scorer never told us" (R2).
  const stillMissing = labels.filter((l) => !parsed.has(l.label));
  if (stillMissing.length > 0) {
    console.error(`  score: unrecoverable for ${stillMissing.map((l) => l.cond).join(", ")} — omitted, not zeroed`);
  }
  return labels.filter((l) => parsed.has(l.label)).map((l) => {
    const s = parsed.get(l.label);
    return { cond: l.cond, accuracy: s.accuracy ?? 0, completeness: s.completeness ?? 0, truthfulness: s.truthfulness ?? 0, note: s.note ?? "" };
  });
}

// --- run ----------------------------------------------------------------------
const CONDITIONS = [
  ["solo-glm", (p) => solo("glm-5.2", p)],
  ["solo-kimi", (p) => solo("kimi-k2.7-code", p)],
  ["solo-deepseek-flash", (p) => solo("deepseek-v4-flash", p)],
  ["solo-gemini-flash", (p) => solo("gemini-3-flash-preview", p)],
  ["solo-qwen-coder", (p) => solo("qwen3-coder-next", p)],
  ["fusion", (p) => chat(PROXY, "local-no-auth", { model: "fusion-coder", max_tokens: 4096, messages: [{ role: "user", content: p }] })],
  // The delivery question: what an AGENT actually experiences — the smart
  // router may send a task to plain glm-5.2 (zero amplification by design).
  // Per-task comparison against `fusion` and `solo-glm` measures router recall.
  ["fusion-agents", (p) => chat(PROXY, "local-no-auth", { model: "fusion-agents", max_tokens: 4096, messages: [{ role: "user", content: p }] })],
  ["self-kimi", selfFusion],
];

// --- running stats + stability, printed after EVERY task -----------------------
function printSummary(results, note) {
  const agg = {};
  for (const r of results) {
    if (!r.scores) continue;
    for (const s of r.scores) {
      agg[s.cond] ??= { acc: 0, comp: 0, truth: 0, n: 0 };
      agg[s.cond].acc += s.accuracy; agg[s.cond].comp += s.completeness; agg[s.cond].truth += s.truthfulness; agg[s.cond].n += 1;
    }
  }
  console.log(`\n--- running stats (${note}) ---`);
  console.log("cond          acc   comp  truth  total(avg)  n");
  for (const [cond, a] of Object.entries(agg)) {
    const t = (a.acc + a.comp + a.truth) / a.n;
    console.log(`${cond.padEnd(12)} ${(a.acc / a.n).toFixed(2)}  ${(a.comp / a.n).toFixed(2)}  ${(a.truth / a.n).toFixed(2)}   ${t.toFixed(2)}      ${a.n}`);
  }
}

const results = [...resumed];
let runningErrors = 0, runningAnswers = 0;
for (const task of tasks) {
  console.error(`[${task.id}] generating…`);
  const answers = [];
  for (const [cond, fn] of CONDITIONS) {
    try {
      const t0 = Date.now();
      const { text, finish } = await fn(task.prompt);
      answers.push({ cond, text, finish, ms: Date.now() - t0 });
      console.error(`  ${cond}: ${text.length} chars, finish=${finish}, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    } catch (e) {
      answers.push({ cond, text: "", finish: "error", ms: -1, error: String(e).slice(0, 120) });
      console.error(`  ${cond}: ERROR ${e}`);
    }
  }
  const taskErrors = answers.filter((a) => a.finish === "error").length;
  runningErrors += taskErrors;
  runningAnswers += answers.length;
  try {
    const scored = await score(task, answers);
    results.push({ id: task.id, cat: task.cat, scores: scored, times: Object.fromEntries(answers.map((a) => [a.cond, a.ms])), answers: Object.fromEntries(answers.map((a) => [a.cond, a.text])) });
    console.error(`  scored: ${scored.map((s) => `${s.cond}=${s.accuracy}/${s.completeness}/${s.truthfulness}`).join(" ")}`);
  } catch (e) {
    console.error(`  scoring FAILED: ${e}`);
    results.push({ id: task.id, cat: task.cat, scores: null, error: String(e).slice(0, 200), times: Object.fromEntries(answers.map((a) => [a.cond, a.ms])), answers: Object.fromEntries(answers.map((a) => [a.cond, a.text])) });
  }
  console.error(`  stability: ${taskErrors} error(s) this task, ${runningErrors}/${runningAnswers} total call-errors so far`);
  writeFileSync(OUT, JSON.stringify(results, null, 2));
  printSummary(results, `after ${results.length}/${allTasks.length} tasks`);
}

// --- final summary ---------------------------------------------------------------
printSummary(results, "FINAL");
console.log(`\nstability: ${runningErrors}/${runningAnswers} total call-errors across the run`);
console.log(`results: ${OUT}`);
