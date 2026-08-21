#!/usr/bin/env node
/**
 * fusion-bench — measures whether the fused virtual model beats every
 * individual panel member (the project's mission statement), and whether
 * self-fusion (same model ×3) approaches the mixed panel (OpenRouter's
 * synthesis>diversity hypothesis, R0).
 *
 * fusion-coder panel as of this run (3 models — `fusion.yaml:112`). Re-run from
 * scratch, not resumed: the panel changed again since the 4-model run of
 * 2026-07-08, so those fusion/fusion-agents scores no longer describe the
 * current pipeline. Two conditions from that run are GONE, not dropped for
 * cost: `gemini-3-flash-preview` and `qwen3-coder-next` were retired upstream
 * on 2026-07-15 and now answer HTTP 410. Neither is on the panel any more.
 *   solo-glm         : glm-5.2 direct                (panel member AND synth)
 *   solo-kimi        : kimi-k2.7-code direct         (panel member; self-fusion baseline)
 *   solo-deepseek-pro: deepseek-v4-pro:0813-cloud direct (panel member as of 2026-08-21)
 *   fusion           : fusion-coder via the LOCAL PROXY (the real product path)
 *   fusion-agents    : the smart router — what an agent actually experiences
 *   self-kimi        : kimi ×3 samples -> judge glm -> synth kimi, replicated in-script
 *                with the product's judge/synth prompt texts (config forbids
 *                duplicate panel members, so this cannot run through the proxy)
 *
 * The mission claim under test: fusion beats EVERY individual panel member.
 * All three panel members are therefore solo conditions; every non-member is
 * out of scope for that claim and is not paid for.
 *
 * Scoring: blind, by TWO independent scorers, three axes 0-10 (accuracy,
 * completeness, truthfulness) against per-task criteria; answers are shuffled
 * and anonymized per task, and both scorers see the identical shuffled block.
 * The reported score per condition is the mean of the two; each scorer's raw
 * numbers are kept in `byScorer` so disagreement stays auditable.
 *
 * Scorer choice is a correctness constraint, not a preference: deepseek-v4-pro
 * was the sole scorer until 2026-08-21, when it JOINED the fusion-coder panel —
 * it would now be grading its own answer. Every model inside the pipeline
 * (kimi-k2.7-code, glm-5.2, deepseek-v4-pro, deepseek-v4-flash) is therefore
 * disqualified. gpt-oss:120b and minimax-m3 are the two capable models on this
 * provider that sit outside it entirely.
 *
 * Usage: node bench/fusion-bench.mjs [--tasks N] [--out bench/results.json]
 */
import { readFileSync, writeFileSync } from "node:fs";

const OLLAMA = "https://ollama.com/v1/chat/completions";
// `server.port` in fusion.yaml is 8081; the old hard-coded 8080 silently turned
// both proxy conditions into connection errors. Env override for a non-default run.
const PROXY = process.env.FUSION_PROXY_URL ?? "http://127.0.0.1:8081/v1/chat/completions";
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
  // The proxy pools upstream connectors behind a circuit breaker with a 60s
  // cooldown (`connector_cooldown_s`, src/config.ts). One transient upstream 503
  // therefore blackholes the proxy for a full minute — while the solo conditions,
  // which call the provider directly, just retry and succeed. A 3s/6s backoff
  // lands entirely inside that cooldown, so the proxy conditions would score 0
  // for an outage the baselines never felt. Wait the breaker out instead.
  const isProxy = url === PROXY;
  const backoff = (attempt) => (isProxy ? [5000, 65000, 65000][attempt] ?? 65000 : 3000 * (attempt + 1));
  if (isProxy) tries = 4;
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
      // A reasoning model that spends its whole budget thinking returns
      // finish=length with EMPTY content. The `content || reasoning` fallback
      // above then hands the scorers a raw chain-of-thought dump and they grade
      // it ~1/10 — which measures the token cap, not the model. Flag it so the
      // caller can drop the answer instead of scoring the cap.
      const reasoningOnly = finish === "length" && content.trim().length === 0 && reasoning.trim().length > 0;
      return { text, finish, reasoningOnly };
    } catch (e) {
      if (e.fatal || a >= tries - 1) throw e;
      await new Promise((r) => setTimeout(r, backoff(a)));
    }
  }
}
/**
 * Generation budget for EVERY condition, solo and fused alike.
 *
 * 4096 (the previous value) is not a neutral choice once a reasoning model is
 * on the panel: deepseek-v4-pro:0813-cloud spent the whole 4096 on `reasoning`
 * and returned finish=length with EMPTY content, so the harness fell back to
 * printing its raw chain-of-thought and both scorers correctly graded it ~0.5/10
 * — measuring the cap, not the model. 16384 was not enough either: on T03 the
 * same model burned all 16384 on reasoning (63928 chars, finish=length) and
 * scored 1/0.5/1.5. A baseline the harness truncates would hand the "fusion
 * beats every panel member" claim a win it did not earn, so the budget goes to
 * 32768 for EVERY condition and `reasoningOnly` answers are dropped from
 * scoring rather than graded as if the model had answered.
 */
const GEN_TOKENS = 32768;
const solo = (model, prompt) => chat(OLLAMA, KEY, { model, max_tokens: GEN_TOKENS, messages: [{ role: "user", content: prompt }] });

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
    model: "kimi-k2.7-code", max_tokens: GEN_TOKENS,
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

/** Models outside every condition — see the scorer-independence note in the header. */
const SCORERS = ["gpt-oss:120b", "minimax-m3"];

async function score(task, answers /* {cond, text}[] */) {
  /**
   * Two kinds of non-answer are dropped from scoring rather than graded:
   *
   *  - `reasoningOnly` — the model spent its whole budget thinking and returned
   *    empty content. Scoring it grades the token cap, not the model.
   *  - `finish === "error"` / empty text — the upstream call failed after every
   *    retry. Observed live: one HTTP 503 gave solo-deepseek-pro an empty answer
   *    on T01 and the scorers rated it 5/30, dropping its average from ~26 to
   *    18.75. That is an availability event being counted as answer quality —
   *    and since solo-deepseek-pro is a PANEL MEMBER, it would have handed
   *    "fusion beats every panel member" a win bought with someone else's outage.
   *
   * Dropping biases the affected condition upward, which is the safe direction:
   * it makes the product's claim harder to prove, not easier. Availability is
   * not lost — it is reported separately as the call-error count. Every drop
   * is printed.
   */
  const isNonAnswer = (a) => a.reasoningOnly || a.finish === "error" || a.text.trim().length === 0;
  const dropped = answers.filter(isNonAnswer);
  if (dropped.length > 0) {
    console.error(`  score: dropped ${dropped.map((d) => `${d.cond}(${d.reasoningOnly ? "reasoning-only" : d.finish})`).join(", ")} — not an answer, not scored`);
  }
  const scorable = answers.filter((a) => !isNonAnswer(a));
  const shuffled = [...scorable].sort(() => Math.random() - 0.5);
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

  /** One scorer's full pass over every label, with the scoped retry. */
  async function runScorer(model) {
    const ask = (msgs) => chat(OLLAMA, KEY, { model, max_tokens: 4000, temperature: 0, messages: msgs });
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
    return parsed;
  }

  // Both scorers see the IDENTICAL shuffled block, so their disagreement is
  // about the answers and not about a different anonymisation draw.
  const passes = await Promise.all(SCORERS.map(async (m) => {
    try {
      const parsed = await runScorer(m);
      // A scorer that ANSWERS but whose reply recovers zero labels degrades the
      // task to single-rater in silence — the throw path below never fires.
      // Observed on T01: minimax-m3 returned, parsed to nothing, and only the
      // `raters` field in the results JSON showed it.
      if (parsed.size === 0) console.error(`  score: scorer ${m} returned nothing parseable — task falls back to the other rater`);
      return { model: m, parsed };
    } catch (e) {
      // One dead scorer must not sink the task: the other still produces a
      // usable (single-rater, and labelled as such) score.
      console.error(`  score: scorer ${m} FAILED (${String(e).slice(0, 100)}) — task falls back to the other rater`);
      return { model: m, parsed: new Map() };
    }
  }));

  const AXES = ["accuracy", "completeness", "truthfulness"];
  // Labels that no scorer recovered are dropped, not defaulted to a fake 0 —
  // a 0/0/0 "score" would silently corrupt the aggregate as if every model
  // failed the task, when the truth is just "the scorers never told us" (R2).
  const stillMissing = labels.filter((l) => !passes.some((p) => p.parsed.has(l.label)));
  if (stillMissing.length > 0) {
    console.error(`  score: unrecoverable for ${stillMissing.map((l) => l.cond).join(", ")} — omitted, not zeroed`);
  }
  return labels
    .filter((l) => passes.some((p) => p.parsed.has(l.label)))
    .map((l) => {
      const raters = passes.filter((p) => p.parsed.has(l.label));
      const out = { cond: l.cond, raters: raters.length, byScorer: {} };
      for (const p of raters) {
        const s = p.parsed.get(l.label);
        out.byScorer[p.model] = {
          accuracy: s.accuracy ?? 0, completeness: s.completeness ?? 0,
          truthfulness: s.truthfulness ?? 0, note: s.note ?? "",
        };
      }
      // Mean across whichever raters answered for this label.
      for (const axis of AXES) {
        out[axis] = raters.reduce((a, p) => a + (p.parsed.get(l.label)[axis] ?? 0), 0) / raters.length;
      }
      out.note = raters.map((p) => `${p.model}: ${p.parsed.get(l.label).note ?? ""}`).join(" | ");
      return out;
    });
}

// --- run ----------------------------------------------------------------------
const CONDITIONS = [
  // One solo condition per CURRENT panel member (fusion.yaml:112), using the
  // exact model id the panel uses — a baseline against a different snapshot of
  // the same family would not be the same model.
  ["solo-glm", (p) => solo("glm-5.2", p)],
  ["solo-kimi", (p) => solo("kimi-k2.7-code", p)],
  ["solo-deepseek-pro", (p) => solo("deepseek-v4-pro:0813-cloud", p)],
  ["fusion", (p) => chat(PROXY, "local-no-auth", { model: "fusion-coder", max_tokens: GEN_TOKENS, messages: [{ role: "user", content: p }] })],
  // Same pipeline, web grounding off. The solos get no web context either, so
  // this is the like-for-like test of the panel/judge/synth mechanism; the gap
  // between `fusion` and `fusion-noweb` is what Tavily contributes.
  ["fusion-noweb", (p) => chat(PROXY, "local-no-auth", { model: "fusion-coder-noweb", max_tokens: GEN_TOKENS, messages: [{ role: "user", content: p }] })],
  // The delivery question: what an AGENT actually experiences — the smart
  // router may send a task to plain glm-5.2 (zero amplification by design).
  // Per-task comparison against `fusion` and `solo-glm` measures router recall.
  ["fusion-agents", (p) => chat(PROXY, "local-no-auth", { model: "fusion-agents", max_tokens: GEN_TOKENS, messages: [{ role: "user", content: p }] })],
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
      const { text, finish, reasoningOnly } = await fn(task.prompt);
      answers.push({ cond, text, finish, reasoningOnly: reasoningOnly === true, ms: Date.now() - t0 });
      console.error(`  ${cond}: ${text.length} chars, finish=${finish}${reasoningOnly ? " REASONING-ONLY" : ""}, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
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
