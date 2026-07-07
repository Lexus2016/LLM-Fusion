#!/usr/bin/env node
/**
 * fusion-bench — measures whether the fused virtual model beats every
 * individual panel member (the project's mission statement), and whether
 * self-fusion (same model ×3) approaches the mixed panel (OpenRouter's
 * synthesis>diversity hypothesis, R0).
 *
 * Conditions per task:
 *   solo-glm   : glm-5.2 direct
 *   solo-kimi  : kimi-k2.7-code direct        (also the fusion synth — self-fusion baseline)
 *   solo-gptoss: gpt-oss:120b direct
 *   fusion     : fusion-coder via the LOCAL PROXY (the real product path)
 *   self-kimi  : kimi ×3 samples -> judge glm -> synth kimi, replicated in-script
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

const tasks = JSON.parse(readFileSync(new URL("./tasks.json", import.meta.url), "utf8")).slice(0, N);

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

async function score(task, answers /* {cond, text}[] */) {
  const shuffled = [...answers].sort(() => Math.random() - 0.5);
  const labels = shuffled.map((a, i) => ({ label: `S${i + 1}`, ...a }));
  const block = labels.map((l) => `### ${l.label}\n${l.text || "(empty answer)"}`).join("\n\n");
  const rubric =
    "You are a strict, impartial evaluator. Score EACH answer against the TASK and the EXPECTED CRITERIA on three axes, " +
    "integers 0-10: accuracy (technically correct, criteria satisfied), completeness (covers all required aspects), " +
    "truthfulness (no fabricated APIs, flags, RFCs, or facts; hedged uncertainty is fine, confident fabrication is 0-3). " +
    "Grade HARSHLY and use the full scale: 10 means flawless expert work; deduct for every missed criterion; a typical " +
    "good answer lands 6-8. An answer that is cut off mid-text is scored only on what is present. " +
    'Respond with ONLY JSON, no reasoning text: {"scores":[{"label":"S1","accuracy":n,"completeness":n,"truthfulness":n,"note":"<=15 words"},...]}';
  const ask = (messages) => chat(OLLAMA, KEY, {
    model: "deepseek-v4-pro", max_tokens: 4000, temperature: 0,
    response_format: { type: "json_object" }, messages,
  });
  const messages = [
    { role: "system", content: rubric },
    { role: "user", content: `TASK:\n${task.prompt}\n\nEXPECTED CRITERIA:\n${task.criteria}\n\nANSWERS:\n${block}` },
  ];
  let raw = (await ask(messages)).text;
  let parsed;
  try {
    parsed = JSON.parse(extractJson(raw));
  } catch {
    // The scorer narrated instead of emitting JSON — one conversion retry.
    raw = (await ask([...messages, { role: "assistant", content: raw.slice(0, 4000) },
      { role: "user", content: "Now output ONLY the JSON object with the scores. No other text." }])).text;
    parsed = JSON.parse(extractJson(raw));
  }
  return labels.map((l) => {
    const s = parsed.scores.find((x) => x.label === l.label) ?? {};
    return { cond: l.cond, accuracy: s.accuracy ?? 0, completeness: s.completeness ?? 0, truthfulness: s.truthfulness ?? 0, note: s.note ?? "" };
  });
}

// --- run ----------------------------------------------------------------------
const CONDITIONS = [
  ["solo-glm", (p) => solo("glm-5.2", p)],
  ["solo-kimi", (p) => solo("kimi-k2.7-code", p)],
  ["solo-gptoss", (p) => solo("gpt-oss:120b", p)],
  ["fusion", (p) => chat(PROXY, "local-no-auth", { model: "fusion-coder", max_tokens: 4096, messages: [{ role: "user", content: p }] })],
  // The delivery question: what an AGENT actually experiences — the smart
  // router may send a task to plain glm-5.2 (zero amplification by design).
  // Per-task comparison against `fusion` and `solo-glm` measures router recall.
  ["fusion-agents", (p) => chat(PROXY, "local-no-auth", { model: "fusion-agents", max_tokens: 4096, messages: [{ role: "user", content: p }] })],
  ["self-kimi", selfFusion],
];

const results = [];
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
  try {
    const scored = await score(task, answers);
    results.push({ id: task.id, cat: task.cat, scores: scored, times: Object.fromEntries(answers.map((a) => [a.cond, a.ms])), answers: Object.fromEntries(answers.map((a) => [a.cond, a.text])) });
    console.error(`  scored: ${scored.map((s) => `${s.cond}=${s.accuracy}/${s.completeness}/${s.truthfulness}`).join(" ")}`);
  } catch (e) {
    console.error(`  scoring FAILED: ${e}`);
    results.push({ id: task.id, cat: task.cat, scores: null, error: String(e).slice(0, 200), times: Object.fromEntries(answers.map((a) => [a.cond, a.ms])), answers: Object.fromEntries(answers.map((a) => [a.cond, a.text])) });
  }
  writeFileSync(OUT, JSON.stringify(results, null, 2));
}

// --- summary -------------------------------------------------------------------
const agg = {};
for (const r of results) {
  if (!r.scores) continue;
  for (const s of r.scores) {
    agg[s.cond] ??= { acc: 0, comp: 0, truth: 0, n: 0 };
    agg[s.cond].acc += s.accuracy; agg[s.cond].comp += s.completeness; agg[s.cond].truth += s.truthfulness; agg[s.cond].n += 1;
  }
}
console.log("\ncond          acc   comp  truth  total(avg)  n");
for (const [cond, a] of Object.entries(agg)) {
  const t = (a.acc + a.comp + a.truth) / a.n;
  console.log(`${cond.padEnd(12)} ${(a.acc / a.n).toFixed(2)}  ${(a.comp / a.n).toFixed(2)}  ${(a.truth / a.n).toFixed(2)}   ${t.toFixed(2)}      ${a.n}`);
}
console.log(`\nresults: ${OUT}`);
