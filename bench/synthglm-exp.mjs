#!/usr/bin/env node
/**
 * synth=glm experiment — reruns ONLY the fusion condition against the isolated
 * 8081 proxy (bench/fusion-synthglm.yaml: fusion-coder with synth glm-5.2,
 * kimi stays on the panel), scores per-answer, and prints the delta against
 * the baseline fusion scores from results-full.json (+overlay).
 */
import { readFileSync, writeFileSync } from "node:fs";

const KEY = process.env.OLLAMA_API_KEY;
if (!KEY) { console.error("OLLAMA_API_KEY required"); process.exit(1); }
const tasks = JSON.parse(readFileSync("bench/tasks.json", "utf8"));

async function stream(url, key, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 150)}`);
  const dec = new TextDecoder();
  let buf = "", content = "", reasoning = "";
  for await (const raw of res.body) {
    buf += dec.decode(raw, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith("data:")) continue;
      const p = line.slice(5).trim();
      if (p === "[DONE]") continue;
      try {
        const c = JSON.parse(p).choices?.[0];
        if (typeof c?.delta?.content === "string") content += c.delta.content;
        if (typeof c?.delta?.reasoning === "string") reasoning += c.delta.reasoning;
      } catch {}
    }
  }
  return (content.trim().length > 0 ? content : reasoning).replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

function extractJson(s) {
  const start = s.indexOf("{");
  if (start === -1) throw new Error("no JSON");
  let d = 0, q = false, e = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (e) { e = false; continue; }
    if (c === "\\") { e = true; continue; }
    if (c === '"') q = !q;
    if (q) continue;
    if (c === "{") d++;
    else if (c === "}" && --d === 0) return s.slice(start, i + 1);
  }
  throw new Error("unbalanced");
}

const RUBRIC =
  "You are a strict, impartial evaluator. Score the ANSWER against the TASK and the EXPECTED CRITERIA on three axes, " +
  "integers 0-10: accuracy, completeness, truthfulness (confident fabrication is 0-3). Grade HARSHLY, full scale; " +
  'typical good answers land 6-8. Respond with ONLY JSON: {"accuracy":n,"completeness":n,"truthfulness":n,"note":"<=15 words"}';

async function scoreOne(task, text) {
  const messages = [
    { role: "system", content: RUBRIC },
    { role: "user", content: `TASK:\n${task.prompt}\n\nEXPECTED CRITERIA:\n${task.criteria}\n\nANSWER:\n${text || "(empty)"}` },
  ];
  let raw = await stream("https://ollama.com/v1/chat/completions", KEY, { model: "deepseek-v4-pro", max_tokens: 2000, temperature: 0, response_format: { type: "json_object" }, messages });
  try { return JSON.parse(extractJson(raw)); }
  catch {
    raw = await stream("https://ollama.com/v1/chat/completions", KEY, { model: "deepseek-v4-pro", max_tokens: 2000, temperature: 0, response_format: { type: "json_object" },
      messages: [...messages, { role: "assistant", content: raw.slice(0, 3000) }, { role: "user", content: "Now output ONLY the JSON object. No other text." }] });
    return JSON.parse(extractJson(raw));
  }
}

const out = [];
for (const task of tasks) {
  try {
    const t0 = Date.now();
    const text = await stream("http://127.0.0.1:8081/v1/chat/completions", "local-no-auth",
      { model: "fusion-coder", max_tokens: 4096, messages: [{ role: "user", content: task.prompt }] });
    const s = await scoreOne(task, text);
    out.push({ id: task.id, cat: task.cat, accuracy: s.accuracy, completeness: s.completeness, truthfulness: s.truthfulness, note: s.note, chars: text.length, ms: Date.now() - t0 });
    console.error(`${task.id}: ${s.accuracy}/${s.completeness}/${s.truthfulness} (${text.length}ch, ${((Date.now() - t0) / 1000).toFixed(0)}s) «${s.note}»`);
  } catch (e) {
    out.push({ id: task.id, cat: task.cat, error: String(e).slice(0, 150) });
    console.error(`${task.id}: ERROR ${e}`);
  }
  writeFileSync("bench/results-synthglm.json", JSON.stringify(out, null, 2));
}

// Delta vs baseline fusion
const base = JSON.parse(readFileSync("bench/results-full.json", "utf8"));
let ov = {}; try { ov = JSON.parse(readFileSync("bench/rescore-overlay.json", "utf8")); } catch {}
for (const r of base) if (ov[r.id]) r.scores = ov[r.id];
let nSum = 0, bSum = 0, n = 0;
console.log("\ntask  synth=glm  synth=kimi(base)");
for (const r of out) {
  if (r.error) continue;
  const b = base.find((t) => t.id === r.id)?.scores?.find((s) => s.cond === "fusion");
  if (!b) continue;
  const nt = r.accuracy + r.completeness + r.truthfulness, bt = b.accuracy + b.completeness + b.truthfulness;
  nSum += nt; bSum += bt; n++;
  console.log(`${r.id}   ${String(nt).padStart(2)}/30      ${String(bt).padStart(2)}/30`);
}
console.log(`\nfusion(synth=glm): ${(nSum / n / 30).toFixed(2)}  vs  fusion(synth=kimi): ${(bSum / n / 30).toFixed(2)}  (n=${n})`);
