#!/usr/bin/env node
/**
 * Re-score one task's SAVED answers with the CURRENT criteria from tasks.json
 * (used when a criteria bug is found mid-run — generation is never repeated).
 * Writes/updates bench/rescore-overlay.json: { "<taskId>": scores[] }.
 * summarize.mjs merges the overlay over the base results.
 * Usage: node bench/rescore.mjs T04 [results.json]
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const id = process.argv[2];
const file = process.argv[3] ?? "bench/results-full.json";
if (!id) { console.error("usage: rescore.mjs <taskId> [results.json]"); process.exit(1); }
const KEY = process.env.OLLAMA_API_KEY;
if (!KEY) { console.error("OLLAMA_API_KEY required"); process.exit(1); }

const task = JSON.parse(readFileSync("bench/tasks.json", "utf8")).find((t) => t.id === id);
const rec = JSON.parse(readFileSync(file, "utf8")).find((t) => t.id === id);
if (!task || !rec?.answers) { console.error(`no task/answers for ${id}`); process.exit(1); }

function extractJson(s) {
  const start = s.indexOf("{");
  if (start === -1) throw new Error(`no JSON: ${s.slice(0, 120)}`);
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
  throw new Error("unbalanced JSON");
}

async function chatText(body) {
  const res = await fetch("https://ollama.com/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
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

// Per-answer scoring: one answer per scorer call. Joint 6-answer scoring
// drowned the scorer in ~100k chars and it narrated instead of emitting JSON
// (T05/T09 failed even with a conversion retry). Shorter inputs are reliable;
// the rubric is criteria-anchored, so comparative context is not required.
const rubric =
  "You are a strict, impartial evaluator. Score the ANSWER against the TASK and the EXPECTED CRITERIA on three axes, " +
  "integers 0-10: accuracy, completeness, truthfulness (confident fabrication is 0-3). Grade HARSHLY, full scale; " +
  'typical good answers land 6-8. An answer cut off mid-text is scored on what is present. ' +
  'Respond with ONLY JSON: {"accuracy":n,"completeness":n,"truthfulness":n,"note":"<=15 words"}';
const scores = [];
for (const [cond, text] of Object.entries(rec.answers)) {
  const messages = [
    { role: "system", content: rubric },
    { role: "user", content: `TASK:\n${task.prompt}\n\nEXPECTED CRITERIA:\n${task.criteria}\n\nANSWER:\n${text || "(empty answer)"}` },
  ];
  let raw = await chatText({ model: "deepseek-v4-pro", max_tokens: 2000, temperature: 0, response_format: { type: "json_object" }, messages });
  let s;
  try { s = JSON.parse(extractJson(raw)); }
  catch {
    raw = await chatText({ model: "deepseek-v4-pro", max_tokens: 2000, temperature: 0, response_format: { type: "json_object" },
      messages: [...messages, { role: "assistant", content: raw.slice(0, 3000) }, { role: "user", content: "Now output ONLY the JSON object with the scores. No other text." }] });
    s = JSON.parse(extractJson(raw));
  }
  scores.push({ cond, accuracy: s.accuracy ?? 0, completeness: s.completeness ?? 0, truthfulness: s.truthfulness ?? 0, note: s.note ?? "" });
  console.error(`  ${cond}: ${s.accuracy}/${s.completeness}/${s.truthfulness}`);
}
const overlayFile = "bench/rescore-overlay.json";
const overlay = existsSync(overlayFile) ? JSON.parse(readFileSync(overlayFile, "utf8")) : {};
overlay[id] = scores;
writeFileSync(overlayFile, JSON.stringify(overlay, null, 2));
console.log(`${id} re-scored:`, scores.map((s) => `${s.cond}=${s.accuracy}/${s.completeness}/${s.truthfulness}`).join(" "));
