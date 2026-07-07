#!/usr/bin/env node
// Per-answer control re-score of solo-glm (methodology parity with synthglm-exp).
import { readFileSync, writeFileSync } from "node:fs";
const KEY = process.env.OLLAMA_API_KEY;
const tasks = JSON.parse(readFileSync("bench/tasks.json", "utf8"));
const base = JSON.parse(readFileSync("bench/results-full.json", "utf8"));
async function stream(body) {
  const res = await fetch("https://ollama.com/v1/chat/completions", { method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
    body: JSON.stringify({ ...body, stream: true }) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const dec = new TextDecoder(); let buf = "", content = "", reasoning = "";
  for await (const raw of res.body) { buf += dec.decode(raw, { stream: true }); let i;
    while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith("data:")) continue; const p = line.slice(5).trim(); if (p === "[DONE]") continue;
      try { const c = JSON.parse(p).choices?.[0];
        if (typeof c?.delta?.content === "string") content += c.delta.content;
        if (typeof c?.delta?.reasoning === "string") reasoning += c.delta.reasoning; } catch {} } }
  return (content.trim() || reasoning).replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}
function extractJson(s){const a=s.indexOf("{");if(a<0)throw new Error("no JSON");let d=0,q=false,e=false;
 for(let i=a;i<s.length;i++){const c=s[i];if(e){e=false;continue}if(c==="\\"){e=true;continue}
  if(c==='"')q=!q;if(q)continue;if(c==="{")d++;else if(c==="}"&&--d===0)return s.slice(a,i+1)}throw new Error("unbalanced")}
const RUBRIC = "You are a strict, impartial evaluator. Score the ANSWER against the TASK and the EXPECTED CRITERIA on three axes, integers 0-10: accuracy, completeness, truthfulness (confident fabrication is 0-3). Grade HARSHLY, full scale; typical good answers land 6-8. Respond with ONLY JSON: {\"accuracy\":n,\"completeness\":n,\"truthfulness\":n,\"note\":\"<=15 words\"}";
const out = [];
for (const task of tasks) {
  const text = base.find((t) => t.id === task.id)?.answers?.["solo-glm"];
  if (!text) continue;
  const messages = [ { role: "system", content: RUBRIC },
    { role: "user", content: `TASK:\n${task.prompt}\n\nEXPECTED CRITERIA:\n${task.criteria}\n\nANSWER:\n${text}` } ];
  let raw = await stream({ model: "deepseek-v4-pro", max_tokens: 2000, temperature: 0, response_format: { type: "json_object" }, messages });
  let s; try { s = JSON.parse(extractJson(raw)); } catch {
    raw = await stream({ model: "deepseek-v4-pro", max_tokens: 2000, temperature: 0, response_format: { type: "json_object" },
      messages: [...messages, { role: "assistant", content: raw.slice(0, 3000) }, { role: "user", content: "Now output ONLY the JSON object. No other text." }] });
    s = JSON.parse(extractJson(raw)); }
  out.push({ id: task.id, accuracy: s.accuracy, completeness: s.completeness, truthfulness: s.truthfulness });
  console.error(`${task.id}: ${s.accuracy}/${s.completeness}/${s.truthfulness}`);
  writeFileSync("bench/results-sologlm-pa.json", JSON.stringify(out, null, 2));
}
const t = out.reduce((a, x) => a + x.accuracy + x.completeness + x.truthfulness, 0) / out.length / 30;
console.log(`solo-glm (per-answer control): ${t.toFixed(2)} (n=${out.length})`);
