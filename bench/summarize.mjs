#!/usr/bin/env node
/**
 * fusion-bench summarizer — normalized 0.00-1.00 scores with visual bars.
 * Usage: node bench/summarize.mjs [results.json]  (default bench/results-full.json)
 * Works on partial results (the bench writes incrementally).
 */
import { readFileSync } from "node:fs";

const file = process.argv[2] ?? "bench/results-full.json";
const results = JSON.parse(readFileSync(file, "utf8"));
// Criteria-fix overlay: re-scored tasks replace their base scores (see rescore.mjs).
try {
  const overlay = JSON.parse(readFileSync("bench/rescore-overlay.json", "utf8"));
  for (const r of results) if (overlay[r.id]) r.scores = overlay[r.id];
} catch { /* no overlay */ }

const scored = results.filter((r) => r.scores);
if (scored.length === 0) { console.log("no scored tasks yet"); process.exit(0); }

const agg = {}; // cond -> {acc, comp, truth, n, perCat: {cat:{sum,n}}}
for (const r of scored) {
  for (const s of r.scores) {
    const a = (agg[s.cond] ??= { acc: 0, comp: 0, truth: 0, n: 0, perCat: {} });
    a.acc += s.accuracy; a.comp += s.completeness; a.truth += s.truthfulness; a.n += 1;
    const c = (a.perCat[r.cat] ??= { sum: 0, n: 0 });
    c.sum += s.accuracy + s.completeness + s.truthfulness; c.n += 1;
  }
}

const norm = (x) => x / 10; // axis 0-10 -> 0.00-1.00
const bar = (v, width = 24) => {
  const full = Math.round(v * width);
  return "█".repeat(full) + "░".repeat(width - full);
};

const rows = Object.entries(agg)
  .map(([cond, a]) => ({
    cond,
    acc: norm(a.acc / a.n),
    comp: norm(a.comp / a.n),
    truth: norm(a.truth / a.n),
    total: (a.acc + a.comp + a.truth) / a.n / 30,
    n: a.n,
    perCat: a.perCat,
  }))
  .sort((x, y) => y.total - x.total);

console.log(`\n== fusion-bench: ${scored.length} задач оцінено (${file}) ==\n`);
console.log("ЗАГАЛЬНА ОЦІНКА (0.00-1.00)\n");
for (const r of rows) {
  console.log(`  ${r.cond.padEnd(14)} ${bar(r.total)} ${r.total.toFixed(2)}   (n=${r.n})`);
}
console.log("\nПО ОСЯХ                точність   повнота   достовірність");
for (const r of rows) {
  console.log(`  ${r.cond.padEnd(14)}   ${r.acc.toFixed(2)}       ${r.comp.toFixed(2)}       ${r.truth.toFixed(2)}`);
}

// Per-category composite
const cats = [...new Set(scored.map((r) => r.cat))];
console.log(`\nПО КАТЕГОРІЯХ (композит 0.00-1.00)`);
console.log(`  ${"".padEnd(14)} ${cats.map((c) => c.padEnd(10)).join("")}`);
for (const r of rows) {
  const cells = cats.map((c) => {
    const pc = r.perCat[c];
    return (pc ? (pc.sum / pc.n / 30).toFixed(2) : "  -  ").padEnd(10);
  });
  console.log(`  ${r.cond.padEnd(14)} ${cells.join("")}`);
}

// Mission verdicts
const get = (c) => rows.find((r) => r.cond === c);
const fusion = get("fusion"), agents = get("fusion-agents"), selfK = get("self-kimi");
const solos = rows.filter((r) => r.cond.startsWith("solo-"));
const bestSolo = solos[0];
if (fusion && bestSolo) {
  const d = fusion.total - bestSolo.total;
  console.log(`\nВЕРДИКТИ МІСІЇ`);
  console.log(`  fusion vs найкраще соло (${bestSolo.cond}): ${d >= 0 ? "+" : ""}${d.toFixed(2)} ${d > 0.02 ? "✅ ПЕРЕВЕРШУЄ" : d < -0.02 ? "❌ ПОСТУПАЄТЬСЯ" : "≈ нарівні"}`);
}
if (selfK && fusion) {
  const d = selfK.total - fusion.total;
  console.log(`  self-fusion vs змішана панель: ${d >= 0 ? "+" : ""}${d.toFixed(2)} ${Math.abs(d) <= 0.02 ? "≈ синтез важить більше за різноманітність" : d > 0 ? "self ВИЩЕ — декорелятори переоцінені?" : "змішана панель виграє — різноманітність працює"}`);
}
if (agents && fusion && get("solo-glm")) {
  const dF = agents.total - fusion.total, dG = agents.total - get("solo-glm").total;
  console.log(`  fusion-agents: ${dF.toFixed(2)} до fusion, ${dG >= 0 ? "+" : ""}${dG.toFixed(2)} до solo-glm (частка підсилення, що доходить крізь роутер)`);
}
console.log("");
