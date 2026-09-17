# PROPOSAL: Sharper Fusion Quality on a Flat Token Budget

To: llm-fusion dev team
From: audit session, 2026-08-21
Context: v0.1.36 · companion docs: `AUDIT-v0.1.36.md` (defects), `ADVERSARIAL-REVIEW-v0.1.30*.md`

## Goal

Raise task-solution quality and ensemble intelligence of the multi-model pipeline
(`panel → judge → synth`, `smart`, `bineval`) **without a proportional token-cost increase**.
Every item below pays tokens only where they demonstrably add intelligence, or reduces the
baseline spend directly.

## P1 — Cascading panel instead of fixed size (biggest lever)

Today `runFusion` always fans out to every configured panel member (`src/strategies/fusion.ts`,
`runPanel`). Replace with progressive deliberation:

1. Run the first wave = the 2 strongest members.
2. Judge evaluates as today but reports agreement (`confidence`, `fragile_claims` — both fields
   already exist in `JudgeAnalysisSchema`; no schema change needed).
3. High disagreement / low confidence / fragile claims present → run the remaining members and
   re-judge once. Otherwise go straight to synth.

Cost model: easy tasks settle at ~40% of full-panel spend (1 judge call on half the answers);
hard tasks pay full price plus at most one extra judge call — bounded, never recursive.
Quality is preserved *by construction*: disagreement is precisely the signal that warrants more
experts.

Composition note: this pairs with `escalate_on_tool_error` (`smart.ts`) — today a tool-error
trigger jumps straight to a FULL panel even when the failure is a typo'd command. Under the
cascade, escalation still routes to fusion, but only genuinely contested failures pay for the
full panel.

## P2 — Judge on compressed expert answers (replaces "skip the judge")

Do NOT skip the judge on apparent consensus: naive agreement metrics (n-gram overlap) are gamed
by shared boilerplate across experts ("Here is the complete implementation…"). Instead:

- Feed `renderPanelForJudge` output through head+tail caps (reuse the `capPanelMessageContent`
  machinery already built for the panel view, `fusion.ts` ~PANEL_MSG_HEAD/TAIL). The judge needs
  adjudication material, not verbatim artifacts — artifacts still reach the synth raw via
  `buildSynthContext`.
- This shrinks the single most expensive call (judge ≈ one full-context request) by ~half with
  no consensus-detection risk.
- Reserve full judge *skipping* only for the cascade's explicit `confidence:"high"` path (P1),
  which is model-assessed, not heuristic.

## P3 — Bineval as a repair loop, not telemetry

`attachBinevalHeaders` computes the score and stops. Wire the existing threshold:

```
score < bineval.threshold  →  ONE retry of synth with targeted feedback
                              (the failed dimensions from result.dimensions),
                              then ship whichever attempt scored higher.
```

Extra cost lands only on failing responses (a minority); the quality floor rises sharply.
Bound: one retry, no recursion — same fail-open discipline as `retrySynthForCompletion`.

## P4 — Panel lineup: 2 strong + 2 cheap beats 4 strong

At equal token budget, self-consistency among strong lineages and lineage decorrelation beat
adding more frontier models. Recommended default lineup shape: two strong members of DIFFERENT
training lineages + two cheap diverse members + the adversarial slot. The project's own
adversarial review already established that shared-priors agreement is a failure mode; cheap
diversity buys decorrelation cheaper than strength does.

## P5 — Prefix-cache-friendly prompt layout (direct input-token discount)

OpenRouter/DeepInfra-class providers discount cached prefixes. Two current habits break caching:

- `buildPanelBody` inserts web context via `insertBeforeLastUser` — mid-conversation mutation
  invalidates everything before it. Move volatile payloads (web context, untrusted-data fences,
  adversarial/tools notices) strictly to the END of the message list; keep system + stable
  history as a byte-stable prefix.
- Keep `JUDGE_SYSTEM_PROMPT` / router prompts constant (they already are) and avoid injecting
  anything per-request (dates, nonces) ahead of them.

Expected effect: −20…50% billed input tokens on long-loop conversations, zero behavior change.
Verify with provider cache-hit counters if available.

## P6 — Quality-per-dollar regression harness

Everything needed is already emitted: `cost_usd` + token counts in usage logs, bineval scores in
headers, attribution lines, and `bench/fusion-bench.mjs`. Add one number to the bench report:
`quality_per_dollar = mean(bineval score) / mean(cost_usd)` and gate config/lineup changes on it.
What gets measured stops being guesswork; run an A/B of every item above against this metric.

## P7 — Cheap-route few-shot lift (optional, later)

Accumulate (request → high-bineval-scored answer) exemplars offline; inject 2–3 similar ones
into the `simple` route's prompt (capped chars). Raises the cheap branch without touching fusion
spend. Requires a small local store + similarity lookup; keep it out of scope until P1–P6 land.

## Do NOT do

- No "just add panel members" headroom — marginal members cost full price and dilute the judge.
- No synth-before-judge speculative overlap — duplicates spend on disagreement.
- No aggressive response caching — staleness corrupts agent loops (the smart-router decision
  cache stays decision-level, not answer-level, deliberately).

## Rollout order

| # | Item | Effort | Expected impact |
|---|---|---|---|
| 1 | P5 prefix layout | S | direct −20…50% input cost |
| 2 | P2 compressed judge | S | −~50% on the largest single call |
| 3 | P3 bineval repair | M | raises quality floor, pays only on failures |
| 4 | P1 cascade | M | big win: easy-task cost ↓, hard-task quality ↑ |
| 5 | P4 lineup rebalance | S (config + bench runs) | quality at equal budget |
| 6 | P6 q/$ harness gate | S | protects all of the above |
| 7 | P7 few-shot simple route | L | deferred |

Each step ships behind its existing config knobs where possible (panel waves can start as
`min_panel_success`-style integers; bineval retry behind a `bineval.repair: true` flag).
