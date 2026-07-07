# Changelog

All notable changes to this project will be documented in this file.

## [0.1.25] - 2026-07-07

### Changed

- **`fusion-coder` BinEval evaluator: `glm-5.2` → `deepseek-v4-pro`.** After v0.1.23 glm-5.2 became the coder synth, so a glm evaluator would grade its own answers. deepseek-v4-pro sits outside the coder pipeline entirely; BinEval runs rarely (non-streaming responses only), so sharing the concurrency-capped background-traffic model is fine. `fusion-researcher` keeps glm (its synth is kimi — independent).
- README (en/ru/ua): the recovery-fallback description now reflects the v0.1.24 behavior (judge, or a panel member when the judge IS the synth). Pilot/smoke benchmark result files committed for provenance.

## [0.1.24] - 2026-07-07

### Fixed

- **Synth-recovery fallback works again when `judge === synth`.** v0.1.23 made glm-5.2 both judge and synth for `fusion-coder`, and the recovery fallback (`judge !== synth ? judge : null`) silently lost its cross-model insurance — an empty synth answer had no second model to recover on. The fallback now picks the first PANEL member that differs from the synth in that case. Stale "kimi is the fusion-coder synth" wording in `docs/claude-code.md` and the `fusion-claude` launcher comment updated to the post-v0.1.23 reality.

## [0.1.23] - 2026-07-07

### Changed

- **`fusion-coder` synth: `kimi-k2.7-code` → `glm-5.2`** — the first configuration change driven by the project's own fusion-lift benchmark (`bench/`, analysis in `docs/llm-fusion-audit.md`). As synth, kimi emitted 17–19k-char draft-style output on factual/reasoning tasks and dragged the always-on fusion composite to 0.71 — below every solo panel member. With glm-5.2 as synth the same pipeline scored **0.99 vs 0.89 for the best solo member** (n=14, blind 3-axis scoring by a model outside all conditions; per-answer methodology control kept solo-glm at 0.89). The fused model now exceeds every individual member — the project's core promise, measured. kimi stays on the panel; `fusion-agents` inherits the change via its `fusion-coder` reference.

## [0.1.22] - 2026-07-07

### Added

- **`partial_coverage` judge dimension.** The judge now also names the aspects of the request that SOME expert answers cover and others miss (and which expert covered each), and the synth is instructed to take each such aspect from the covering expert instead of averaging it away. Closes the one dimension missing versus OpenRouter Fusion's five-dimension judge (accepted item #2 of the comparative audit, verification 2).

## [0.1.21] - 2026-07-07

### Fixed

- **`<think>` tags no longer leak across SSE fragment boundaries.** The think-tag stripper ran per stream fragment, so an SSE boundary splitting the tag itself (`"<th"` + `"ink>"`) leaked the literal tag into the visible output, and a block whose body arrived in later fragments leaked the private reasoning as answer text. A stateful stream filter (`createThinkTagStreamFilter`) now carries a possible partial tag across fragment boundaries, suppresses everything inside an open block until its close tag, strips orphan close tags, and surfaces a false-partial tail (e.g. a literal `"<tho…"`) as text instead of swallowing it. Wired into both streaming paths — the reasoning-promotion transform (OpenAI clients) and the Anthropic stream transform (Claude Code) — with separate filter instances per source field, so an unterminated block in `reasoning` can never suppress real `content`; a reasoning-phase false partial is merged before the first content fragment, preserving order. The synthetic tail chunk before `[DONE]` is framed as its own blank-line-closed SSE event and carries the stream's `id`/`model` metadata (pre-tag adversarial review findings).

## [0.1.20] - 2026-07-07

### Added

- **Russian and Ukrainian READMEs.** Full translations (`README.ru.md`, `README.ua.md`) pinned to v0.1.19, with a language switcher in all three files.

### Changed

- **README refreshed for v0.1.16–19.** Fixed the stale model lineup (`gemini-3-flash-preview` → `gpt-oss:120b` in the fusion-coder panel; `glm-5.2` on the fusion-agents simple route), documented the agent-loop reliability features (synth completeness guard with judge fallback + SSE keepalive, honest `max_tokens` stop_reason, per-model concurrency budgets, separated background-traffic class), updated concurrency/timeout numbers to the shipped presets, added `FUSION_SYNTH_RECOVERY_PING_MS` to the environment table, removed a dangling Docker cross-reference.

## [0.1.19] - 2026-07-07

### Added

- **Per-model concurrency budgets.** All three consilium advisors independently flagged head-of-line blocking: one global `p-limit` meant a burst of background small-model calls occupied the whole FIFO queue ahead of interactive fusion turns. Every real upstream model now gets its own gate in front of the global limiter (`resilience.limiterFor(model)`), with strict model-gate → global-slot acquisition order (uniform ordering — no lock cycle; confirmed deadlock-free in a pre-tag adversarial review). A saturated model holds at most its budget of global-queue positions. Config: `upstream.per_model_concurrency` (map, keyed by real upstream model name) and `upstream.per_model_concurrency_default`; both unset = behavior identical to the previous single global limiter. The shipped `fusion.yaml` caps `deepseek-v4-pro` (the Claude Code background-call carrier) at 2. All strategy fallbacks wire the same budgets via `resilienceForUpstream`, and `/v1/models` capability discovery is keyed too.

## [0.1.18] - 2026-07-07

### Fixed

- **Inline `<think>` stall detection.** `detectIncompleteSynth` judged the RAW `content`, so an R1/QwQ-style synth answer consisting of one inline `<think>…</think>` block (narration, no artifact) sailed through as complete and the completeness guard never fired. Both the answer and the raw-content checks now judge the think-stripped text. Documented trade-off: an answer that legitimately consists entirely of literal think markup costs one wasted recovery attempt (the original is kept when recovery fails) — never content loss.
- **Length-cut recovery retries are no longer adopted.** The strict recovery retry can itself hit the token cap mid tool call (`finish_reason: "length"`, truncated/non-string `arguments`); such a result was accepted as "recovered". It now yields to the fallback-model attempt instead.
- **SSE keepalive during recovery.** The recovery retry runs synchronously inside the stream flush — previously the client saw total silence (up to two full upstream call latencies) and could time out. SSE comment lines (`: keepalive`) now flow every 5s (override via `FUSION_SYNTH_RECOVERY_PING_MS`), and an unexpected throw inside recovery fails open to the original terminal chunk instead of breaking the stream.

Source: a three-provider consilium panel review (kimi / deepseek / Gemini) of the v0.1.16-17 fixes, plus a pre-tag adversarial re-review of this diff.

## [0.1.17] - 2026-07-07

### Fixed

- **Length-cut turns keep COMPLETE tool calls runnable.** Refines the 0.1.16 `stop_reason` change (flagged in cross-provider peer review): the token cap can land exactly after a tool call's JSON finished — discarding such a call as `max_tokens` wasted a whole (possibly multi-minute) turn. `finish_reason: "length"` now maps to `tool_use` when EVERY tool call's input JSON parses to completion (streamed `input_json_delta` fragments are reassembled per block and validated), and to `max_tokens` only when the input is actually truncated. Both the JSON and streaming paths.

## [0.1.16] - 2026-07-07

### Fixed

- **Synth judge-model fallback.** kimi-k2.7-code intermittently answers a tool-turn with reasoning-only / empty output (`finish_reason: "stop"`, no tool_calls, no content) even after the strict completion nudge — one model could stall a whole agent loop. After the same-synth retry, the completeness guard now makes ONE additional non-streamed attempt on the judge model (a different lineage, empirically the most reliable structured-output model in the panel). At most two recovery calls, covering both the streaming and non-streaming synth paths.
- **Honest `stop_reason` for length-truncated tool calls.** A Write/Edit tool call cut by the token limit arrived at the Anthropic endpoint with `stop_reason: "tool_use"` and a partial `input` JSON; Claude Code then executed the broken input, failed, and escalated. `finish_reason: "length"` now maps to `stop_reason: "max_tokens"` BEFORE tool-presence domination, on both the JSON and streaming paths, so clients recover instead of running truncated tool input.

### Changed

- `bin/fusion-claude`: default `ANTHROPIC_SMALL_FAST_MODEL` moved `fast-kimi` → `fast-deepseek`. Claude Code fires background bursts of 80-130 small-model calls/min; on 2026-07-06 those bursts 429-starved kimi — which is also the fusion synth that writes file contents — killing large-file generation mid-loop. deepseek-v4-pro is used by no panel/synth/judge/router/simple route, so its rate-limit bucket is free to burn.
- `fusion.yaml`: `web_search` disabled for `fusion-coder` (kept for `fusion-researcher`). Grounding added ~12s of pre-panel silence per deliberate agent turn (measured 12.7s of a 28.7s time-to-first-byte) while searching the web for things like "create page.html" — pure latency on agent tool-loops.

Validated end-to-end: a pure fusion-coder agent loop produced a 3364-line / 117 KB file in 297 s (the whole file in one streamed 119 KB Write tool call); a fusion-agents loop produced a 5000-line / 169 KB file in ~21 min with zero 429s and zero synth retries.

## [0.1.15] - 2026-07-06

### Fixed

- **Streaming synth completeness guard.** Tool-call requests over `stream: true` could hang indefinitely: the synth model would emit `finish_reason: "stop"` with no real content/tool_calls (stalled mid-plan) and, unlike the non-streaming path, nothing detected or retried it on the streaming path. `makeSynthStreamCompletenessGuard` now mirrors the existing non-stream completeness guard on the streaming path — non-terminal chunks pass through live (no added latency), only the terminal chunk is buffered and checked, and a single strict non-stream retry recovers the answer/tool call if the synth stalled.
- SSE framing: the terminal/replacement chunk emitted by the streaming completeness guard now correctly closes its own event with a blank line (`\n\n`) before `data: [DONE]` opens the next one; the prior single `\n` merged both into one SSE event and broke client-side `JSON.parse`.
- Recovered legacy `function_call` tool calls (`{name, arguments}`, no `function` wrapper) are now normalized into the OpenAI streaming `delta.tool_calls` shape (`{type: "function", function: {name, arguments}}`) instead of being spread at the root, and any other passthrough fields (e.g. `id`) are preserved rather than discarded.

## [0.1.12] - 2026-06-28

### Added

- **Web grounding for the fusion panel.** A fusion model can opt into `web_search: { enabled: true, max_results, timeout_s, max_context_chars }`; the proxy runs one Tavily search before the panel fans out and injects the cleaned results as prose context into every panel member (no member receives real tools, so the one-`tool_calls`-per-step invariant holds). Gated two ways — `TAVILY_API_KEY` in the environment *and* `web_search.enabled` on the model — so it is fully OFF by default with no latency/cost. The context is injected as a `user` turn (not `system`) and prefixed with the current date, so models with a stale training cutoff (kimi-k2.7-code) use the fresh facts instead of refusing. A failed search degrades silently to an ungrounded panel; large prompts skip web grounding (size gate). Respects `fusion_planning_turn_only`.
- **Calibrated judge confidence.** The judge analysis now emits `confidence` (`high`/`medium`/`low`) and `fragile_claims` (the disputed / singly-supported / thin claims). The synth hedges fragile and low-confidence claims rather than laundering shared training priors into false certainty. Consistency rule: any `hallucination_flags` or `fragile_claims` forces `confidence` to `medium`/`low` — never `high`.
- **Adversarial panel slot.** A fusion model can set `adversarial: <panel-member>`; that member is instructed to steelman the opposite case and hunt for flaws, hidden assumptions, edge cases, and race conditions instead of agreeing with the consensus — role-based decorrelation on top of the lineage-based one. The adversarial member is never early-cancelled (its red-team contribution is the whole point) and the panel does not resolve until it finishes, so its answer is never dropped. Validated to be an existing panel member.
- **BinEval observability header.** When bineval is configured but the evaluation does not run, the proxy sets `X-Fusion-Bineval-Skipped: <reason>` so a client can tell "score is high" apart from "evaluation never ran": `streaming`, `synth_error` (synth ≥400), `eval_failed` (evaluator errored/timed out/circuit open/non-2xx, or the eval call threw), `empty_output` (tool-only response), `non_json_body`, `synth_only` (the planning-turn-only mid-loop / bypass path skips the panel). The skip reason is now labelled precisely — JSON parsing of the synth body and the evaluator call are separated, so an evaluator exception is reported as `eval_failed` rather than mislabelled `non_json_body`. BinEval remains **report-only** — it does not trigger a re-deliberation loop.

### Changed

- `upstream.request_timeout_s` default raised 120s → 170s (still under the ~182s Ollama Cloud ceiling): a full fusion-coder/researcher path (panel → judge → synth → bineval) on a heavy prompt could exceed 120s at the synth stage and 504 before bineval attached its headers; 170 lets the synth + bineval stages finish.

### Fixed

- Smart-router array-text rendering corrected and a `simple` fallback added; the router now grounds its reason in the literal latest message and no longer invents multimodal content (a pasted clipboard text no longer trips a false "image present" guard).
- Panel context compression now properly caps array-based multimodal message content, so a long agent loop with image-bearing messages no longer overflows member context windows. The compression keeps system messages + the original task + the recent window, replaces the mechanical middle with omission markers.
- Config validation rejects a fusion panel that lists the same model twice (a duplicate wastes an upstream call and, for the adversarial slot, only the first copy is protected from early abort) and an `adversarial` member not present in the panel.
- Anthropic `/v1/messages` accepts unknown content blocks so long fusion agent loops don't 400.

## [0.1.11] - 2026-06-28

### Added

- Smart-router decision cache: an identical router request body now reuses the prior classification instead of re-paying a router round-trip every turn. Only successfully parsed decisions are cached; failures (timeout, non-OK status, unparseable JSON) fall through to the default route and are NOT cached, so a transient blip self-heals on the next identical request.
- In-flight coalescing: concurrent identical router requests share a single upstream call, so a burst of identical in-flight requests does not pay N router round-trips.
- LRU-bounded cache (max 256 entries) to keep memory predictable across long-running sessions.

## [0.1.10] - 2026-06-28

### Added

- Fusion panel gains conditional abort: slow panel members are cancelled once enough survivors have answered, cutting tail latency while preserving `min_panel_success`.
- Thinking/reasoning tags are stripped from panel answers before they reach the judge, keeping the analysis focused on substantive content.
- Smart-router prompt tuning improves routing accuracy between `simple` and `fusion` strategies.
- Anti-hallucination safeguards in the fusion pipeline:
  - Judge now emits `hallucination_flags`, cross-referencing panel experts and flagging claims supported by only one member as suspect.
  - Synth is instructed to omit or caveat flagged items and to acknowledge uncertainty when experts disagree irreconcilably.

## [0.1.9] - 2026-06-27

### Fixed

- Anthropic `/v1/messages` endpoint no longer rejects valid Claude Code requests with `messages.N.content: Invalid input`.
  - Accepts `content: null` on user/assistant/system messages and normalises it to an empty string for upstream translation.
  - Accepts `thinking` and `redacted_thinking` content blocks (e.g. from extended-thinking models) and ignores them when translating to OpenAI.
  - Accepts `tool_result` blocks with `content: null`.
  - Accepts `tool_result` blocks whose content is an array containing image blocks; images are converted to a URL/data-URI string because OpenAI `tool` messages only support string content.
  - Ensures assistant messages that end up with no content and no tool calls fall back to `content: ""`, keeping the upstream OpenAI request valid.

### Added

- Exported `AnthropicRequestSchema` for reuse in tests and diagnostics.
- Unit and integration tests covering null content, thinking blocks, and image tool results.

## [0.1.8] - 2026-06-27

### Previous release baseline

- Existing Fusion Proxy functionality up to and including the Anthropic compatibility layer, strategy dispatcher (single/fusion/smart/failover), usage accounting, auth, and capability discovery.
