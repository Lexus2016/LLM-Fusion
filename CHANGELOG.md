# Changelog

All notable changes to this project will be documented in this file.

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
