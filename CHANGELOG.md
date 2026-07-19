# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.31] - 2026-07-19

### Security

- **Fail-closed client auth.** `server.auth_token_env` naming an UNSET env var (e.g. a typo) previously disabled auth silently while the operator believed it was on; it now hard-errors every request (500 "configured but empty") with a loud startup log, matching the existing empty-string handling.
- **Refuse to start unauthenticated on a non-loopback bind.** The Docker image binds `0.0.0.0`, so a plain `docker run` published an open proxy (spending the owner's key, plus the admin API). Startup now throws unless a client token resolves, the bind is loopback (any `127.*`), or `FUSION_ALLOW_OPEN=1` is set explicitly. The image enables `auth_token_env: FUSION_PROXY_TOKEN` in the baked config (one `sed` on the stock file — single source of truth), and the Dockerfile documents the required `-e FUSION_PROXY_TOKEN=...`.
- **CSRF + DNS-rebinding guard on `/admin/*`.** When client auth is off (the localhost default and any `FUSION_ALLOW_OPEN` deployment) the admin API is now **loopback-only**: the `Host` header must be a loopback name (strict `127.0.0.0/8` octets, `localhost`, `::1`) and a missing/malformed `Host` fails closed — an `Origin`==`Host` equality check alone does NOT stop DNS rebinding (both headers carry the attacker's own name after the rebind). A present `Origin` must still match `Host`, and a mutating request with a body must be `application/json` (`transfer-encoding` and a nonzero/malformed `content-length` both count as a body, closing the chunked / bogus-length bypass). When a token IS configured, a non-loopback Host is allowed (front-with-your-own-auth). Previously a malicious page could rewrite a provider `base_url` and exfiltrate the upstream key.
- **`base_url` constrained at the schema seam.** A `base_url` carries the upstream API key as `Authorization: Bearer`, and the no-YAML panel editor can rewrite it — so it is now HTTPS-only (with an explicit `http://` exception for a loopback host, i.e. local Ollama) and may not embed userinfo (`user:pass@host`). Defense-in-depth behind the admin-API guard.
- **Prompt-injection defense-in-depth for the fusion synth.** The synth is the only stage holding the client's real `tools`, and untrusted material reaches it — live web-search results, the panel's own answers, and the judge analysis. All three are now wrapped in a per-request UUID-nonce fence (`<<UNTRUSTED_DATA …>>`, the nonce defeats delimiter spoofing) plus a system notice that the fenced content is reference **data, not instructions**. The judge schema also drops unknown keys (`.strip()`) so a prompt-injected judge cannot smuggle arbitrary keys into the synth context. Raises the bar against indirect prompt injection; not a guarantee against a fully compliant model — an empirical injection eval is a recommended follow-up.
- **Upstream key no longer forwarded on redirects.** `redirect: "error"` on every upstream fetch — undici re-sends explicit `Authorization` headers to redirect targets (the same guard the Tavily client already had).
- **`GET /admin/config` masks `extra_headers` values** (keys shown, values `•••`) — some providers authenticate via custom headers. The PUT write path restores the real on-disk values for untouched/masked keys so editing a provider never persists the placeholder.
- **Inbound body cap (50 MB)** on `/v1/*` and the admin API — unbounded bodies were a memory-exhaustion DoS on any non-loopback deployment.

### Fixed

- **Mid-stream upstream failures now reach the client as failures.** The usage decorator's pump closed the stream *gracefully* on an upstream error, appending a synthetic usage chunk + `[DONE]` — truncation looked like a clean end-of-turn and the failover strategy's deliberate committed-stream error was nullified. The pump now aborts the transform (the client's fetch body rejects) and distinguishes a client disconnect (debug log) from an upstream failure (warn).
- **Tool-turn guard no longer corrupts a client's tool-call arguments.** The streaming guard forwarded `delta.tool_calls` fragments live; on a length-cut mid-arguments truncation its recovery retry re-emitted the call at `index: 0`, so an index-keyed client (openai-python, Vercel AI SDK, OpenCode) concatenated the truncated old `arguments` with the new ones into invalid JSON — the exact `broken_tool_call` the guard exists to prevent. Tool-call fragments are now **buffered** (never forwarded live; content/reasoning still stream, so first-token latency is unchanged) and re-emitted as one assembled call at the terminal chunk. Because nothing partial reached the client, recovery now splices a clean replacement with no concatenation — the narrate-and-stop recovery feature is preserved. Reproduced and verified with a byte-level client-accumulator PoC (assembled arguments are now valid JSON).
- **Anthropic: a truncated tool call is no longer reported as runnable.** A tool call whose input JSON did not fully parse now maps to `stop_reason: "max_tokens"` for ANY finish reason (not just `"length"` — a truncated stream can end on `"stop"` or `null`), so Claude Code recovers instead of executing partial/empty tool input.
- **Bineval breaker hygiene.** A non-availability 4xx now records success (releasing the half-open probe — it previously jammed the model process-wide until restart), and a client disconnect abandons the probe instead of recording a health failure.
- **`connector_down_recheck_s: 0` really means "manual reset only".** A hard down is parked at an unreachable cooldown instead of being re-probed on every request; the panel shows "parked — manual reset only". Also: a stale failure from an old epoch can no longer free a probe slot a newer probe is holding.
- **Capability discovery follows live provider rebuilds.** `CapabilityService` (and `createApp`'s fallbacks) held the pre-reload pool — discovery kept querying decommissioned accounts with old keys after a `providers:` edit. A delegating client resolves the current pool per call. The reload signature now also covers `api_key_env` / `extra_headers` / `request_timeout_s`, so credential or header edits apply live instead of being silently accepted as no-ops.
- **Anthropic translation passes `stop_sequences` → `stop` and `top_k` through** (both were silently dropped — a standard agent-loop mechanism).
- **Smart escalation degrades to simple on panel failure** (it called fusion directly, surfacing a 502 on exactly the error-recovery step where upstream is most likely sick).
- **`UsageAccumulator.finalize()` is idempotent under overlapping calls** (a stream's tokens could be counted twice).
- **`listModels` consumes error bodies before throwing** (unread bodies made keep-alive sockets ineligible for reuse).
- **`fusion-opencode` no longer wipes a malformed `opencode.json`** — it aborts with a clear error instead of rewriting only the fusion provider. It also backs up the existing config before writing, preserves a stored token when `FUSION_PROXY_TOKEN` is unset (previously silently downgraded to `local-no-auth`), and chmods the file `0600`.
- **Startup warning for panel/small-fast rate-limit contention.** A pure, tested detector (`findPanelContentionOverlaps`) warns at startup when a `single`/`failover` model's upstream target is also a `fusion` panel member in the same provider group — e.g. `ANTHROPIC_SMALL_FAST_MODEL` pointed at a model that a live fusion panel uses, whose burst traffic can 429-starve the panel. Warn-only; no config is changed.
- **Docs sync:** CHANGELOG backfilled (`0.1.13`/`0.1.14`/`0.1.29`/`0.1.30`), README (en/ru/ua) preset tables + env-var tables + hot-reload claims, `fusion.example.yaml` panel lineup, and AGENTS.md's hot-reload / no-`as` claims now match the shipped code. The README's fusion headline was also reworded to an honest claim (decorrelated panel + judge adjudication + synth reconciliation + failover robustness) rather than an unqualified "stronger than any single model" — the committed benchmark does not support the superlative.

## [0.1.30] - 2026-07-15

### Fixed

- **Panel: usable toggles and roomier forms & modals.** Toggle switches stretched to full column width: `.sw` is a direct child of `.fld.toggle`, so the label-wrapper rule (`.fld.toggle > div{flex:1}`, specificity 0,2,1) overrode `.sw{flex:none}` (0,1,0). Restructured — the label gets its own class, the whole toggle is a clickable row-card, and the switch is a fixed 48×28. Form validation errors rendered muted-grey instead of red (`.modal p` 0,1,1 beat `.ferr` 0,1,0) — the rule is now scoped under `.modal`. Scaled up for readability: base 14→15px, form fields 13→14px with a focus ring, a custom single chevron on selects (kills the native double-arrow), and a wider form modal (560→720px) with more padding and larger title/action buttons. Verified in-browser: switch 48px (was full-width), modal 720px, row/switch click both toggle correctly, 0 console errors.

## [0.1.29] - 2026-07-15

### Added

- **Provider groups with automatic within-provider failover.** Upstreams are organised into **providers** (Ollama Cloud, OpenRouter, …), each with an ordered list of **accounts**; the proxy fails over between accounts *of the same provider* (same models), so a fusion built on a provider stays consistent when one account degrades. Each virtual model is bound to one provider via `provider:` (optional when there is a single group), so a model never silently jumps to a provider with a different model catalog. One `ProviderRouter` owns a `ConnectorRegistry` + `PooledUpstreamClient` per group; `dispatch` sets `ctx.client = poolFor(model.provider)` before the strategy runs — strategies are unchanged. Legacy single `upstream.base_url`+`api_key_env` synthesises one `default` group (backward compatible).
- **Health-based failover + classification.** A `429`/`5xx`/network/timeout **cools** an account for `connector_cooldown_s` (auto-probed after, single-flight); a `401`/`402`/quota marks it **down** for `connector_down_recheck_s` (the "billing ended" case). Concurrency-correct under interleaving: per-account **epoch** guard (a late failure can't overwrite a newer success), single-flight probe, monotonic cooldown. When several accounts fail, the pool surfaces the **most-recoverable** error by severity (`429`>`5xx`>network>`401`/`402`>`404`) so a recoverable request is never turned into a hard auth failure; `404`/model-not-found advances health-neutral; a `400`/`403`(passthrough) returns immediately; `Retry-After` sets the cooldown. Design + peer-review notes in `docs/multi-connector-failover-design.md`.
- **Generic `openai-compat` provider type.** Extracted the OpenAI `/v1/chat/completions` path into `OpenAiCompatClient` (`OllamaClient` now extends it, re-adding native `/api/show` + `/api/chat`). One `type: openai-compat` provider covers OpenRouter, DeepInfra, Together, Novita, Nebius, Groq, Cerebras, DeepSeek, Mistral, Baseten, … by config alone — `base_url` + `api_key_env` + per-account `model_map` + optional `extra_headers`. Ranked comparison in `docs/providers-research.md`. FAL-AI researched and **not** added (poor fit for chat).
- **Local panel.** `GET /panel` — a self-contained dark/light dashboard (no external requests, flicker-free in-place reconcile) showing each provider group and its accounts with reason + last error + cooldown countdown + per-account counters, plus manual controls: disable, enable, reset, make-active (pin), unpin. Backed by `GET /admin/providers` (grouped JSON) and `POST /admin/connectors/:id/{disable,enable,reset,pin,unpin}`, auth-gated by the existing client token when configured (the HTML shell carries no secrets).
- **Config:** new optional `providers:` map (backward compatible), `provider:` binding on each model, `upstream.connector_cooldown_s` (default 60) and `upstream.connector_down_recheck_s` (default 900). Validation rejects duplicate account ids (across all providers), a missing provider source, a model bound to an unknown group, and (with multiple groups) a model that omits `provider`.
- **No-YAML config editor.** The panel is now a full config manager, not just a monitor — three tabs: Monitor (live provider/account health), Providers (add/edit/delete provider groups and their accounts: id, `api_key_env`, `base_url`, `model_map`, `extra_headers`, `treat_403_as`, quota markers), and Models (create/edit/delete virtual models with per-strategy forms — fusion, single, failover, smart — every field carrying a plain-language hint; no YAML/XML is ever shown). Backend (`src/panel/config_editor.ts`): `GET /admin/config` (structured, no secret values — env-var names + presence only) and `PUT`/`DELETE /admin/config/{models,providers}/:id`. Each write edits the on-disk config via the `yaml` Document API (comments preserved), validates the WHOLE config with the boot zod schema, writes a timestamped backup, then replaces the file atomically (temp + rename) — nothing invalid is ever written; a bad edit returns a friendly error and leaves the file untouched. Models hot-reload live; `providers:` changes rebuild the router in place (`ProviderRouter.reload`) — no restart. Adding a second provider auto-binds previously-unbound models to the current group (else they'd become ambiguous), and adding a provider collects its first account inline (the schema requires ≥1).
- **Confirmation modal for mutating actions.** Every panel account action (disable/enable/reset/pin/unpin) and every destructive config-editor action opens a custom confirmation modal (not native `confirm()`) before firing; Cancel is auto-focused so a reflexive Enter cancels rather than confirming, and the account id / model name is HTML-escaped.
- **Live provider model picker.** The config editor fetches each provider group's real model list (`OpenAiCompatClient.listModels()` via `GET /v1/models`; Ollama falls back to `/api/tags`) and offers it as datalist suggestions in the model editor, so upstream model ids are picked, not free-typed. Backed by `GET /admin/config/providers/:id/models` (60s cache, 10s timeout, degrades to `200` + an empty list so the form never blocks; empty results are not cached, so a transient outage retries). Upstream-model fields (target/chain/panel/judge/synth/adversarial/router) suggest the provider catalog; smart routes suggest virtual model names.

### Changed

- **Smart-model provider-group validation.** A smart model's string-referenced `simple`/`fusion` routes must be in the SAME provider group as the smart model (dispatch runs them on the smart model's pool) — else a route's members would hit the wrong provider's accounts.

## [0.1.28] - 2026-07-08

### Fixed

- **Fusion synth no longer leaks its own scratchpad reasoning into the answer.** `buildSynthContext` told the synth to "write the single best final answer" but never forbade it from narrating its own synthesis process ("Expert 1 says X, Expert 2 says Y, I will provide...") into the visible response. On harder tasks that scratchpad consumed the entire token budget, leaving the real answer truncated or missing outright. Reproduced on 6/15 `bench/fusion-bench.mjs` tasks across two independent `fusion-coder` panel configurations (the retired mistral panel and the new 4-model panel below) — dragging `fusion`'s bench average to 23.60/30, below two of its own panel members. On the 9/15 tasks where the leak did not fire, `fusion` averaged 29.56/30, beating every panel member including solo glm-5.2 (28.11/30) — confirming the panel/judge/synth mechanism itself is sound; only the narration habit needed fixing. Added `SYNTH_DIRECT_ANSWER_DIRECTIVE`: forbids referencing "the experts/panel/judge" or showing draft reasoning, requires the final answer only, appended last in the synth context for maximum recency weight. Verified live against the three clearest prior failures (fs.watch platform caveats, LRU-TTL cache, Postgres isolation levels) — all three now return complete, direct answers with zero "Expert"/"judge" references.

### Changed

- **`fusion-coder` panel: 4 models, `glm-5.2` drops to judge+synth only.** New panel: `kimi-k2.7-code`, `deepseek-v4-flash`, `gemini-3-flash-preview`, `qwen3-coder-next` (replacing `glm-5.2` + `kimi-k2.7-code` + `mistral-large-3:675b`). Benchmarked on all 15 tasks: the fused answer beat every panel member on the 9/15 tasks unaffected by the synth-leakage bug above. `kimi-k2.7-code` is this panel's weakest voice (17.00/30 bench avg) — prone to long rambling generations that consume its own token budget; no per-panel-member `request_overrides` mechanism exists yet to throttle it (only `single` and the smart `simple` slot support `request_overrides` — see `src/config.ts`). `gemini-3-flash-preview` previously 400'd on foreign tool-call history in agent loops (`thought_signature` error, see v0.1.27 notes); a live spot-check with synthetic multi-turn tool-call history (with and without an active `tools` schema) did not reproduce the 400, but this is not exhaustive — worth watching in real agent-loop sessions via the existing `tool-turn terminal state` log line.

### Added

- **`bench/fusion-bench.mjs` hardening.** `--resume` flag to continue a killed/crashed run without re-paying for completed tasks. Scorer rewritten from one joint `{"scores":[...]}` object to JSON Lines (one object per condition per line) with a three-layer fallback (per-line, bare/prose-wrapped array, legacy object shape) — a bad line or shape mismatch no longer loses the whole task's score; unrecoverable labels are omitted, never defaulted to a fake 0 (which would silently corrupt the aggregate). Per-answer 8000-char cap before the scorer prompt, since a single unusually verbose answer (~19k chars) could overload the joint scorer call. Running stats and a stability/error counter now print after every task, not just at the end.
- Benchmark result files committed for provenance: `bench/results-v0127.json` (6/15 tasks, retired 3-model panel) and `bench/results-v0127-4panel.json` (full 15/15 run, new 4-model panel, 0/120 call errors).

## [0.1.27] - 2026-07-08

### Changed

- **`fusion-coder` panel: `gpt-oss:120b` → `mistral-large-3:675b`.** gpt-oss (131K ctx) was the context bottleneck — it capped the advertised `fusion-agents`/`fusion-coder` window at 131K while every other member has ≥262K; the advertised window is now **262144** (the kimi bound). Selection against the live Ollama Cloud roster with four filters: ctx ≥ 262K; Western lineage so the panel keeps its 2-Chinese-labs + 1-Western decorrelation (Mistral); accepts foreign tool-call history (panel-shaped probe: 200 in 7 s with the correct minimal-fix answer — gemini-3-flash still 400s on `thought_signature`); deliberation calibre (675B MoE). Rejected: nemotron-3-super (120 s timeout on the same probe), gemma4:31b (calibre); runner-up: devstral-2:123b. `fusion-researcher` intentionally keeps gpt-oss. A cross-provider review flagged the honest open item: the quality delta vs gpt-oss is unmeasured until the fusion-lift bench is re-run — bounded meanwhile (one voice of three, weighed by the judge; the synth writes the final code). Preset tables (README en/ua/ru) and the example config were synced, including the example's stale pre-v0.1.23 kimi synth.

### Fixed

- **Tool-turn guard emits canonical single-`[DONE]` framing.** Found by a cross-provider post-release review of v0.1.26: when the upstream ended its stream with `[DONE]` but no `finish_reason` chunk, the guard forwarded that `[DONE]` and then appended its own after the terminal-less recovery — recovery chunks and a second `[DONE]` after the client already saw one. Production clients were shielded by the downstream usage-injection transform (it drops upstream `[DONE]`s and appends exactly one), so no corruption was observable — but the guard's own output must be canonical regardless of what sits behind it. The upstream `[DONE]` is now swallowed; every finish branch appends its own. +1 framing test.

## [0.1.26] - 2026-07-08

### Added

- **Single-route tool-turn completeness guard (`src/strategies/tool_turn_guard.ts`).** Field debugging of "fusion-agents stalls the OpenCode agent loop" found three live-validated failure modes on the smart `simple` passthrough, none of which the single strategy recovered from: (1) *narrate-and-stop* — the model describes the next action in prose and ends the turn with `finish_reason:stop` and no `tool_calls`; (2) *length-cut tool calls* — large-file writes truncate the tool-call arguments mid-JSON (`finish_reason:length`), leaving an unrunnable call the client drops; (3) *upstream stream termination* — Ollama Cloud kills generation streams at ~5 minutes (`terminated`). The guard detects all three on tool-carrying requests and runs ONE streamed recovery retry forwarded to the client live (nudged to emit the tool call and to write large payloads in chunks), failing open to the original response. The stream wrapper is reader-driven rather than a `pipeThrough` TransformStream because `flush()` never runs when the source errors — a transform-based guard is structurally blind to failure (3).
- **Terminal-state instrumentation.** Every tool-carrying single-route stream now logs one `single: tool-turn terminal state` line (finish_reason, tool-call count, guard verdict, content/reasoning lengths, tail), making real-session stalls diagnosable from the log alone.
- **Per-model `request_overrides` (single strategy + smart inline `simple` slot).** Extra request-body fields merged into every upstream call for that model; core keys (`model`, `messages`, `stream`, `tools`, `tool_choice`) are protected. A/B-measured use case: `reasoning_effort:"none"` stops glm-5.2 from deliberating for minutes on mechanical agent steps (reasoning 1692→0 chars, 6s→2s; `think:false` and `reasoning_effort:"low"` are ignored by Ollama Cloud; tool-calling verified intact).
- **Fusion synth: agentic tool-action directive.** When (and only when) the request carries tools, the synth context now states that the panel's prose answers are deliberation, not the final shape — if the best next step is an action, emit the tool call. Inert on the tool-less research/report path, so prose synthesis and its benchmark results are untouched.

### Changed

- **`fusion-agents` simple route now sends `reasoning_effort:"none"`** (via the new `request_overrides`) — mechanical steps don't need deliberation; the smart router escalates hard steps to the fusion panel, and the tool-turn guard covers residual failures. The stale simple-target comment (which still claimed kimi was the fusion-coder synth) now documents the two real constraints: foreign tool-call-history compatibility and a large context window (a real session overflowed kimi's 262K — do not swap kimi in for speed).

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

## [0.1.14] - 2026-06-30

### Fixed

- **Adversarial-audit remediation batch** (`ADVERSARIAL-REVIEW-v0.1.13.md`):
  - Fusion: the panel-member work promise is now created inside `resilience.limiter()`, so `max_concurrency` actually bounds panel-member fetches (previously the fetch started before the slot was acquired).
  - Breaker: new `CircuitBreaker.recordProbeAbandoned()`; all strategy catch blocks skip `recordFailure` on client abort and release a reserved half-open probe, fixing the half-open probe-jam and disconnect-induced breaker trips.
  - Security: an empty configured auth token now returns 500 instead of silently bypassing auth; `/v1/models` is now behind auth; the Tavily fetch uses `redirect: "error"` (the key would otherwise leak to a redirect target).
  - Compatibility: hop-by-hop headers (`content-length`/`content-encoding`) are deleted on mutated responses (server/anthropic); Anthropic errors now use the `{"type":"error","error"}` shape.
  - Usage: only the usage field is stripped from usage chunks — `finish_reason` is forwarded.
  - Timeout: `request_timeout_s` bounds connection/first-response only for streaming — once the model starts delivering, the hard timeout is cleared so a slow but progressing stream is no longer truncated mid-delivery.
- **Signal-based abort detection.** The `isAbortError(err)` check added above misclassified a stage timeout as a client disconnect (a stage timeout also aborts the fetch, producing an `AbortError`), so it would wrongly skip `recordFailure`; client disconnect is now detected via the abort signal only. Also: hop-by-hop header stripping is now applied only when the body is actually rewritten (SSE transform / successful JSON injection), not to pass-through/error bodies — it could break gzipped error bodies.
- **Failover: a client disconnect during the pre-content peek no longer trips the breaker.** `peekFirstChunk` surfaced an `AbortError` as a generic peek failure, which the streaming failover path counted as a member failure; a `ctx.signal?.aborted` guard now releases the half-open probe and throws without recording a failure.
- **Non-streaming synth completeness guard.** A "thinking" synth (kimi-k2.7-code) can return `finish_reason: "stop"` while still mid-plan — empty content or an answer trailing off in planning narration, with no tool calls. `detectIncompleteSynth` now fires on exactly that shape (a tool call counts as a complete final action) and runs ONE stricter non-stream retry with a completion nudge; if the retry is still incomplete the original is kept (no loop). Restricted to the real failure mode — an empty raw `content` with the whole answer in `reasoning`: a non-empty content answer is authoritative and never second-guessed, so a legitimate answer whose tail happens to match a planning marker is no longer replaced by a nudged retry.
- **Smart: the router cache is checked before the circuit breaker (probe leak).** `canAttempt()` reserves the half-open probe slot as a side effect but was called BEFORE the router-decision cache lookup; on a cache hit while the breaker was half-open the reserved probe was never released, wedging the breaker half-open forever — every later request saw `probeInFlight=true` and all routing silently collapsed to the default route until restart. The cache + in-flight coalescing now run ahead of `canAttempt` (a cache hit issues no upstream call and must not touch the breaker).
- **Anthropic: the client abort signal is propagated upstream.** The `/v1/messages` handler dispatched without a signal while the OpenAI handler passed `c.req.raw.signal`; `/v1/messages` is the Claude Code endpoint where Esc cancellation is frequent, so a client disconnect never aborted in-flight upstream work and an entire fusion panel/judge/synth fan-out ran to completion for a gone client.
- **Anthropic: `tool_use` stop_reason is independent of the upstream finish_reason.** An upstream that emits tool_calls deltas but `finish_reason: "stop"` (or null on a truncated stream) produced `stop_reason: "end_turn"` despite tool_use blocks already streamed — Claude Code keys its agent loop on `stop_reason: "tool_use"` and never ran the tool. Tool presence now dominates stop/length on both the streaming and non-streaming paths.
- **Fusion: panel compression never orphans a tool message.** `compressPanelMessages` set the recent-window start without checking the role there; when that index landed on a `tool` result, its parent assistant(tool_calls) message was dropped and an omission marker inserted before the tool — orphaning it. Strict upstreams (Gemini) 400 on an orphaned tool message, thinning the panel below `min_panel_success` → 502, in the exact long agent loops compression exists to handle. The window start now walks back past leading tool results so it opens on the assistant that owns them.
- **Smart: the image hallucination guard ignores output-artifact nouns.** `claimsImage` matched bare nouns (`image|photo|picture|figure|diagram|chart|scan`) anywhere in the router reason, so a plain-text request whose reason mentioned the artifact to PRODUCE ("design an architecture diagram", "generate a chart") tripped the guard and got silently downgraded from fusion to simple — exactly the complex requests that should escalate. The affirmative match is narrowed to real visual INPUT: strong standalone signals (screenshot/multimodal), or a verb of receipt/presence adjacent to a visual noun.
- **Reasoning: complete inline `<think>…</think>` blocks are stripped, not just the tags.** `stripThinkingTags` removed only the markers, so a model that inlines its reasoning in `content` (DeepSeek-R/QwQ: `<think>…</think>answer`) leaked the entire chain of thought into the user-facing answer. Complete blocks are stripped first, then orphan tags (preserved for the separate-reasoning-field case).

## [0.1.13] - 2026-06-28

### Fixed

- **Concurrency and memory-leak remediation across auth, streams, timeouts, and config hot-reload.** Includes: constant-time client-token comparison now SHA-256-hashes both sides before `timingSafeEqual` (the previous length guard returned early on length mismatch, leaking the token length); `UsageAccumulator.finalize()` re-folds pending streams safely instead of memoizing a snapshot that a late `record()` could skew; smart-route references for `simple` now accept `single` OR `failover` targets (previously only `single`); plus concurrency/leak fixes in the stream tap, the config hot-reload watcher, and the server request path. Regression tests added in `test/config`, `test/server`, and `test/smart`.

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
