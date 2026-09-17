# Adversarial Review — llm-fusion v0.1.30

Six independent audit domains (security, HTTP/API layer, strategy logic, connectors/upstream,
accounting/concurrency, tests/docs/UX), reviewed by parallel adversarial auditors.
Top findings were re-verified by hand against the code before inclusion. Nothing below is
fabricated; items we could not fully confirm are marked *(suspected)*.

Previous review: `ADVERSARIAL-REVIEW-v0.1.13.md`.

---

## CRITICAL / HIGH

### H1. Unauthenticated admin config rewrite → `OLLAMA_API_KEY` exfiltration via `base_url`
`src/panel/config_editor.ts:124-132` accepts an arbitrary `base_url` (schema only
`z.string().url()`, `src/config.ts:250`); `src/index.ts:72-93` hot-rebuilds the provider
router live; `src/connectors/resolve.ts:73,86-93` resolves the key from env and hands it to
a client pointed at that URL; `src/upstream/openai_compat.ts:74-78` sends it as
`Authorization: Bearer <key>`. Route auth is a no-op when no token is configured (the default).
**Scenario:** anyone who can reach the port rewrites `providers.ollama-cloud.base_url` to an
attacker host; the next request ships the real Ollama Cloud key there. Full upstream-key theft,
no authentication required. The env-var-name indirection (a genuinely good design) does not
help because the value is resolved server-side and sent wherever `base_url` points.
**Fix:** require auth for all `/admin/*` writes when bind ≠ loopback; allowlist/scheme-restrict
`base_url` (https-only, no userinfo).

### H2. Docker image ships a world-open, unauthenticated proxy by default
`Dockerfile:30` `ENV FUSION_BIND=0.0.0.0`; baked-in `fusion.yaml:57` has `auth_token_env`
commented out; `src/index.ts:100-104` only *warns* when auth is off; the Dockerfile header
recommends plain `-p 8080:8080`.
**Scenario:** `docker run -p 8080:8080` publishes an unauthenticated LLM proxy billed to the
owner's key — plus the unauthenticated admin API of H1.
**Fix:** fail-fast when `bind != 127.0.0.1` and no token resolves (explicit
`FUSION_ALLOW_OPEN=1` escape hatch), or make the image require `FUSION_PROXY_TOKEN`.

### H3. `auth_token_env` configured but env var missing ⇒ auth *silently* disabled *(verified)*
`src/index.ts:95-98` — `getAuthToken` returns `process.env[envName]`, i.e. `undefined` when
the var is absent. `src/auth.ts:31-36` treats `undefined` as "auth intentionally off", while
the very next branch (`src/auth.ts:37-45`) correctly hard-errors on the empty-string variant
of the same misconfiguration.
**Scenario:** operator sets `server.auth_token_env: FUSION_PROXY_TOKEN` but typos the env var
(`-e FUSION_PROXY_TOEKN=...`). Proxy runs fully open while the operator believes it is
authenticated. The code recognizes one spelling of this mistake as fatal and the other as fine.
**Fix:** when `auth_token_env` is configured but the variable is unset, hard-fail at startup.

### H4. No Origin/Host/content-type guards → CSRF & DNS-rebinding into the admin API
No CORS/origin middleware anywhere in `src/server.ts`; Hono's `c.req.json()` does not check
content-type — it reads the body and `JSON.parse`s unconditionally. The config editor accepts
writes this way (`src/panel/config_editor.ts:96-102`).
**Scenario (blind CSRF):** a malicious page visited by the operator fires a cross-origin
`mode:"no-cors"`, `content-type: text/plain` PUT to `/admin/config/providers/x` — no preflight,
the write lands, chaining into H1.
**Scenario (DNS rebinding):** attacker domain alternating between their IP and `127.0.0.1`
drives the whole admin API from the victim's browser (the classic local-proxy vuln class).
**Fix:** reject mismatched `Origin`/`Sec-Fetch-Site`, validate `Host` against bind:port,
require `content-type: application/json` on mutating routes.

### H5. `decorateUsage` masks mid-stream upstream failures as clean `[DONE]` endings *(verified)*
`src/server.ts:291-313` — the pump's catch calls `writer.close()` ("closing gracefully to
client"); `src/usage.ts:305-323` then appends an aggregate usage chunk and `data: [DONE]`.
This directly contradicts `src/strategies/failover.ts:413-416,438-440`, which deliberately
errors the stream on a committed-stream failure ("never a silent re-route", spec §10.5).
**Scenario:** failover commits to member A; A dies mid-generation. The client receives a
well-formed SSE stream that simply ends early — no `finish_reason`, no error event, no
exception. An agent client aggregates a partial delta (possibly half a tool-call's JSON
arguments) and treats the turn as complete. The one deliberate "client must see this break"
mechanism in the codebase is neutralized one layer up.
**Fix:** on upstream stream error use `writer.abort(err)` (or emit a visible error line) and
skip the synthetic usage/`[DONE]` tail when the upstream did not end cleanly.

### H6. Tool-turn guard's cut recovery splices a duplicate/corrupt completion into the client stream *(verified)*
`src/strategies/tool_turn_guard.ts:514-535` (`finishAfterCut`) + `:402-447` (`handleLine`).
`handleLine` forwards every non-terminal `data:` line immediately (line 446). On a mid-flight
upstream cut, `finishAfterCut` runs `runStreamingRecoveryWithKeepalive`, which re-asks the
model and forwards the **entire replacement completion** into the same client stream.
**Scenario:** partial answer + full regenerated answer = duplicated, spliced prose. Worse: if
`delta.tool_calls` fragments were forwarded before the cut, the retry's new tool call restarts
at `index: 0` and the client concatenates truncated old `arguments` with new ones → invalid
JSON — exactly the "broken_tool_call" failure the guard was built to eliminate, now
undetectable. Same duplication in `finishNormally` when `terminalLine === null` (`:461-470`):
a complete answer without a `finish_reason` chunk is delivered **twice**.
**Fix:** only run cut-recovery when nothing was forwarded yet
(`toolCallAcc.size === 0 && content === "" && reasoning === ""`); otherwise error or close
honestly and let the client retry. The code acknowledges this splice risk for
retry-mid-stream breaks (`:294-303`) but not for the primary upstream-cut path.

### H7. Prompt-injection path: untrusted web/panel content flows into the tool-executing synth
`src/strategies/fusion.ts:1860-1864` (`renderPanelForJudge`), `:1819-1858`
(`buildSynthContext`), `:944-950` (web context injection).
Trust chain: arbitrary web pages (Tavily results) → panel prompts → raw panel answers →
verbatim into (a) the judge's user message behind a weak `--- Expert N ---` delimiter and
(b) the synth's system message — and the synth is the **only stage holding real `tools`** in
an agent loop (`:1760-1762`). The judge schema is `passthrough` (`:88`), so injected keys
survive into synth context via `JSON.stringify(analysis)` (`:1843`). The
`SYNTH_TOOL_ACTION_DIRECTIVE` (`:1779`) actively pushes the synth toward emitting tool calls,
lowering the bar further.
**Scenario:** a poisoned web result instructs the judge to set `confidence:"high"` and drop
`hallucination_flags`, or directly instructs the synth to emit an attacker-chosen tool call.
No sanitization, no instruction/data separation beyond a prose delimiter that is itself part
of the injection surface.
**Fix:** UUID-fenced blocks with an explicit "content is data, not instructions" note; strip
unknown keys from judge analysis before serializing; document the residual risk of
`web_search.enabled` + agentic models.

### H8. Bineval leaks the half-open breaker probe on 4xx — model jammed until restart *(verified)*
`src/bineval.ts:213-222`: on a JSON 4xx that is not 429/5xx (e.g. a 400 from a model rejecting
`response_format: json_object`), the code records **neither** `recordFailure` nor
`recordSuccess` nor `recordProbeAbandoned`. Every other call site handles this case explicitly
with `recordSuccess` (`single.ts:90-92`, `fusion.ts:654`, `smart.ts:367`, `failover.ts:193`).
**Scenario:** breaker cools to half-open; bineval probe gets a 400 → `probeInFlight` stays
`true` forever (`concurrency.ts:141-143`). The breaker is shared per real model across all
strategies, so the model is dead process-wide until restart — the exact "jammed until restart"
state the code's comments repeatedly defend against.
**Fix:** `else resilience.breaker.recordSuccess(model)` (or `recordProbeAbandoned`) for the
non-availability 4xx branch, mirroring `single.ts`.

### H9. Bineval counts a client disconnect as a model health failure *(verified)*
`src/bineval.ts:201-209`: the catch block unconditionally calls `recordFailure(model)` — no
`ctx.signal?.aborted` check, unlike every other strategy (`single.ts:68-71`,
`fusion.ts:616-619,1041-1044,1143-1146`, `smart.ts:326-329`, `failover.ts:127-130`).
**Scenario:** client disconnects mid-bineval → `consecutiveFailures` increments against a
healthy model, can trip the breaker; in half-open it re-opens the breaker (`concurrency.ts:160-163`).
**Fix:** check `ctx.signal?.aborted` first and call `recordProbeAbandoned` instead.

### H10. `connector_down_recheck_s: 0` means "probe immediately", not "never" *(verified)*
`fusion.example.yaml:21-23` and `docs/multi-connector-failover-design.md:104,156` both
document `0 = never (manual reset only)`; `src/config.ts:203` allows `min(0)`. But
`src/connectors/registry.ts:246-247` computes `nextUntil = max(cooldownUntil, now + 0) = now`,
and `acquire` (`registry.ts:195`) admits a probe whenever `now >= cooldownUntil` — instantly true.
**Scenario:** operator parks a billing-dead account with `0`; instead every request
single-flight-probes the dead connector first — an extra 401 round-trip per request, forever.
The documented contract is inverted.
**Fix:** treat `downRecheckMs === 0` (with no `retryAfterMs`) as `cooldownUntil = Infinity`;
manual reset becomes the only revival.

### H11. CapabilityService keeps the pre-reload pool after a live providers rebuild *(verified)*
`src/index.ts:49` captures `const client = router.defaultPool` once and injects it into
`CapabilityService` (`:51-55`) and `createApp`. `ProviderRouter.reload()`
(`src/connectors/provider_router.ts:65-67`) replaces the whole groups map — new registries,
new pools, new clients with new keys — but `CapabilityService.deps.client` still points at the
old pool wrapping the old registry/clients.
**Scenario:** operator rotates an account or swaps `base_url`; chat traffic routes correctly,
but all `/api/show` capability discovery, the `/v1/models` aggregation (`server.ts:131-135`)
and the `/ready` fallback silently query the *old*, supposedly-decommissioned accounts with
the old keys. `capabilities.clear()` (`index.ts:76`) makes it worse: cache flushed, then
refilled through the stale pool.
**Fix:** inject a getter (`getClient: () => router.defaultPool`) mirroring the existing
`getOverrides` pattern.

### H12. Concurrency limiter does not bound streamed upstream connections *(verified)*
`src/upstream/openai_compat.ts:157-165` returns as soon as response **headers** arrive for
streams; the limiter callback in `single.ts:59-61`, `fusion.ts:1135-1137` (synth) and
`failover.ts:214` therefore releases the global + per-model slot at header time, while the
connection streams for minutes. `concurrency.ts:5-9` claims the limiter bounds "in-flight
upstream calls across every request and every fusion stage" — under the dominant workload
(long streams, agent loops), concurrent upstream connections are unbounded: N clients
streaming = N connections regardless of `max_concurrency`. The fusion panel does it correctly
(stream fully accumulated inside the limiter, `fusion.ts:568-610`), proving the pattern was known.
**Fix:** hold the slot until the stream drains (release on stream end/cancel — needs a manual
permit, or move consumption inside the callback as the panel does).

### H13. `fusion-opencode` silently destroys the user's entire OpenCode config on a parse error
`bin/fusion-opencode:69-73,88-89`: if `~/.config/opencode/opencode.json` exists but is
malformed (trailing comma, hand edit, a tool writing JSONC), the launcher catches *all*
exceptions, sets `cfg = {}`, and writes back only the `fusion` provider.
**Scenario:** months of OpenCode config (other providers, agents, keybinds); one corrupted
byte; next launcher run wipes it all, silently, no backup. Permanent data loss of a file
outside the project, caused by a "convenience" script.
**Fix:** abort with an error on parse failure, or back up to `opencode.json.bak-<ts>` before writing.

### H14. Working-tree `fusion.yaml` re-breaks the rate-limit invariant `fusion-claude` claims to protect
`bin/fusion-claude:62-68` comments that `deepseek-v4-pro` "is used by no panel, synth, judge,
router, or simple route, so its rate limit is free to burn" — the reason it is the default
`ANTHROPIC_SMALL_FAST_MODEL`. The current (dirty) `fusion.yaml:105` puts `deepseek-v4-pro`
**on the fusion-coder panel** while `fast-deepseek` (`fusion.yaml:79-81`) still targets it.
**Scenario:** Claude Code fires 80–130 background calls/min at `fast-deepseek`; the bursts
429-starve a panel member — the exact incident the comment documents. HEAD is safe; the local
edit reintroduced the hazard and nothing caught it.
**Fix:** revert the panel member, or add a startup warning when the small-fast model overlaps
any pipeline member.

---

## MEDIUM

### Security
- **Upstream key forwarded on cross-origin redirects.** `src/upstream/openai_compat.ts:81-135`
  never sets `redirect:"error"`; undici re-sends an explicit `Authorization` header to redirect
  targets. The codebase knows this — `src/web.ts:89-92` sets `redirect:"error"` for Tavily with
  a comment describing exactly this leak. The clients carrying the valuable keys lack the guard.
- **`GET /admin/config` returns `extra_headers` values.** `config_editor.ts:53-69` serializes
  providers verbatim; `AccountSchema.extra_headers` (`config.ts:224`) is unconstrained. The
  editor's comment claims "no secret values — only env-var names", which is false for
  `extra_headers` (some providers auth via custom headers; schema supports `authScheme:"Key"`).
- **No inbound request-size limit.** No `bodyLimit` middleware anywhere; `@hono/node-server`
  buffers full bodies in memory (`server.ts:183`, `config_editor.ts:98`). Memory DoS in any
  non-loopback deployment.

### API / compatibility
- **Anthropic route passes OpenAI-shaped error bodies straight through** on `/v1/messages`
  (`anthropic.ts:931-934`) despite `toAnthropicErrorResponse` existing (`errors.ts:127-133`).
  Anthropic SDKs can't parse the body; overloaded-vs-rate-limit handling is lost.
- **Anthropic stream translator emits `content_block_delta` for already-stopped blocks** when
  parallel tool-call indices interleave (`anthropic.ts:681-700`): `start(0)→stop(0)→start(1)→
  stop(1)→delta(0)` — an invalid sequence SDKs reject.
- **Empty upstream stream yields `message_delta`/`message_stop` with no `message_start`**
  (`anthropic.ts:716-731` vs `:805-813`; producible via `failover.ts:329-336`).
- **Client disconnect misreported as `UpstreamTimeoutError` (504)** (`openai_compat.ts:113-116,
  128-131`): routine cancels are indistinguishable from real upstream timeouts in logs/metrics.
- **Anthropic translation silently drops `stop_sequences` and `top_k`** (`anthropic.ts:185-232`
  maps only `max_tokens`/`temperature`/`top_p`; schema accepts both). Stop sequences are a
  standard agent-loop mechanism — runaway generations with no error.
- **`Retry-After` parsed upstream but never forwarded to clients** (`openai_compat.ts:168-175,
  273-281` vs `failover.ts:169`): SDKs retry on their own aggressive schedule, worsening the
  rate-limit storm.

### Strategy logic
- **Smart-router in-flight coalescing propagates one client's disconnect to unrelated
  requests** (`smart.ts:258-265,321-329`): a coalesced waiter's request fails with someone
  else's `AbortError`, and no fallback-to-default occurs despite the "a router failure can
  never fail the request" promise (`:42-43`).
- **Escalation path bypasses the fusion→simple fallback** (`smart.ts:159-169` vs `:196-216`):
  `escalate_on_tool_error` calls `fusionStrategy.execute` directly; panel failure → 502. The
  error-recovery step — where upstream is most likely sick — is the one path without the fallback.
- **Deterministic request-shape errors counted as model health failures**
  (`fusion.ts:1134-1156,1888-1902`): `NativeStreamingNotImplementedError` (native mode +
  image + stream) trips the synth breaker after ~5 occurrences; every request to that synth —
  even non-image — then fast-fails `CircuitOpenError`.
- **Half-open probe leak when body construction throws before the upstream call**
  (`fusion.ts:540-550` panel, `:1127-1131` synth): `buildPanelBody`/`buildSynthBody` run
  *before* the try block; a `MessagesSchema.parse` throw escapes with no recorded outcome →
  `probeInFlight: true` stuck forever.
- **Judge prompt is unbounded and multimodal-blind** (`fusion.ts:1872-1883`): all user+system
  messages concatenated verbatim (router caps at 1800 chars/msg, panel at 200k total — judge
  has nothing); array content flattened to the literal `"[multimodal content]"`, discarding
  the text parts — the router fixed this exact blindness (`smart.ts:577-585`), the judge never
  got the fix. On vision requests the judge adjudicates against a request it cannot read.
- **Synth streaming guard is still a `TransformStream`** (`fusion.ts:1185-1188`) — blind to
  the mid-flight cut its sibling `tool_turn_guard` was rewritten reader-driven to catch
  (`tool_turn_guard.ts:356-362`); and the synth recovery retry is non-streamed
  (`retrySynthForCompletion`, `:1359`), which the tool-turn guard abandoned because silent
  regenerations "died on the ~170s non-stream upstream timeout" (`tool_turn_guard.ts:223-231`).
- **Failover treats an empty 200 stream as member success** (`failover.ts:292-336`):
  keepalive-only stream → `recordSuccess` + client gets an empty SSE body with no `[DONE]`;
  failover neither advances nor records failure against the least healthy member in the chain.
- **Loose intent-to-act heuristics can manufacture an unrequested tool call**
  (`tool_turn_guard.ts:38-72,134-135`): unanchored markers like `"i'll write"` matched in the
  last 140 chars; a legitimately finished turn ending "...next I'll write the migration docs
  if you want" triggers a nudged retry ordered to "Emit the tool call NOW" — and in an
  autonomous loop that call is executed. The header comment's claim that "a genuine completion
  summary never ends on those" is false for these markers; the fusion synth's equivalent list
  (`fusion.ts:1226-1244`) is deliberately narrower.
- **Mid-stream failures after commit are invisible to the circuit breaker**
  (`failover.ts:324,417-451`; `tool_turn_guard.ts:514-535`): `recordSuccess` fires at
  peek-commit; a later cut is never recorded. A member that reliably connects then dies 30s in
  accumulates only successes and is preferred forever. *(suspected-design-gap)*

### Connectors / upstream
- **Live-reload signature omits `api_key_env`, `extra_headers`, `request_timeout_s`**
  (`index.ts:56-69`): rotating a credential or editing headers is "accepted", logs
  "configuration reloaded", and silently changes nothing until restart. Related stale comment:
  `index.ts:38-40` still says "provider/account changes need a restart".
- **Unconsumed error bodies in `listModels` leak sockets** (`openai_compat.ts:186-188`,
  `ollama.ts:45-47`): throw without reading the body; with keep-alive the socket is ineligible
  for reuse. Everywhere else the body is consumed.
- **Streaming upstream has no idle timeout** (`openai_compat.ts:98-119`): the hard timeout is
  cleared when headers arrive by design; the only remaining cancellation is client disconnect.
  A stalled mid-SSE stream hangs forever, holding a per-model concurrency slot. Enough zombies
  exhaust the model's budget; the proxy never self-heals.
- **`Retry-After` overrides cooldown for *all* failure statuses** (`pooled_client.ts:164-169`),
  including 401/402 — design scopes it to 429/503 (`multi-connector-failover-design.md:271`).
  `Retry-After: 0` also yields immediate re-probe via the H10 mechanics.
- **`listGroupModels` ignores connector state** (`provider_router.ts:122-125`): picks the first
  keyed account even when `down`-by-auth; the panel's model picker fails although a healthy
  account could answer. The comment claiming "a cooling/down account still answers discovery"
  is false for dead-key accounts.
- **Manual `off`/`pin` state silently discarded on any providers-section reload**
  (`provider_router.ts:60-67`): operator disables a flaky account; an unrelated `providers:`
  edit rebuilds registries and the disabled account springs back to `up` with no warning.

### Accounting
- **Mid-stream upstream failures invisible to breaker and attribution** (`single.ts:83-92`,
  `fusion.ts:1158-1159`): success recorded at header time; the only mid-stream log
  (`server.ts:299-303`) lacks the `stage`/`upstream_model`/`err_kind` fields
  `attribution.ts:5-20` promises ("CONSISTENT structured shape").
- **Connector-level failover invisible to usage counts and attribution shape**
  (`pooled_client.ts:101-186`): one logical call = N HTTP attempts, recorded once; pooled
  connector failures log ad-hoc shapes without `stage`/`err_kind`/`latency_ms`. The module's
  headline promise ("a grep of the proxy log can answer which model and which stage is
  throttling") stops being true the moment multi-connector failover is on.
- **`UsageAccumulator.finalize()` double-counts under concurrent invocation**
  (`usage.ts:145-160`): the `indexOf === -1` guard skips the splice but not the push. Latent
  today (callers are serialized), one future caller away from inflated cost logs. *(confirmed
  by structure, suspected in practice)*

### Docs / tests / UX
- **CHANGELOG two releases behind.** `package.json` = 0.1.30, tags to v0.1.30, but
  `CHANGELOG.md:15` tops out at `[0.1.28]`; `[Unreleased]` describes precisely what v0.1.29
  shipped; `[0.1.13]`/`[0.1.14]` missing entirely.
- **README's headline preset table describes an abandoned panel.** `README.md:171` says
  fusion-coder = glm-5.2 + kimi + mistral-large, and `:178` says Gemini was removed for
  `thought_signature` 400s; the shipped `fusion.yaml:89` panel has no glm, no mistral, and
  Gemini back in. The decorrelator rationale now argues against the shipped config.
- **`fusion.example.yaml` teaches a config that contradicts the shipped one**
  (`fusion.example.yaml:192-195`: old panel + "NOT gemini here" comment). CHANGELOG v0.1.27
  even claims "the example config were synced" — then v0.1.28 changed the panel and the
  example was never re-synced.
- **Hot-reload documentation contradicts itself.** AGENTS.md ("Changing the `upstream:` block
  needs a restart") and the `index.ts:39-40` comment vs the actual live rebuild at
  `index.ts:72-93` and README:292-294. Two authoritative sources, opposite answers.
- **The "no `as` typecasting" rule is unenforced.** AGENTS.md mandates it; ~20 genuine casts
  exist (`fusion.ts` alone has 16, plus `usage.ts:286`, `reasoning.ts:282`,
  `openai_compat.ts:238,249`, `web.ts:81`). Either enforce or reword.
- **`fusion-opencode` silently downgrades a saved auth token** (`bin/fusion-opencode:66,78`):
  `apiKey` is unconditionally rewritten on every run; run once with `FUSION_PROXY_TOKEN`, later
  without → stored key overwritten with `local-no-auth` → OpenCode starts 401ing with no hint
  why. Also contradicts README:89's `opencode auth login` guidance.
- **Coverage gaps in exactly the newest code:** the live provider-rebuild path
  (`index.ts:72-93`) has no test; `src/timeout.ts`, `src/errors.ts`, `src/logging.ts`,
  `src/index.ts` entrypoint untested; `panel/page.ts` (~400 lines of form-building client JS)
  covered only by a "shell contains `admin/providers`" string check.

---

## LOW (selected)

- `fusion-opencode` writes the client token to `opencode.json` with default umask 0644
  (`bin/fusion-opencode:62-91`) — while `fusion-claude:79-81` carefully uses 0600 for the same
  secret. Inconsistent treatment of the same credential.
- Untracked `fusion.yaml.bak-*` backups accumulate in the repo root, not git-ignored
  (`config_editor.ts:85-86`; six already present) — one `git add -A` from committing internal
  topology.
- No `app.onError` handler: unexpected throws in `/v1/models`, `/ready` yield Hono's plain-text
  500 instead of the OpenAI error shape.
- Native-mode message conversion silently drops assistant `tool_calls` (`vision.ts:125-156`)
  while keeping `role:"tool"` results → orphaned tool results on strict upstreams.
- `show()` discovery ignores `model_map` and tries down connectors first
  (`pooled_client.ts:79`, `registry.ts:178-182`), contradicting design §3.3.
- `classifyThrown` maps any unknown thrown value (incl. programming errors) to a "network"
  health failure (`health.ts:115-120`), cooling healthy connectors for a code bug.
- Client-aborted calls vanish from usage accounting entirely (all strategies skip
  `recordError` on abort) — "measured precisely" (`usage.ts:4-6`) is aspirational.
- Client disconnect mislabeled "upstream stream connection failed mid-way" in logs
  (`server.ts:299-303`) — same line for two opposite causes.
- Capability cache: no TTL, no in-flight dedup (`capabilities.ts:39-59`).
- Dead code: `OpenAiCompatClient.updateConfig` (no callers), `hasStartedDelivering`
  (`fusion.ts:443,499`), `AnthropicStreamOpts.created`, unused `toOpenAiUsage` import.
- Panel-expert ordering is completion order, not config order (`fusion.ts:502-509`) —
  judge/synth prompts non-deterministic run to run, defeating prompt caching.
- `"exactly one tool_calls per step"` (README:552, AGENTS.md) is imprecise — nothing enforces
  array length ≤ 1; the true invariant is "only the synth stage sees `tools`".
- `start.sh` claims to load `.env` (it doesn't) and hardcodes the developer's home path.
- `package.json` `bin` omits `fusion-claude`; neither launcher checks for `opencode`/`claude`/
  `python3` or detects a stale proxy answering `/health`.
- `.env.example` exists on disk but is untracked.
- Design doc drift: flat `connectors:` schema vs shipped `providers:` map; §3.1 contradicts
  §11; section numbering 9→11→10.
- Probe-slot accounting edge in `recordFailure` (`registry.ts:236-237`): `probeInFlight`
  cleared before the epoch guard; at worst a bounded two-probe stampede. *(suspected)*
- `hostOf` fallback can display URL-embedded credentials in the panel on malformed URLs
  (`resolve.ts:112-119`).
- Predictable `/tmp` log path in both launchers (symlink-clobber on shared `/tmp`, LOW on
  macOS per-user TMPDIR).

---

## What is genuinely solid (honest negatives)

- Constant-time token comparison done right (`auth.ts:11-15` — SHA-256 both sides before
  `timingSafeEqual`).
- Client credentials never forwarded upstream; upstream keys never taken from client headers.
- No secrets in git history or image; `.env` properly ignored; thorough log redaction deny-list
  (`logging.ts:10-34`); prompts never logged.
- Panel XSS: single `innerHTML` sink, every dynamic value through `esc()`; token only in
  localStorage → `Authorization` header.
- Atomic config writes with whole-config zod revalidation and timestamped backups; invalid
  hot-reloads keep the previous config; no path traversal via URL params.
- `[DONE]` framing correct across all three SSE guards; CRLF upstreams handled; cancel
  propagation sound per WHATWG spec; no unhandled rejections on the timeout path.
- Failover retry semantics correct and bounded — no infinite-retry path.
- Tool-call truncation mapped to `max_tokens` rather than `tool_use` so clients don't execute
  broken input (`anthropic.ts:490-514`).
- Registry concurrency (epochs, single-flight probes, monotonic cooldowns) matches its design
  doc clause-for-clause; breaker state machine sound; keyed-limiter ordering has no lock cycle.
- No shared-request mutation between panel members; pricing math correct; no unbounded growth
  in tracking structures.
- Test suite: 371 green, mocks injected at the fetch seam — strategies run real code, not
  mocked-away units.

## Verdict

The engineering core — breaker state machines, failover commit rules, SSE framing, token
hygiene, log redaction — is well above average for a project this size, with comments that
document failure modes most projects never find. The failures are at the **seams and the
defaults**:

1. **Deployment defaults are broken.** Docker binds `0.0.0.0` with auth optional and silently
   off when the env var is misnamed; every admin route degrades to unauthenticated; no
   Origin/Host guards — so the config editor's `base_url` rewrite is a one-request
   unauthenticated key-exfiltration primitive (directly, via CSRF, or via DNS rebinding).
   "Localhost single-user" is a hope, not an enforcement.
2. **Streams are the blind spot.** Mid-stream failures are masked as clean `[DONE]` (H5),
   recovered by splicing corrupt completions (H6), recorded as breaker successes (H12
   accounting), unbounded by the concurrency limiter (H12), and unwatched by any idle timeout.
3. **The newest code forgot the old lessons.** Bineval skips the two breaker-hygiene rules
   every other strategy implements (H8, H9); the synth guard didn't get the reader-driven
   rewrite its sibling got; the judge didn't get the router's prompt bounding; the live-reload
   signature didn't get the new fields.
4. **Docs drift faster than code.** CHANGELOG, README preset tables, the example config,
   AGENTS.md's hot-reload and no-`as` claims, and the design doc all assert mutually
   contradictory versions of the truth.

Fix order: H1–H4 (one auth/origin hardening pass), H5–H6 (stream error semantics), H8–H9
(two-line bineval fixes), H13 (back up before overwriting foreign config). The rest is a
hardening backlog.
