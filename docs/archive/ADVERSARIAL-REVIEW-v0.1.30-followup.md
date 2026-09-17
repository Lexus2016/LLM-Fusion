# Adversarial Review — llm-fusion v0.1.30 (follow-up)

Follow-up to `ADVERSARIAL-REVIEW-v0.1.30.md`, run against the **working tree** (the in-progress
hardening pass, ~1.2k uncommitted lines). Method: four parallel adversarial auditors
(fix-verification, streaming/strategy correctness, panel security+UX, ops+benchmark integrity),
hand-verified against the current source, plus two independent external reviewers via Consilium
(OpenAI `codex`, Google `agy`). Nothing below is fabricated; unconfirmed items are marked
*(suspected)*. Where this pass **applied a fix**, it is labelled **[PATCHED here]** and lists the
test that exercises it.

---

## 0. Executive verdict

The hardening pass is real and mostly correct: **H2, H3, H5, H8, H9, H10, H11, H13**, the
live-reload signature, `extra_headers` redaction+restore, `redirect:"error"`, `/v1/*`+`/admin/*`
body caps, smart escalation→fallback, and Anthropic `stop_sequences`/`top_k` are **genuinely
closed with tests that exercise the vulnerability**. XSS discipline is genuinely solid.

Three things undercut the release:

1. **The product's core claim is contradicted by its own committed benchmark data** (§1). This is
   the deepest issue and is *not* a code bug — it is a claim the evidence does not support.
2. **The one security fix that "looked done" was not** — the CSRF guard advertised DNS-rebinding
   protection its code did not implement, leaving the H1 key-exfiltration primitive reachable.
   **[PATCHED here]** (§2).
3. **Streams remain the blind spot** even after the pass: silent tool-call corruption on recovery
   (H6, partial), truncated-tool-call → `tool_use` (Anthropic), unbounded streamed connections
   (H12). Documented, not patched — see §3 for why.

---

## 1. CRITICAL — the benchmark refutes the headline claim

`README.md:7` says fusion answers "come out noticeably stronger than what any one of those models
produces alone." The project's own most recent **blind, full 15/15** run
(`bench/run-v0127-4panel.log`, 0/120 call errors) says the opposite (total /30, recomputed):

| condition | total/30 |
|---|---|
| fusion-agents (smart-router → mostly solo-glm) | 27.47 |
| **solo-glm** (one model) | **27.13** |
| **solo-gemini-flash** (one panel member) | **27.13** |
| solo-deepseek-flash | 24.27 |
| **fusion (the product)** | **23.60** |

Fusion loses to a single model by ~13%. The only winning "fusion" condition is the smart-router,
which by design mostly routes to plain solo-glm.

- **C2 — methodology-shopping.** The marketed "fusion=0.99 vs 0.89 solo" (`CHANGELOG.md`,
  `docs/llm-fusion-audit.md`, `fusion.yaml`) comes from `results-synthglm.json` (n=14) and
  `results-sologlm-pa.json` (n=15) — scored **per-answer, not blind/shuffled/joint**, unequal n,
  and **the generated answers are not committed**, so it cannot be re-scored. The blind runs that
  *are* committed and auditable show fusion losing.
- **C3 — cherry-pick.** The "29.56/30 on the 9/15 clean tasks" claim excludes exactly fusion's
  worst 6 tasks. No committed clean 15/15 run has fusion winning.
- **Cost/latency omitted.** Fusion is ~6× the upstream calls (4 panel + judge + synth + bineval)
  and slower (log T15: 41s vs solo 4–32s) — absent from the verdict.

**Honest credit:** the benchmark *harness* is engineered honestly — blind/shuffled/anonymized
joint scoring, an independent third-model grader (`deepseek-v4-pro`), and it prints
"❌ ПОСТУПАЄТЬСЯ" and **commits the disconfirming data**. The integrity failure is in the
marketing layer (README/CHANGELOG/fusion.yaml/docs) selectively quoting the one favorable,
non-blind, uncommitted-answers experiment.

**Recommendation:** either drop the "beats every single model" claim, or publish a clean, blind,
committed 15/15 run (answers included) where fusion wins on quality *net of its 6× cost*. As
shipped, the data says a single glm-5.2 is better, cheaper, and faster.

---

## 2. HIGH — security: the DNS-rebinding half-fix  **[PATCHED here]**

The in-progress `adminApiGuard` closed classic CSRF (Origin+content-type) but its docstring called
itself a "CSRF / DNS-rebinding guard" while checking only `Origin === Host` — which does **not**
stop DNS rebinding: after `evil.tld → 127.0.0.1` rebind, the page is same-origin, so
`Origin == Host == evil.tld:port` match and `application/json` is allowed. Combined with `base_url`
still being `z.string().url()` (any scheme/host/userinfo), the H1 upstream-key exfiltration
primitive stayed reachable against the default loopback + no-auth deployment. Confirmed
independently by both Consilium reviewers.

**[PATCHED here]** — two defense-in-depth layers, either of which alone closes it:

- **Host pinning** (`src/panel/config_editor.ts`, `makeAdminApiGuard`): when auth is **off** (the
  vulnerable default AND any `FUSION_ALLOW_OPEN` deployment), the `Host` must be a strict loopback
  name (`127.0.0.0/8` with valid octets, `localhost`, `::1`); a **missing or malformed Host fails
  closed** (403). A rebound page always carries its own non-loopback hostname → rejected, and a
  non-browser client (HTTP/1.0, `curl --http1.0`) cannot slip past on an allow-open box either.
  Skipped when auth is enforced (the "front it with your own auth" deployment; a rebinding page
  cannot read another origin's token). `transfer-encoding` and a nonzero/malformed `content-length`
  both count as a body (closes the chunked / bogus-length content-type bypass). Wired via
  `authEnforced: () => Boolean(getAuthToken())` from `createApp`.
  Tests: `test/panel.test.ts` — rebinding Host → 403, `127.evil.com`-class → 403, missing Host →
  403 (fail-closed), chunked-no-content-type → 415, non-loopback Host allowed when auth on.
- **`base_url` constraint** (`src/config.ts`, shared `baseUrlSchema`): https-only, with an explicit
  loopback-http exception (local Ollama), and no embedded userinfo.
  Tests: `test/config.test.ts` — rejects `http://evil…`, rejects userinfo, accepts https + loopback
  http, applies to per-account `base_url`.

Reviewed by Consilium twice (`codex`, `agy`): core rebinding closure **confirmed**; the octet regex
was tightened, the missing-Host case was made fail-closed, the bracketed-IPv6 parse was hardened,
and the `content-length` NaN edge was closed — all per reviewer notes.

**Operator note (behavior change worth documenting):** an **unauthenticated** admin plane is now
strictly loopback-only, *including* under `FUSION_ALLOW_OPEN`. To administer remotely you must
either configure a client token (`server.auth_token_env`), or have your fronting reverse proxy
rewrite the `Host` header to `localhost` before forwarding to `/admin/*`.

**Still open (documented, not patched):**
- No CSP / `X-Frame-Options` on `/panel` → clickjacking of the token-authenticated panel; the
  banner is `http://` so on any non-loopback deploy the Bearer token crosses the LAN in cleartext.
- `src/index.ts` `isLoopbackBind` uses `bind.startsWith("127.")` (F3): a `127.example.com` bind
  (DNS → public IP) is misclassified as loopback → fail-fast skipped → open public proxy.
  Contrived, but a real hole in a security gate. (My new guard code uses a strict octet regex; the
  boot-time check in `index.ts` — actively being edited — was left for the owner.)
- Log redaction is a **deny-list** (fragile): misses arbitrary-named `extra_headers` auth values,
  `password`/`secret`/`cookie`/`proxy-authorization`, and nesting beyond one wildcard level.

---

## 3. HIGH — correctness: streams

- **H6 — tool-turn guard splices a duplicate index-0 tool call (still present, verified in code,
  NOT patched — owner decision).** `src/strategies/tool_turn_guard.ts`: `handleLine` forwards every
  non-terminal `data:` line live (line 462), **including `delta.tool_calls` fragments**. On a
  `finish_reason:"length"` mid-tool-args truncation, the terminal chunk is held and
  `finishNormally` runs `runStreamingRecoveryWithKeepalive`, which re-emits a tool call restarting
  at `index:0`. Index-keyed client accumulators (openai-python, Vercel AI SDK, OpenCode)
  **concatenate** it onto the truncated arguments already delivered → invalid JSON — the exact
  `broken_tool_call` the guard exists to prevent.
  **Why not patched here:** the naive gate (skip recovery once tool fragments were forwarded)
  *breaks the developers' deliberate, tested feature* (`test/single.test.ts:392` recovers exactly
  this case). The truly correct fix is a design decision the owner must make:
  **(a)** buffer `tool_calls` deltas instead of forwarding them live (then a truncation delivered
  nothing and recovery is clean — preserves the feature), or **(b)** fail honest: forward the
  terminal `finish_reason:"length"` chunk and let the client retry the turn (loses the feature but
  never corrupts). Option (a) re-architects the guard's hot path — out of scope for a drive-by
  patch on an actively-edited file. Flagged for the owner rather than silently reversing a tested
  behavior.
- **Anthropic: truncated tool call → `stop_reason:"tool_use"`** for every `finish_reason` except
  `"length"` (`src/anthropic.ts:495-519`). A `"stop"` (or no-terminal) end with unparseable args
  yields `input:{}` + `tool_use` → the client executes an empty/broken tool call. Fix: apply the
  `toolInputsComplete` check whenever `hasToolBlocks`. *(not patched — actively-edited file)*
- **H12 — concurrency limiter releases at header time** (`openai_compat.ts` returns `stream` on
  headers; `single`/`fusion`-synth/`failover` release then) → streamed/agent connections are
  unbounded; no stream idle timeout. Real DoS/budget-exhaustion on any non-loopback deploy. Fix:
  hold the slot until the stream drains (as the fusion panel already does). *(architectural — owner)*
- Also still present, lower: interleaved parallel tool indices emit `content_block_delta` after
  `content_block_stop`; empty upstream stream emits `message_delta`/`message_stop` with no
  `message_start`; smart-router in-flight coalescing propagates one client's abort to an unrelated
  request (500); synth guard is a `TransformStream` (blind to mid-flight cut; leaks `planning_tail`
  reasoning to visible content when `promote` is on); failover records an empty keepalive-only 200
  stream as member success (an always-empty member is preferred forever).

---

## 4. MEDIUM / ops

- **H14 reintroduced in the working tree** (`fusion.yaml`): `deepseek-v4-pro` on the fusion-coder
  panel while it is also the `fast-deepseek` target and the default `ANTHROPIC_SMALL_FAST_MODEL` —
  Claude Code's 80–130 background calls/min will 429-starve a live panel member (the exact incident
  the `fusion-claude` comment claims to prevent). It also taints any bench re-run (the scorer,
  `deepseek-v4-pro`, becomes a panel member → self-preference). *(their uncommitted config — not
  touched; revert to `deepseek-v4-flash` or add a startup warning on small-fast/pipeline overlap.)*
- **Docker:** `node:24-slim` is a floating tag (not digest-pinned); no `USER` (runs as root); no
  `HEALTHCHECK`; ships `tsx` at runtime in prod (no build/typecheck gate).
- 19 `as` casts remain despite `AGENTS.md`'s ban (`fusion.ts` has 12); `fusion.ts` is 2104 lines /
  62 functions (God file). `start.sh` is untracked+unignored, hardcodes a wrong-case home path, and
  claims to load `.env` but does not.

---

## 5. UI / UX (panel)

- **U1 [HIGH-UX, data loss]:** the model form silently discards typed input when the async catalog
  resolves (`rebuild()`) and on provider-dropdown change — type a judge/synth, switch provider,
  it's gone; save then writes partial.
- **A11y [MED]:** toggles are keyboard/screen-reader dead (no `role="switch"`/`aria-checked`/focus);
  input labels are not associated (`for`/`id`); modal focus is not trapped or restored; toasts have
  no `aria-live`. Parts of fusion-model config are unreachable without a mouse.
- Lower: adding a 2nd provider silently re-homes unbound models; `extra_headers` are invisible and
  un-editable in the panel; every action confirms (even non-destructive); a wrong token gives no
  distinct feedback; a fixed `setTimeout(400ms)` post-save races the hot-reload.

---

## What this pass changed

Applied and verified (403 tests green, `tsc` clean, Consilium-reviewed twice):
`src/config.ts`, `src/panel/config_editor.ts`, `src/panel/routes.ts`, `src/server.ts`
(Host-pinning + `base_url` constraint + chunked/NaN-body hardening + missing-Host fail-closed), plus
`test/config.test.ts`, `test/panel.test.ts`, `test/config_editor.test.ts`, `test/model_discovery.test.ts`.
`src/strategies/tool_turn_guard.ts` was **explored and reverted** — the H6 fix is an owner design
decision (see §3).

## Fix order (remaining)
1. §1 — reconcile the benchmark claim with the data (drop it, or publish a clean winning run).
2. §3 — H6 (pick buffer-vs-fail-honest), Anthropic `toolInputsComplete`, H12 slot hold.
3. §2 residual — CSP/X-Frame-Options; `isLoopbackBind` strict match; redaction allow-list.
4. §4 — revert the `deepseek-v4-pro` panel member; Docker `USER`/digest-pin/healthcheck.
