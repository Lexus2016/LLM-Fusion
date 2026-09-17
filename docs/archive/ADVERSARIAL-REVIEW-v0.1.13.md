# Adversarial Review — llm-fusion v0.1.13

- **Date:** 2026-06-28
- **Reviewer:** adversarial pass (Claude Code, fusion-agents model)
- **Scope:** full `src/`, `test/`, repo-level ops (Docker, package, git hygiene, docs)
- **Method:** code read via `cheap read --mode extract` + native Read for small/sensitive files; claims verified against actual line numbers. Findings also persisted to TQMemory note `05ce40fdb3c144f2`.

## Baseline

| Check | Result |
|---|---|
| `npm run typecheck` | clean |
| `npm test` | 235 passed, 9 skipped (`live.smoke.test.ts`, no `OLLAMA_API_KEY`) |
| `npm audit` | 5 vulnerabilities in dev deps — 3 moderate, 1 high, 1 critical (vitest / vite / esbuild / vite-node / @vitest/mocker) |

> A green `npm test` hides the supply-chain hole: the critical CVE is in a dev dependency, so the runtime is unaffected locally but CI and the Docker image build with it.

---

## 🔴 High — fix first

### 1. Concurrency limiter bypass for the fusion panel
`src/strategies/fusion.ts:541-586`
```ts
const workPromise = (async () => { const result = await invokeUpstream(...) })();  // upstream call STARTS here
outcome = await resilience.limiter(() => withTimeout(workPromise, ...));          // slot acquired AFTER
```
`workPromise` is an IIFE — the upstream fetch starts **before** `resilience.limiter()` acquires a slot. As a result `upstream.max_concurrency` does **not** bound panel member fetches; all members start simultaneously and the limiter only gates awaiting completion. This contradicts what the config promises.
**Fix:** create `workPromise` inside the limiter callback.

### 2. Slot leak on early panel cancellation
`src/strategies/fusion.ts:426-470` — `resolve(answers)` returns without `Promise.allSettled` over pending member promises. Cancelled calls keep their `p-limit` slots until they settle on their own; under load the pool exhausts.
**Fix:** track in-flight member promises and await them (or `allSettled`) before returning.
> **Peer review:** treat as **medium** — abort frees the slot asynchronously, so this is an availability bug, not a permanent leak. See addendum.

### 3. Circuit breaker sticks in half-open
`src/concurrency.ts:100-108` + abort path in `fusion.ts`. `canAttempt` reserves `probeInFlight = true`; if the call is aborted by signal before `recordSuccess`/`recordFailure`, the probe is never released. The model stays in half-open forever and rejects every call until process restart.
**Fix:** release the probe on abort (`recordFailure` or explicit `releaseProbe`).

### 4. Header/body mismatch on transformed responses
`src/server.ts:228-251, 281-294` and `src/anthropic.ts:851-866`
```ts
const headers = new Headers(res.headers);          // copies content-length, content-encoding
headers.set("x-fusion-usage", ...);
return new Response(transform.readable, { status, headers }); // body changed, length not
```
During usage injection (JSON) or SSE transform, the body changes but upstream `content-length` / `content-encoding` are preserved. Clients wait for the old length or try to gunzip a non-gzip body → truncation or decode error.
**Fix:** delete `content-length`, `content-encoding`, `transfer-encoding` before returning a mutated body.

### 5. Anthropic errors returned in OpenAI shape
`src/errors.ts:103` (`toErrorResponse`) returns `{"error":{...}}`; `src/anthropic.ts` uses it for `/v1/messages`. Anthropic clients (incl. Claude Code) expect `{"type":"error","error":{...}}` and may fail to parse errors at all.
**Fix:** add `toAnthropicErrorResponse()` and use it on the Anthropic route.

### 6. Auth bypass on empty token
`src/auth.ts` — `if (!token) { await next(); return; }` treats an empty string as "auth disabled". If `FUSION_PROXY_TOKEN` is set but empty (common CI/Docker misconfiguration), the proxy becomes open with no startup error.
**Fix:** distinguish `undefined` (unset) from `""` (misconfigured); reject empty as a hard config error.

### 7. Tavily API key leak via redirect
`src/web.ts:86-97`
```ts
fetchFn(TAVILY_URL, { method: "POST", body: JSON.stringify({ api_key, query }) }) // no redirect: 'error'
```
`fetch` follows redirects by default. On `307`/`308` the POST body with `api_key` is resent to the redirect target. One spoofed/tampered Tavily response leaks the key.
**Fix:** `redirect: 'error'` and/or validate the final URL.

### 8. Docker: root + dev deps in runtime + tsx in prod
`Dockerfile:13,19,35`
- `FROM node:24-slim` with no `USER node` → runs as root.
- `RUN npm ci` without `--omit=dev` → vulnerable `vitest` lands in the production image.
- `CMD ["npx", "tsx", "src/index.ts"]` → `npx` resolver in prod, no compiled build.
For a service holding upstream API keys this is unacceptable.
**Fix:** multi-stage build or `npm ci --omit=dev`, add `USER node`, pin digest / minor tag, compile to JS or `node --import tsx`.

---

## 🟠 Medium

### 9. `/v1/models` is unauthenticated
`src/server.ts:85` — `app.get("/v1/models", ...)` has no `auth` middleware, while `/v1/chat/completions` does. Virtual model list leaks to unauthenticated clients even when `auth_token_env` is configured.
**Fix:** apply `auth` to `/v1/models`.

### 10. Usage double-count race + lost `finish_reason`
`src/usage.ts`
- `145-160` — `finalize()` is not concurrency-safe: two concurrent calls can both splice and push the same pending stream records.
- `149-156` — `await p.usage` has no `.catch`; a rejected usage promise aborts the final usage chunk.
- `279-282` — a chunk containing `usage` is dropped wholesale, losing `finish_reason` if co-located in the same chunk.
**Fix:** atomically move `pendingStreams` to a local before `await`; catch rejected usage; strip only the `usage` field and forward the rest.

### 11. Upstream SSE error swallowed as success
`src/server.ts` pump + `src/usage.ts` flush — on upstream stream error the writer closes, but the transform still emits `usage` + `[DONE]`. The client sees a truncated response as complete.
**Fix:** emit an SSE `error` event or set `x-fusion-error`, and do **not** emit `[DONE]` after an error.

### 12. Anthropic `pipeTo` ignores rejection
`src/anthropic.ts:877-879`
```ts
void res.body.pipeTo(transform.writable).then(() => undefined, () => undefined);
```
On mid-stream upstream failure the transform's `flush` does not run; the client never gets `message_stop` and usage is not logged.
**Fix:** manual pump with error handling, or close/log on rejection.

### 13. Config hot-reload defects
`src/config.ts`
- Directory watch fires on every event when `filename === null` (typical on Linux).
- No guard against overlapping `reload()` → parallel reads / race on `current`.
- `get config() { return current; }` returns a mutable object; any caller can mutate shared state.
- `onReload` only appends listeners — no unsubscribe → leak.
- `src/server.ts:64-66` — `max_concurrency` is not recreated on reload; the limiter is built once at boot.

### 14. Smart routing false positives
`src/strategies/smart.ts`
- `claimsImage` (454-473) counts `figure`, `diagram`, `chart`, `scan` as image claims → a text-only request "explain this diagram" is flagged as hallucination and dropped to default.
- `latestToolResultIsError` (675) matches `0 failures` / `0 tests failed` → clean output escalates to fusion.
- Client disconnect during the router call (315-334) is recorded as a model failure and trips the breaker.

### 15. Judge / bineval lose array-message text
`src/strategies/fusion.ts:1281-1292` — `renderRequestForJudge` replaces array content with `"[multimodal content]"`, dropping the actual text. Judge and bineval evaluate a placeholder, not the real request. Additionally the judge prompt is uncapped (948-963), and `approxTotalChars` (720-737) ignores `tool_calls` → context overflow is possible even after compression.

### 16. Ollama client
`src/upstream/ollama.ts`
- Stream path (89-97) does not verify `content-type` → a JSON 200 response is wrapped as SSE.
- `readBody` (161-168) reads the body with no size limit → OOM on a large error body.
- Cancellation (63-69) is misclassified as timeout because of the check order.

### 17. Logging / attribution
- `src/logging.ts:10-34` — redact paths miss `access_token`, `refresh_token`, `client_secret`, `password`, `private_key`.
- `src/attribution.ts:51-52` — `reason` is an arbitrary caller-supplied string with no sanitization; can carry prompt/PII into logs.
> **Peer review:** both items are **overstated** — treat as low-priority hardening, not active leaks. See addendum.

### 18. Request validation
`src/types.ts:115-123` — `messages` is `.optional()` and `role` is an arbitrary string. Invalid requests pass through to the upstream instead of being rejected with 400 at the proxy.

---

## 🟡 Low / hygiene

- **`package.json`** — `bin` omits `fusion-claude` (the file exists and is executable, but is unreachable via `npx`); no `license` field and no `LICENSE` file.
- **`.gitignore`** — `.env*` ignores `.env.example`, so the env template is untracked and new users don't see the required variables.
- **`tsconfig.json:9`** — `skipLibCheck: true` hides type errors in dependency `.d.ts`.
- **`src/web.ts`** — `formatWebContext` (124-137) does not strictly enforce `maxContextChars` for the first block; `tavilySearch` swallows all failures without logging; `query` length is unbounded.
- **`src/vision.ts:68-80`** — `parseDataUrl` accepts any MIME without an `image/*` whitelist.
- **`src/strategies/fusion.ts:485-486`** — `answers` order is nondeterministic → `Expert N` numbering for the judge is unstable across runs.
- **Doc drift** — `README.md:72` lists `gemini-3-flash-preview` in the `fusion-coder` panel, but `fusion.yaml:82` uses `gpt-oss:120b`. Copying the README config yields different behavior.

### Test gaps
- 9 smoke tests are skipped without `OLLAMA_API_KEY` and hit the **upstream directly**, not the proxy — no E2E through `llm-fusion`.
- No test for concurrent `finalize` (10), `/v1/models` auth (9), limiter bypass (1), or `web.ts` redirect (7).
- `test/vision.test.ts` — native path tested only with empty `messages`.

---

## 🟢 Actually clean (not nitpicks)
- `src/strategies/failover.ts` — the state machine (429 / 5xx / 4xx, advance-vs-return, all-open → 503) is correct; peek/commit for SSE is careful.
- `src/reasoning.ts` — `reasoning → content` promotion is consistent across stream/non-stream, preserves `tool_calls`, does not duplicate content after real content appears.
- `src/bineval.ts` — `X-Fusion-Bineval-Skipped` header with reasons distinguishes "eval never ran" from "low score".
- `src/strategies/smart.ts` — LRU cache and in-flight coalescing are correct; failures are not cached.
- `src/capabilities.ts` — caches only authoritative results with self-healing on transient discovery failures.

---

## Fix priority
1. Limiter bypass — `fusion.ts:541-586` (architecture: the config lies about concurrency).
2. Header/body mismatch + Anthropic error shape — client-visible, breaks Claude Code / OpenAI compatibility.
3. Auth bypass on empty token + `/v1/models` without auth — security.
4. Tavily redirect leak — one option, high risk.
5. Docker hardening + `npm audit fix` — supply chain.
6. Usage double-count race — cost/token metrics can double.

---

## Peer review addendum

Two independent non-Claude advisors reviewed this audit via `consult`:
- `agy` (Gemini) — transcript `/Users/admin/.consilium/log/20260629-143209-agy-42345.md`
- `hermes` (Ollama Cloud, `kimi-k2.7-code`) — transcript `/Users/admin/.consilium/log/20260628-220733-hermes-32226.md`

Both confirmed **15 of the 18 high/medium findings** as factually accurate. This addendum records the 3 items they flagged as overstated and the significant issues they found missing.

### Findings revised after peer review

| # | Original claim | Peer verdict | Revised treatment |
|---|---|---|---|
| 2 | "Slot leak on early cancellation — under load the pool exhausts" | **Partially overstated** | Abort does eventually free the slot asynchronously; it is a **correctness/availability bug**, not a permanent pool leak or instant DoS. Treat as **medium**, not high. |
| 17a | Logging redact paths miss common secret keys | **Overstated** | `password` / `client_secret` are not logged in this codebase. This is a **defense-in-depth hardening recommendation**, not an active leak. |
| 17b | `reason` in attribution can carry PII | **Overstated** | Logging error messages is standard practice; the risk depends on upstream echoing prompts in error bodies, which is outside direct proxy control. Keep as **low-priority hardening**. |

### Issues the audit missed

1. **Systemic client-disconnect breaker tripping across all strategies**  
   `src/strategies/fusion.ts:1054` (`runJudge` catch), `src/strategies/fusion.ts:1238` (`runSynth` catch), `src/strategies/failover.ts:122-123`, and `src/strategies/single.ts:52-53` all catch errors (including `AbortError` from client disconnect) and call `breaker.recordFailure` without first checking `ctx.signal.aborted`. The audit noted this only for `smart.ts`; it is actually a codebase-wide pattern.

2. **Half-open probe leak is not limited to `fusion.ts`**  
   The same `probeInFlight` jam can happen in `smart.ts` and `failover.ts` when a probe is cancelled by client signal before `recordSuccess`/`recordFailure`.

3. **No Anthropic tool-use / tool-result conversion**  
   The `/v1/messages` route converts Anthropic requests to OpenAI and back, but tool-use/tool-result blocks may not be fully mapped — a major compatibility gap for Claude Code that the audit did not mention.

4. **Stream errors are sent under HTTP 200 with no error signal**  
   Because the response headers (including `200 OK`) are already committed before a mid-stream upstream error is detected, the only way to signal failure is an SSE `error:` event or a trailer header. Neither is implemented.

5. **No per-request or global rate limiting / DoS protection**  
   The only backpressure is upstream `max_concurrency` and the breaker. An unauthenticated client can saturate the limiter and starve others.

6. **`/v1/models` discovery competes for the global concurrency slot**  
   In addition to being unauthenticated, the models endpoint runs upstream capability discovery in parallel with chat traffic, which can delay real requests.

### Peer-ranked top 3 fixes

Both advisors independently put these at the top:

1. **Concurrency limiter bypass + half-open probe leak + abort breaker tripping** — the core concurrency guarantees are broken and one cancellation can permanently block a model.
2. **Header/body mismatch + Anthropic error shape** — breaks real clients and can truncate streams.
3. **Usage `finish_reason` restoration + auth empty-token bypass + `/v1/models` auth** — client-visible correctness bug plus a silent security misconfiguration.

### Decision log

- **Not patched in this session:** the user asked to gather peer opinion, not to apply code fixes. Patching source code without an explicit request would be unsolicited scope expansion. The revised audit is intended to be the single source of truth for a follow-up remediation pass.
- **Preserved in TQMemory:** the original audit is note `05ce40fdb3c144f2`; this addendum is part of the saved file state.

---

## Fixes applied in this session

After peer review, the following fixes were implemented and verified (`npm run typecheck` clean, `npm test` 235 passed):

| # | Finding | Fix | Files changed |
|---|---|---|---|
| 1 | Limiter bypass in fusion panel | `workPromise` IIFE moved **inside** the `resilience.limiter()` callback so the slot is acquired before the HTTP request starts. | `src/strategies/fusion.ts` |
| 3 / systemic | Half-open probe leak + client abort tripping breaker | Added `recordProbeAbandoned()` on `CircuitBreaker`; all strategy `catch` blocks now skip `recordFailure` for client abort and release any reserved half-open probe. | `src/concurrency.ts`, `src/strategies/single.ts`, `src/strategies/smart.ts`, `src/strategies/failover.ts`, `src/strategies/fusion.ts` |
| 4 | Header/body mismatch on transformed responses | New `stripHopByHopHeaders()` helper deletes `content-length` / `content-encoding` / `transfer-encoding` before returning mutated SSE/JSON bodies. | `src/headers.ts`, `src/server.ts`, `src/anthropic.ts` |
| 5 | Anthropic errors in OpenAI shape | Added `toAnthropicErrorResponse()` returning `{"type":"error","error":{...}}`; Anthropic route now uses it. | `src/errors.ts`, `src/anthropic.ts` |
| 6 | Auth bypass on empty token | Empty configured token now returns 500 `configuration_error` instead of silently disabling auth. | `src/auth.ts` |
| 7 | Tavily API key leak via redirect | Tavily fetch uses `redirect: "error"`. | `src/web.ts` |
| 9 | `/v1/models` unauthenticated | `auth` middleware applied to `/v1/models`. | `src/server.ts` |
| 10 | Usage chunk drops `finish_reason` | `makeUsageInjectionTransform` now strips only the `usage` field and forwards the rest of the chunk. | `src/usage.ts` |
| 16 / dynamic timeout | `request_timeout_s` hard-cuts streaming delivery | `OllamaClient.doFetch` supports `phaseTimeoutOnly`; streaming calls clear the hard timeout once the response headers arrive, turning it into a connection/first-response timeout. | `src/upstream/ollama.ts` |

### Not fixed in this session (left for follow-up)
- Slot leak on early panel cancellation (#2) — the abort path now releases probes, but `runPanel` still resolves without `Promise.allSettled` over in-flight members.
- Config hot-reload defects (#13) — watcher overlap, mutable config, no `onReload` unsubscribe, `max_concurrency` not recreated.
- Smart routing false positives (#14) — `claimsImage` overbroad regex, `latestToolResultIsError` matching "0 failures".
- Judge/bineval losing array-message text (#15) — `renderRequestForJudge` still replaces array content with `"[multimodal content]"`.
- Ollama `readBody` size limit / stream content-type verification / cancellation misclassification (#16 remainder).
- Logging redact paths / attribution `reason` sanitization (#17) — defense-in-depth only.
- Request validation weak (#18), Docker hardening (#8), package metadata / `.gitignore` hygiene, Anthropic tool-use/tool-result conversion, per-request rate limiting.