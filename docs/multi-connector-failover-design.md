# Multi-connector failover + connector panel — design

> Status: proposed (2026-07-15). Extends llm-fusion from a single upstream to an
> ordered **pool of connectors** (multiple Ollama Cloud accounts + other
> OpenAI-compatible providers) with automatic health-based failover and a local
> web panel. Backward compatible: existing single-`upstream` configs keep working
> unchanged.

## 1. Goal

1. Run **several Ollama Cloud accounts** (and other providers) behind the proxy
   and **switch automatically** when one degrades, hits its limits, or its
   billing lapses.
2. A **local web panel** showing which connector serves now, which are down, and
   **why** — with manual controls (disable / enable / reset / force-switch).
3. Make the provider layer **pluggable** so new OpenAI-compatible providers
   (OpenRouter first; DeepInfra/Together/Novita/… by config) need no new code.

## 2. Current architecture (baseline)

- One `upstream` block → one `OllamaClient` built once in `index.ts`, shared
  everywhere via `ctx.client`.
- `failover` strategy chains over **model names** on that single account; the
  circuit breaker in `concurrency.ts` is keyed **per model**, not per account.
- `attribution.ts` classifies failures as `rate_limit` (429) / `server_error`
  (5xx) / `timeout` / `client_error` — but has **no auth/payment class**, which
  is exactly the "billing ended" signal we need.
- All strategies (`single`/`failover`/`fusion`/`smart`) call
  `ctx.client.chatCompletions(...)`. This is the seam we exploit.

## 3. Core idea — a transparent connector pool

Introduce a `PooledUpstreamClient` that **implements the existing
`UpstreamClient` interface** and is injected as `client` in place of the single
`OllamaClient`. Every strategy keeps calling `ctx.client.chatCompletions(...)`
unchanged; connector-level failover happens **inside** the pooled client. Two
independent, composed failover dimensions:

| Dimension | Scope | Where | Visible in panel |
|---|---|---|---|
| **Connector failover** (NEW) | across accounts/providers | `PooledUpstreamClient` + `ConnectorRegistry` | yes |
| **Model failover** (existing) | across model names in a `chain` | `strategies/failover.ts` | no (unchanged) |

### 3.1 Selection + failover algorithm (per upstream call)

1. Ask the registry for the ordered list of **usable** connectors (config order):
   `up` first, then `cooling`/`down` connectors whose cooldown/recheck window has
   elapsed (as single probes). Skip `off` and connectors still in cooldown.
2. For each candidate, translate `body.model` via its `model_map` (identity for
   Ollama accounts; `qwen3-coder:480b → qwen/qwen3-coder` for OpenRouter), then
   call the connector's client.
3. Classify the outcome and act:

   | Outcome | Reason | Connector action | Loop |
   |---|---|---|---|
   | `< 400` | — | mark `up`, reset counters | **return** |
   | `401` | `auth` | mark `down` | advance |
   | `402` | `payment` | mark `down` | advance |
   | `403` | `moderation` | leave state; treat as client error | **return** (passthrough — not account death) |
   | `429` | `rate_limit` (or `quota` if body says daily-cap/insufficient) | `cooling` (or `down` for quota) | advance |
   | `5xx` | `server_error` | `cooling` | advance |
   | thrown network | `network` | `cooling` | advance |
   | thrown timeout | `timeout` | `cooling` | advance |
   | client abort | — | no state change (`recordAbandoned`) | rethrow |

4. If every candidate failed, return the **last failing result** (so upstream
   error semantics pass through to the client / to the model-`failover` strategy
   above), or throw a typed `AllConnectorsFailedError` (502) / `NoConnectorAvailableError`
   (503) when all were skipped as unusable.

Rationale: multi-account absorbs single-account throttling and billing outages.
Only when the **whole pool** is throttled does the surfaced 429 reach the
existing strategy-level backoff — no double-handling, no behaviour change for the
common single-connector case.

### 3.2 Streaming

We do **not** re-peek in the pool. `OllamaClient`/`OpenAiCompatClient` already
return `kind:"json"` (not a stream) for any non-OK response — even when
`stream:true` was requested. So the pool can safely advance on any pre-stream
failure (a `json` result with a bad status, or a thrown fetch error) and
**commits** to a connector only once it returns `kind:"stream"` (HTTP 200,
headers received, body flowing). A mid-stream failure then surfaces as a stream
error — you cannot switch accounts after the first byte is committed to the
client, which also matches how OpenRouter reports mid-stream limits (SSE
`finish_reason:"error"` under HTTP 200). The existing `failover.ts` peek still
runs on top for model chains, unchanged.

### 3.3 Capability discovery / `show()`

`PooledUpstreamClient.show(model)` routes to the first **healthy connector whose
provider supports native `/api/show`** (i.e. an `ollama` connector). If none, it
throws and `CapabilityService` degrades to overrides/defaults (its existing
graceful path). Generic OpenAI-compatible connectors don't implement `show`.

## 4. Connector health model

State machine per connector (owned by `ConnectorRegistry`):

- **`up`** — serving.
- **`cooling`** — transient failure; auto-probes after `cooldown_s`. Reasons:
  `rate_limit`, `server_error`, `network`, `timeout`.
- **`down`** — hard failure that won't self-heal on a short timer; auto-probes
  only after the longer `down_recheck_s` (default 900 s; `0` = never). Reasons:
  `auth`, `payment`, `quota`. This is the "billing ended" case — stays down until
  topped up or reset.
- **`off`** — operator-disabled via the panel; only manual enable revives it.

Per-connector record (no secrets): `id`, `provider`, `base_url` (host only),
`state`, `reason`, `lastError` (message, never prompt content), `stateChangedAt`,
`cooldownUntil`, `consecutiveFailures`, `totalRequests`, `totalFailures`,
`lastSuccessAt`, `lastFailureAt`, `lastLatencyMs`.

**Active connector** = the first `up` connector in priority order (the one the
next request will use). The panel highlights it.

429-body heuristic (cheap substring scan, no prompt content): if a 429 body
mentions `insufficient` / `quota` / `per day` / `daily`, escalate to
`quota`→`down` instead of `rate_limit`→`cooling`. Covers Groq/Cerebras daily
caps that have no 402.

## 5. Provider abstraction

`src/upstream/`:

- **`openai_compat.ts`** — `OpenAiCompatClient implements UpstreamClient`. The
  OpenAI `/v1/chat/completions` path extracted from today's `OllamaClient`:
  Bearer auth, streaming/non-streaming, `usage`, optional `extra_headers`
  (e.g. OpenRouter's `HTTP-Referer` / `X-Title`), configurable auth scheme
  (`Bearer` default). `show()`/`chatNative()` throw `NotImplemented`.
- **`ollama.ts`** — `OllamaClient extends OpenAiCompatClient`, re-adds native
  `/api/show` (capability discovery) and `/api/chat` (native vision). Behaviour
  identical to today.
- **`provider.ts`** — `createUpstreamClient(connector, opts)` factory / registry
  mapping `provider` → client class. Adding a provider = one entry.

Provider types shipped: **`ollama`** (native-capable) and **`openai-compat`**
(generic). OpenRouter is `provider: openai-compat` + its base URL + optional
ranking headers — and the *same* type immediately supports DeepInfra, Together,
Novita, Nebius, Hyperbolic, DeepSeek, Mistral, Baseten, etc. by config alone.

## 6. Config schema (backward compatible)

`connectors` is new and additive. When absent, a one-connector pool is
synthesised from `upstream` (existing configs unchanged). Pool-wide settings
(`max_concurrency`, `request_timeout_s`, `per_model_concurrency*`, `api_mode`)
stay in `upstream` as defaults; each connector may override `request_timeout_s`,
`base_url`, `api_key_env`.

```yaml
upstream:
  # base_url / api_key_env become OPTIONAL when `connectors` is present.
  max_concurrency: 8
  request_timeout_s: 180
  connector_cooldown_s: 45          # cooling → probe window (transient)
  connector_down_recheck_s: 900     # down → probe window (auth/payment/quota); 0 = never

connectors:
  - id: ollama-primary
    provider: ollama
    base_url: https://ollama.com
    api_key_env: OLLAMA_API_KEY
  - id: ollama-backup
    provider: ollama
    base_url: https://ollama.com
    api_key_env: OLLAMA_API_KEY_2
  - id: openrouter-1
    provider: openai-compat
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    extra_headers:
      HTTP-Referer: https://localhost
      X-Title: llm-fusion
    model_map:
      "qwen3-coder:480b": "qwen/qwen3-coder"
      "glm-5.2":          "z-ai/glm-4.6"
```

Validation (`superRefine`): at least one connector; unique `id`s; each connector
resolves `base_url` + `api_key_env`; either top-level `connectors` OR
`upstream.base_url`+`api_key_env` must be present.

## 7. Web panel (local, quality-first)

Mounted on the same Hono app (localhost bind, gated by the existing
`server.auth_token_env` when set):

- `GET /panel` — self-contained HTML/CSS/JS dashboard (no external requests).
- `GET /admin/connectors` — JSON snapshot the panel polls (~3 s).
- `POST /admin/connectors/:id/disable` · `/enable` · `/reset` — manual controls.

### 7.1 Panel design (thought through up front, per request)

Dark, calm, developer-tool aesthetic; responsive; accessible contrast; motion is
subtle and informative, never decorative.

- **Top bar**: product mark · overall pill (`All healthy` / `N cooling` /
  `N down`) · active-connector name · live "updated Xs ago" + a soft pulse dot.
- **Summary strip**: total / up / cooling / down·off, aggregate requests &
  failure-rate — stat tiles with `tabular-nums`.
- **Connector cards** (grid, active card lifted with a ring): status dot
  (green `up` / amber `cooling` / red `down` / slate `off`) + state label;
  `ACTIVE` badge; provider badge + host; **reason + last error** line; metrics
  (requests, failures, success-rate, last-success/last-failure relative time,
  `cooldown ends in …` countdown for cooling/down); actions
  (Disable / Enable / Reset) with optimistic UI + toast on result.
- Details: system font stack, mono for ids/hosts, tabular numbers for metrics,
  1px hairline borders, low-spread shadows, 150–200 ms ease transitions, a gentle
  stagger as cards mount, `prefers-reduced-motion` honored, `prefers-color-scheme`
  aware. Empty/error states are explicit (never a blank grid).

## 8. Files

**New**
- `src/upstream/openai_compat.ts`, `src/upstream/provider.ts`
- `src/connectors/registry.ts` (health + counters + selection + manual controls)
- `src/connectors/pooled_client.ts` (`UpstreamClient` impl; connector failover)
- `src/connectors/health.ts` (states, reasons, status→reason classifier)
- `src/panel/routes.ts` (Hono sub-app), `src/panel/page.ts` (HTML string)
- Tests mirroring each module under `test/`

**Changed**
- `src/config.ts` (connectors schema, backward-compat synthesis, validation)
- `src/upstream/ollama.ts` (extend the extracted base)
- `src/index.ts` (build registry + pooled client, wire hot-reload, mount panel)
- `src/server.ts` (accept registry, mount panel; `/ready` reflects pool health)
- `README*.md`, `fusion.example.yaml` (connectors + OpenRouter example + provider list)

## 9. Non-goals (this iteration)

- FAL-AI chat connector — **poor fit** (its only OpenAI-compatible path resells
  OpenRouter and uses `Authorization: Key`; real value is queue-based media).
  Documented in `providers-research.md`; not built.
- No per-connector token/cost accounting in the panel (counters only) yet.
- No weighted/round-robin load-balancing — strict priority order with failover.

## 11. Peer-review refinements (incorporated)

An independent architecture review (Grok, 2026-07-15) surfaced real correctness
issues that are folded into the design above:

1. **Rank the surfaced failure, don't return the *last* one.** If earlier
   connectors 429/5xx and the last is 401/402, returning the 401 would make the
   model-`failover` strategy treat it as a healthy 4xx (`recordSuccess`) and hand
   the client an auth error for a recoverable situation. The pool instead returns
   the **most-recoverable** failure by severity: `rate_limit(429)` > `server(5xx)`
   > `network`/`timeout` > `payment`/`auth`/`quota` > `not_found`. Thrown errors
   are rethrown; returned statuses are returned.
2. **`404` / model-not-found → advance, no state change.** A 404 usually means a
   bad `model_map` entry on *that* connector, not connector ill-health; another
   connector may serve the model. `400`/`422` → return immediately (bad request;
   the connector is healthy, trying others won't help).
3. **Empty usable set → deterministic `NoConnectorAvailableError` (503)**, never a
   hang/undefined.
4. **Concurrency (single-thread is NOT enough — interleaving at `await`):**
   - **epoch/generation per connector** — an attempt captures the epoch at start;
     a failure is applied only if the epoch is unchanged; **success bumps the
     epoch**. Prevents a late 429 from overwriting a concurrent success.
   - **single-flight probe** — exactly one probe when a `cooling`/`down` cooldown
     expires (mirrors the circuit breaker's `probeInFlight`); concurrent callers
     skip a probing connector. Prevents a probe stampede.
   - **monotonic cooldown** — `cooldownUntil = max(existing, now+Δ)` and hard
     (`down`) beats soft (`cooling`); a shorter later failure never shortens a ban.
5. **Classification is provider-tunable, not hardcoded:**
   - `403` default → passthrough, but per-connector `treat_403_as: down` for
     gateways (some OpenRouter cases) that use 403 for no-credits/disabled.
   - `429` → `cooling` by default; escalate to `quota`→`down` **only** on an
     explicit per-connector `quota_markers` allowlist (opt-in) — never a broad
     `includes("quota")`, which would false-DOWN a transient rate-limit.
   - `Retry-After` header, when present on a 429/503, sets the cooldown duration
     (surfaced as an optional `retryAfterMs` on the json result) instead of the
     fixed constant.

## 10. Testing & verification

- Unit: registry state machine (each reason → state, cooldown/recheck probes,
  manual controls), status→reason classifier, pooled client selection/advance
  (json + stream, commit rule), config backward-compat + validation, provider
  factory, panel JSON + auth gating.
- Keep the existing suite green (`tsc` clean, all current tests pass).
- Manual: boot with a 2-connector config, force a connector down, watch the panel
  switch the active connector and show the reason; screenshot.
