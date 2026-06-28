# llm-fusion — Fusion Proxy

> **Self-hosted, OpenAI-compatible and Anthropic Messages API deliberation proxy for Ollama Cloud.**  
> One virtual model name runs a panel of models, a judge, and a synthesizer — or a smart router that decides per request whether that heavy treatment is even worth it.

If you know **OpenRouter Fusion**, you already get the idea: *many models think, one answers*.  
**llm-fusion is the same pattern, rebuilt for developers who want control, transparency, and agent-loop safety.** It runs locally, routes to Ollama Cloud, and adds the `smart` strategy that OpenRouter leaves to the caller.

No database. No build step. Node 24 + TypeScript + Hono, runs `.ts` directly via `tsx`. One YAML config.

---

## Quickstart

```bash
git clone https://github.com/Lexus2016/LLM-Fusion.git llm-fusion
cd llm-fusion
npm install                                        # Node ≥ 24
printf 'OLLAMA_API_KEY=ollama-your-key\n' > .env    # your Ollama Cloud key
npm start                                          # proxy on http://127.0.0.1:8080
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1` and ask for a model by name — or launch OpenCode in one command:

```bash
./bin/fusion-opencode fusion-coder                # starts proxy if needed, wires OpenCode, opens TUI
```

New here? Read **[Which model do I use?](#which-model-do-i-use)** next.

---

## Why llm-fusion instead of OpenRouter Fusion?

Both run a prompt through a panel of models plus a judge/synth step to produce a stronger answer. The difference is **where, when, and how** that pipeline runs.

| | **llm-fusion** | **OpenRouter Fusion** |
|---|---|---|
| **Hosting & control** | Runs on your machine / Docker. Single-tenant, inspectable, one config file. | Fully managed SaaS on OpenRouter. |
| **Bill** | Pay Ollama Cloud directly. | Pay OpenRouter; usage is bundled. |
| **Provider scope** | Ollama Cloud only (one bill, one upstream). | Any provider on OpenRouter (broader model catalog). |
| **Automatic routing** | Built-in `smart` strategy: a fast LLM router picks `single` (cheap) vs `fusion` (deep) **per request**. | Fusion always runs the full panel; routing is manual or a separate router. |
| **Agent-loop safety** | Emits **exactly one `tool_calls`** per step; panel never touches real tools in `deliberate` mode. | Plugin returns analysis; the calling model decides final tool use. |
| **Cost knob for long runs** | `fusion_planning_turn_only`: full panel on the planning turn, then synth-only (5 calls → 1) for mechanical mid-loop steps. | No per-request auto-downgrade inside a loop. |
| **Web grounding** | Optional, opt-in: one Tavily search before the panel, results injected as prose context. Gated on `TAVILY_API_KEY` (env) **and** `web_search.enabled` (config) — fully OFF unless both are set. | Built-in `web_search` + `web_fetch` for panel and judge. |
| **Resilience** | `failover` chains + graceful degradation + `max_concurrency` cap. | Managed reliability + recursion protection. |
| **Config style** | Single hot-reloaded YAML with empirically tuned presets. | Plugin JSON / server tool / model slug; presets + web UI. |

**Use llm-fusion when:**

- You run an autonomous agent for **hundreds of steps** and need deterministic, exactly-once side effects.
- You want routine steps to cost **1 upstream call**, with fusion reserved for hard or error-recovery steps.
- You prefer a **local, transparent process** with one config file and no extra vendor lock-in.
- Your models and budget live on **Ollama Cloud**.

**Use OpenRouter Fusion when:**

- You need **web-grounded research** across frontier models from many providers.
- You want a **managed service** with a web playground and global infrastructure.
- You are doing one-off deep research, not long agent loops.

**Bottom line:** OpenRouter Fusion optimizes for single-answer ceiling on research tasks. **llm-fusion optimizes for running inside an agent loop** — controlling cost, preventing duplicate tool calls, and keeping the heavy deliberation where it actually pays off.

---

## Which model do I use?

Three **task-specialized** presets ship in `fusion.yaml`, each assembled from an empirical model shoot-out (8 models × 3 task probes — coding, research, and tool-calling):

| Call this model | For | Strategy | How it is built |
|---|---|---|---|
| **`fusion-coder`** | programming, planning, code audit | `fusion` | panel `glm-5.2` + `kimi-k2.7-code` + `gemini-3-flash-preview` → judge `glm-5.2` → synth `kimi-k2.7-code` |
| **`fusion-researcher`** | research, analysis, reports | `fusion` | panel `kimi-k2.7-code` + `glm-5.2` + `gpt-oss:120b` → judge `glm-5.2` → synth `kimi-k2.7-code` |
| **`fusion-agents`** | autonomous agent loops | `smart` | router `glm-5.2`; easy steps → `gemini-3-flash-preview`, hard / error-recovery steps → the `fusion-coder` panel |

Two rules came straight out of the data:

- **Coding uses fusion, not a single model.** Architecture and planning genuinely benefit from multiple viewpoints. (A pure code *audit* — just enumerating issues — is the one coding-shaped task a single model wins, and it is not representative of programming.)
- **Panels mix model lineages on purpose.** `glm`/`kimi`/`deepseek`/`minimax`/`qwen` are all Chinese labs and share blind spots; every panel adds a Western decorrelator (`gemini` = Google, `gpt-oss` = OpenAI) so panel errors are less correlated — that is the whole point of a panel.

The original generic presets (`fusion-1`, `smart-1`, `fast-glm` / `fast-kimi` / `fast-deepseek`) still ship for ad-hoc use.

---

## Architecture

A request to `POST /v1/chat/completions` carries a **virtual model name**. The proxy resolves that name to one of four strategies and dispatches:

| Strategy | What it does |
|----------|--------------|
| `single` | 1:1 passthrough to one upstream model (stream + non-stream). The primitive everything else is built on. |
| `failover` | An ordered chain; on a pre-first-token failure it advances to the next member. Resilience without fan-out. |
| `fusion` | The deliberate pipeline: **panel → judge → synth** (below). |
| `smart` | An LLM **router** classifies each request and dispatches to `single` (cheap) or `fusion` (deep). |

### Fusion: panel → judge → synth

1. **Panel** — the request goes to N models *in parallel* (default 3). In `deliberate` tool mode the panel members get the tool *descriptions as prose* but not the real `tools` schema, so only one canonical tool call is ever emitted downstream. Optionally one member runs as an **adversarial reviewer** (red-team mandate: find the flaw, steelman the opposite case) — see `Known limitations`. Long contexts are compressed before the panel fans out so member context windows are not overflowed.
2. **Judge** — one structured-JSON call compares the panel answers (consensus, disagreements, unique insights, blind spots, hallucination_flags) and emits a **calibrated `confidence`** (`high`/`medium`/`low`) plus `fragile_claims`; the synth hedges fragile/low-confidence claims rather than laundering shared priors into false certainty. If the judge returns invalid JSON the stage degrades gracefully to the raw panel answers rather than failing.
3. **Synth** — a final model writes the answer the client receives, streams when asked, and is the **only** stage that receives the real `tools` schema (so exactly one `tool_calls` reaches the agent). Optionally followed by a **BinEval** factual-consistency score returned as headers.

### Smart: the router

`smart` makes **one** non-streamed, `temperature: 0`, JSON-only call to a fast `router` model, reads `{ "route": "simple" | "fusion" }`, and dispatches to the matching sub-strategy. Any router error, timeout, or unparseable reply falls back to the configured `default` route (default `simple`, because cost control is the whole point). The sub-strategies can be inline blocks or string references to other configured models for DRY reuse. Router decisions are cached per identical request body and in-flight identical requests are coalesced, so a burst of identical turns reuses one router round-trip. A latest tool result that looks like a failure (error / exception / non-zero exit) escalates straight to `fusion`, skipping the router — recovery is where deliberation pays.

### Optional fusion features

These are **all OFF by default**; opt in per fusion model in `fusion.yaml`. Full reference: [`fusion.example.yaml`](./fusion.example.yaml).

- **Calibrated judge confidence** — the judge emits `confidence` (`high`/`medium`/`low`) and `fragile_claims` (disputed / singly-supported / thin claims). The synth hedges fragile and low-confidence claims rather than laundering shared training priors into false certainty. The judge is **instructed** that any `hallucination_flags` or `fragile_claims` must drive `confidence` to `medium`/`low` (this is a prompt-level rule the judge model is expected to follow, not a hard code-level check — the schema accepts the judge's JSON as-is).
- **Adversarial panel member** — `adversarial: <panel-member>` makes one seat a red-team reviewer: steelman the opposite case, hunt for flaws, hidden assumptions, edge cases, and race conditions instead of agreeing with the consensus. Role-based decorrelation on top of the lineage-based one. The adversarial member is **never early-cancelled** and the panel waits for it, so its answer is never dropped; it is told to say "I cannot find a real problem" when the consensus is solid. Validated to be an existing panel member.
- **Web grounding (Tavily)** — `web_search: { enabled: true, max_results, timeout_s, max_context_chars, max_prompt_chars }`. The proxy runs **one** Tavily search before the panel fans out and injects the cleaned results as prose context into every panel member (no member receives real tools, so the one-`tool_calls` invariant holds). Gated on `TAVILY_API_KEY` in the environment *and* `web_search.enabled` on the model (no key and no opt-in → no search, no latency, no cost), **plus two runtime skips**: if the latest user message has no usable query text, or the prompt already exceeds `max_prompt_chars` (default 80 000) — the size gate prevents web context from overflowing a smaller-context panel member (e.g. gpt-oss:120b at 128k) in long agent-loop continuations. Context is injected as a `user` turn (not `system`) and prefixed with the current date, so models with a stale training cutoff (kimi-k2.7-code) use the fresh facts instead of refusing. A failed search degrades silently to an ungrounded panel. Respects `fusion_planning_turn_only`. Set the key in `.env`:
  ```bash
  echo 'TAVILY_API_KEY=tvly-...' >> .env
  ```
- **BinEval post-synth quality check** — `bineval: { enabled: true, model: <eval>, threshold: 0.7, dimensions: [...] }`. After a *non-streaming* synth that succeeds, one extra evaluator call scores the answer on factual consistency (or your custom binary questions) and returns the results as response headers: `X-Fusion-Bineval-Score` (0–1), `X-Fusion-Bineval-Dimensions` (per-dimension JSON), and `X-Fusion-Bineval-Low-Score: true` when the overall score is below `threshold`. When bineval is configured but the evaluation does not run, the proxy sets `X-Fusion-Bineval-Skipped: <reason>` so a client can tell "score is high" from "evaluation never ran" — reasons: `streaming`, `synth_error` (synth ≥400), `eval_failed` (evaluator errored/timed out/circuit open), `empty_output` (tool-only response), `non_json_body`, `synth_only` (planning-turn-only mid-loop / bypass path). **BinEval is report-only** — it does not drive re-routing or a re-deliberation loop.

---

## The honest cost note (read this)

Full fusion runs on **every** step. An agent loop (read → think → edit → run tests → re-read …) multiplies upstream **model API calls**:

```
upstream_calls_per_step = N_panel + 1 (judge) + 1 (synth) = 3 + 1 + 1 = 5
```

A typical coding task of **15–25 steps** therefore issues roughly **75–125 upstream model calls** if fusion is always on.

- **Tool executions do NOT multiply** — the tools run once per step (only the synth emits the canonical `tool_calls`). Your agent's actions are unaffected.
- **Model API calls DO multiply** — by the panel + judge + synth factor. That is the cost, and it is accepted for v1.

**Mitigation (shipped, not deferred): the `smart` strategy.** Point your agent at `fusion-agents` and routine steps (`read_file`, `grep`, …) take the cheap `single` path (1 call), while genuinely hard steps still get full fusion. Other levers baked in:

- small default panel (3),
- global `max_concurrency` cap (4),
- tight judge/panel timeouts (60 s / 90 s, both under the ~182 s upstream ceiling),
- `fusion_planning_turn_only` knob (run the full panel on every planning turn — any request whose latest message is a fresh user instruction — and degrade to synth-only — 5 calls → 1 — only on mid-loop tool-result continuations; a new task deep in a long session still gets the panel).

That is how llm-fusion keeps long agent loops affordable without sacrificing deliberation where it matters.

---

## Install

```bash
git clone https://github.com/Lexus2016/LLM-Fusion.git llm-fusion
cd llm-fusion
npm install
```

Requires **Node ≥ 24** (the proxy uses native `process.loadEnvFile`, top-level features, and runs TypeScript through `tsx` — there is no separate build step).

---

## Update

`llm-fusion` is tracked in git, not published to npm, so "upgrading" is pulling the latest and refreshing dependencies:

```bash
git pull                          # fast-forward to the newest commit / tag
npm install                       # pick up any dependency changes
```

Then restart the proxy. Your `fusion.yaml` and `.env` are yours — `git pull` never overwrites local changes to them (the shipped `fusion.yaml` carries local model presets you may have edited, so if you edited it, either commit/stash it or keep your edits in a copy referenced via `FUSION_CONFIG`). To pin a specific release instead of tracking `main`:

```bash
git fetch --tags
git checkout v0.1.12              # or any tag from the Releases page
npm install
```

The `fusion.example.yaml` is the fully annotated reference and does get updated across releases — diff it against your working config when a new version lands to see new options. Changelog: [`CHANGELOG.md`](./CHANGELOG.md), releases: <https://github.com/Lexus2016/LLM-Fusion/releases>.

---

## Configure

The proxy loads `./fusion.yaml` by default (override the path with `FUSION_CONFIG`). A complete, working config:

```yaml
upstream:
  base_url: https://ollama.com
  api_key_env: OLLAMA_API_KEY      # env var NAME holding the key — never the key itself
server:
  bind: 127.0.0.1
  port: 8080
models:
  fast-glm:                        # single — 1:1 passthrough
    strategy: single
    target: glm-5.2
  fusion-1:                        # fusion — panel -> judge -> synth
    strategy: fusion
    panel: [glm-5.2, kimi-k2.7-code, deepseek-v4-pro]
    judge: glm-5.2
    synth: deepseek-v4-pro
  smart-1:                         # smart — router picks single vs fusion
    strategy: smart
    router: glm-5.2
    default: simple
    simple: fast-glm               # string ref to the single model above
    fusion: fusion-1               # string ref to the fusion model above
```

Those are the core primitives. The **shipped `fusion.yaml`** also defines the three task-specialized presets from [Which model do I use?](#which-model-do-i-use) — `fusion-coder`, `fusion-researcher`, `fusion-agents`. Everything not specified falls back to documented defaults. For the **fully annotated reference** — every key, every default, `failover`, `tool_mode`, `fusion_planning_turn_only`, `overrides`, inline smart sub-blocks — see [`fusion.example.yaml`](./fusion.example.yaml).

Strategy cheat-sheet for the `models:` map:

- **`single`** → `{ strategy: single, target: <model> }`
- **`failover`** → `{ strategy: failover, chain: [<m1>, <m2>, …] }`
- **`fusion`** → `{ strategy: fusion, panel: [...], judge: <model>, synth: <model> }`
- **`smart`** → `{ strategy: smart, router: <model>, default: simple|fusion, simple: <single-block-or-ref>, fusion: <fusion-block-or-ref> }`

Config is **hot-reloaded**: edit `fusion.yaml` and routing/model changes apply live (an invalid edit is rejected and the previous config kept). Changing the `upstream` block — base URL, key env, concurrency — needs a restart.

---

## Environment variables

The proxy reads plain environment variables. It also **auto-loads a local `.env`** at startup (Node 24 native; an absent file is fine).

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OLLAMA_API_KEY` | Yes, for live use | — | The Ollama Cloud Bearer key. The name is whatever `upstream.api_key_env` points to (default `OLLAMA_API_KEY`). Held server-side only; never sent to clients or logged. |
| `FUSION_PROXY_TOKEN` | No | unset | When `server.auth_token_env: FUSION_PROXY_TOKEN` is set in the config, clients must send `Authorization: Bearer <this value>`. Unset ⇒ the proxy is unauthenticated (localhost single-user) and warns at startup. |
| `FUSION_CONFIG` | No | `./fusion.yaml` | Path to the config file to load. |
| `LOG_PRETTY` | No | unset | `LOG_PRETTY=1` enables human-readable pretty logs (otherwise structured JSON). |
| `LOG_LEVEL` | No | `info` | pino log level (`debug`, `info`, `warn`, …). |
| `FUSION_BIND` | No | `server.bind` | Overrides the bind address without editing the config (used by the Docker image — see below). |

A `.env.example` cannot be committed here (sandbox guard). Create your own `.env` in the project root with these literal contents:

```dotenv
# Required for any live (non-test) use:
OLLAMA_API_KEY=ollama-your-key-here

# Optional — uncomment to require a client token (also set
# server.auth_token_env: FUSION_PROXY_TOKEN in fusion.yaml):
# FUSION_PROXY_TOKEN=choose-a-long-shared-secret

# Optional overrides:
# FUSION_CONFIG=./fusion.yaml
# LOG_PRETTY=1
```

---

## Run

```bash
npm run dev      # tsx watch — restarts on source changes
npm run start    # tsx — one-shot
```

Or with an inline key (no `.env`):

```bash
OLLAMA_API_KEY=ollama-... npm run start
```

On boot it prints a banner: the listen URL, the loaded virtual models and their strategies, whether client auth is on, and whether the upstream key is present.

---

## OpenCode integration

Point OpenCode at the proxy with a provider block in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "fusion": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Fusion Proxy",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "fusion-coder": { "name": "Fusion Coder" },
        "fusion-researcher": { "name": "Fusion Researcher" },
        "fusion-agents": { "name": "Fusion Agents" },
        "fusion-1": { "name": "Fusion 1" },
        "smart-1": { "name": "Smart 1" }
      }
    }
  }
}
```

The model ids must match the virtual names in your `fusion.yaml`. If you set `FUSION_PROXY_TOKEN` (client auth), add the token for provider id `fusion` via `opencode auth login` (or your `auth.json`) so OpenCode sends it as the Bearer token.

For an autonomous agent loop, **`fusion-agents`** is the right default — it keeps routine steps cheap (router → a fast single model) and reserves the fusion panel for the hard or error-recovery steps. For a coding/planning agent use **`fusion-coder`**; for research, **`fusion-researcher`**.

### Running OpenCode with a Fusion model

With the proxy running (`npm start`) and the provider block above in your `opencode.json`
(project-local, or global `~/.config/opencode/opencode.json`), pick the model with `-m fusion/<name>`:

```bash
opencode -m fusion/fusion-coder                         # interactive TUI, coding / planning
opencode -m fusion/fusion-agents                        # interactive TUI, autonomous routing
opencode run -m fusion/fusion-researcher "Summarize X"  # headless, one prompt
opencode run -m fusion/fusion-agents "Fix the bug in Y" # headless
```

### One-command launcher (`fusion-opencode`)

`ollama` itself can't be extended, but this repo ships its own launcher so you don't have to
start the proxy or edit configs by hand. It (1) starts the proxy if it isn't already up,
(2) writes the `fusion` provider into your global OpenCode config — idempotent, it only touches
the `fusion` key — then (3) launches OpenCode with the model you name:

```bash
./bin/fusion-opencode fusion-coder             # TUI for programming / planning
./bin/fusion-opencode fusion-researcher        # TUI for research / reports
./bin/fusion-opencode fusion-agents            # TUI for autonomous agent loops
./bin/fusion-opencode fusion-agents run "Fix the failing test in src/foo.ts"   # headless one-shot
```

Put it on your `PATH` to call `fusion-opencode <model>` from any directory:

```bash
npm link            # or: ln -s "$PWD/bin/fusion-opencode" ~/.local/bin/fusion-opencode
fusion-opencode fusion-coder
```

Honors `FUSION_PROXY_URL` (default `http://127.0.0.1:8080`) and, if your proxy requires client
auth, `FUSION_PROXY_TOKEN` (used as the apiKey OpenCode sends).

### Claude Code integration

Claude Code uses the **Anthropic Messages API**. llm-fusion exposes `POST /v1/messages` on the
same base URL, so you can point Claude Code at the proxy root with a few environment variables.

#### One-command launcher (`fusion-claude`)

`bin/fusion-claude` starts the proxy (if needed), exports the Anthropic env vars, and launches
Claude Code with the model you choose:

```bash
./bin/fusion-claude fusion-agents            # autonomous agent loops (default)
./bin/fusion-claude fusion-coder             # programming / planning
./bin/fusion-claude fusion-researcher        # research / reports
./bin/fusion-claude fusion-agents run "Fix the failing test in src/foo.ts"
```

It honors `FUSION_PROXY_URL` and `FUSION_PROXY_TOKEN` exactly like `fusion-opencode`. The key
difference from OpenAI-compatible clients is the base URL: Claude Code uses the proxy **root**
(`http://127.0.0.1:8080`), not `/v1`, because it calls `/v1/messages`.

For the full setup guide, env-var reference, and troubleshooting, see
[`docs/claude-code.md`](./docs/claude-code.md).

### For AI agents (any OpenAI-compatible client)

The proxy is a drop-in OpenAI Chat Completions endpoint, so any agent framework that speaks OpenAI works — not just OpenCode (Continue, Cline, Aider with an OpenAI base URL, your own loop). Wire it with:

- **Base URL:** `http://127.0.0.1:8080/v1`
- **API key:** any non-empty string while the proxy is unauthenticated (e.g. `local-no-auth`); the real `FUSION_PROXY_TOKEN` value if you enabled client auth.
- **Model:** pick by task — `fusion-coder`, `fusion-researcher`, or `fusion-agents` (see [Which model do I use?](#which-model-do-i-use)).

The proxy emits **exactly one** `tool_calls` per step regardless of strategy, so an existing agent tool-loop runs unchanged — fusion's panel and judge happen entirely server-side and are invisible to the agent. For long autonomous runs prefer `fusion-agents`: it is `smart`, so it spends the full panel only where it pays off, falls back to a single fast model otherwise, and re-deliberates automatically when a tool result comes back as an error (`escalate_on_tool_error`). A minimal raw call:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fusion-agents","messages":[{"role":"user","content":"List the open TODOs in this repo"}]}'
```

---

## Endpoints

| Method & path | Purpose |
|---------------|---------|
| `POST /v1/chat/completions` | Main inference entrypoint. OpenAI-compatible; supports `stream`, `tools`, `tool_choice`, and image content blocks. Routed by the virtual `model` name. |
| `POST /v1/messages` | Anthropic Messages API entrypoint for Claude Code. Translates Anthropic content blocks to/from the internal OpenAI pipeline; supports streaming and tool use. |
| `GET /v1/models` | Lists the configured virtual models (OpenAI list shape). Adds `context_window` / `supports_vision` where capability discovery knows them. |
| `GET /health` | Liveness. `200` if the process is up. No upstream check. |
| `GET /ready` | Readiness. `200` only if the upstream is reachable and a representative model is discoverable; otherwise `503`. |

---

## Phase 0 — live verification

The upstream adapter's correctness depends on assumptions about Ollama Cloud (Bearer + SSE, per-model tool-calling, `/api/show` discovery, the vision format). Verify them against the real API with your key:

```bash
OLLAMA_API_KEY=ollama-... npm run smoke
```

This runs `test/live.smoke.test.ts` against `https://ollama.com`. Each check is its own test and logs a clear result:

- **A-1 — Bearer + completion** *(VERIFIED 2026-06-26)*: a non-stream chat to `glm-5.2` returns a completion (confirms the key and the openai-compat path).
- **SSE — streaming** *(VERIFIED 2026-06-26)*: the same request with `stream: true` returns SSE `data:` chunks.
- **A-2 — tool-calling per model** *(VERIFIED 2026-06-26 — `tool_calls` on all of `glm-5.2`, `kimi-k2.7-code`, `deepseek-v4-pro`)*: a `tools` request to each of `glm-5.2`, `kimi-k2.7-code`, `deepseek-v4-pro`; logs **PASS/FAIL** per model (does *not* fail the suite if a model lacks tools — but the synth must be a PASS model, since the whole deliberate flow depends on it emitting `tool_calls`).
- **A-3 — `/api/show`** *(VERIFIED 2026-06-26 — `capabilities[]` + `*.context_length` returned for all three)*: confirms `capabilities[]` and a `*.context_length` come back for those models, and logs what it found.
- **A-4/A-5 — vision** *(STILL OPEN — re-verify against `kimi-k2.7-code`; the prior test model `qwen3-vl:235b` was retired 2026-06-16)*: POSTs a 1×1 PNG `image_url` to a vision model (`kimi-k2.7-code` by default; override with `FUSION_VISION_MODEL`) and logs whether the OpenAI image format is accepted.

It prints a summary you use to finalize two settings in `fusion.yaml`: `upstream.api_mode` (`openai` if the vision check accepted the OpenAI format, else `native`) and which model is safe to use as a fusion `synth` (one whose A-2 row is PASS).

Without `OLLAMA_API_KEY` the suite is **skipped**, so it never runs in CI or the default test run.

---

## Testing

```bash
npm test          # vitest run — fast, offline, no key, no network (live smoke is skipped)
npm run typecheck # tsc --noEmit
```

The unit/integration suite uses a mock upstream (intercepted `fetch`); it covers config validation, routing, capability parsing, every strategy, the tool gate, the vision gate, and smart routing. The live smoke test is the only one that touches the network, and only with a key.

---

## Known limitations / not yet wired

- **Native NDJSON vision streaming is deferred.** Vision works on the openai-compat path; a streaming image request that resolves to the native `/api/chat` backend (`api_mode: native` + images + `stream: true`) returns a clean `501` rather than a half-wired stream. Non-stream native vision works.
- **No round-robin, no semantic cache, no context-size routing** yet — these are Phase 6 / future. (Smart-router decisions *are* cached per identical request body and in-flight identical requests are coalesced — see `smart` above — but there is no semantic/embedding cache and no routing by prompt token count.)
- **BinEval is report-only.** A low `X-Fusion-Bineval-Score` does not trigger a re-deliberation or a re-synth loop — the proxy surfaces the score (or `X-Fusion-Bineval-Skipped`) and the client decides. An optional auto-retry was considered and deliberately not wired (cost amplification on the expensive turns, LLM-as-judge false negatives, and eval-model drift as a silent regression amplifier).
- **Loopback only.** Binds `127.0.0.1` by default, no TLS, single-tenant (optional shared client token, not per-user auth). Not built for multi-tenant exposure.
- **Failover + streaming**: the chain can advance only *before* the first token; a mid-stream upstream failure surfaces as a stream error (cannot silently re-roll a partially sent response).
