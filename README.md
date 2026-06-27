# llm-fusion — Fusion Proxy

An OpenAI-compatible HTTP proxy in front of [Ollama Cloud](https://ollama.com) that turns one virtual model name into a **multi-model deliberation pipeline**. Point any OpenAI-compatible client (OpenCode, Continue, your own scripts) at it, ask for the model `fusion-coder`, and behind the scenes a panel of models answers, a judge cross-checks them, and a synthesizer writes the final answer — or, with `smart`, an LLM router decides per request whether that heavy treatment is even warranted. It is a small, single-process Node 24 + TypeScript + [Hono](https://hono.dev) service: no database, no build step (runs `.ts` directly via `tsx`), config in one YAML file.

Three **task-specialized** presets ship ready to use — `fusion-coder`, `fusion-researcher`, `fusion-agents` — each assembled from an empirical model shoot-out (jump to [Which model do I use?](#which-model-do-i-use)).

---

## Quickstart

```bash
npm install                                        # Node ≥ 24
printf 'OLLAMA_API_KEY=ollama-your-key\n' > .env    # your Ollama Cloud key
npm start                                          # proxy on http://127.0.0.1:8080
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1` and ask for a model by name — or launch OpenCode in a single command:

```bash
./bin/fusion-opencode fusion-coder                 # starts the proxy if needed, wires OpenCode, opens the TUI
```

New here? Read **[Which model do I use?](#which-model-do-i-use)** next.

---

## Which model do I use?

Three **task-specialized** presets ship in `fusion.yaml`, each assembled from an empirical model shoot-out (8 models × 3 task probes — coding, research, and tool-calling):

| Call this model | For | Strategy | How it is built |
|---|---|---|---|
| **`fusion-coder`** | programming, planning, code audit | `fusion` | panel `glm-5.2` + `kimi-k2.7-code` + `gemini-3-flash-preview` → judge `glm-5.2` → synth `kimi-k2.7-code` |
| **`fusion-researcher`** | research, analysis, reports | `fusion` | panel `kimi-k2.7-code` + `glm-5.2` + `gpt-oss:120b` → judge `glm-5.2` → synth `kimi-k2.7-code` |
| **`fusion-agents`** | autonomous agent loops | `smart` | router `glm-5.2`; easy steps → `gemini-3-flash-preview`, hard / error-recovery steps → the `fusion-coder` panel |

Two rules came straight out of the data:

- **Coding uses fusion, not a single model.** Architecture and planning genuinely benefit from multiple viewpoints. (A code *audit* — pure enumeration — is the one coding-shaped task a single model wins, and it is not representative of programming.)
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

1. **Panel** — the request goes to N models *in parallel* (default 3). In `deliberate` tool mode the panel members get the tool *descriptions as prose* but not the real `tools` schema, so only one canonical tool call is ever emitted downstream.
2. **Judge** — one structured-JSON call compares the panel answers (consensus, disagreements, unique insights, blind spots). If the judge returns invalid JSON the stage degrades gracefully to the raw panel answers rather than failing.
3. **Synth** — a final model writes the answer the client receives, streams when asked, and is the **only** stage that receives the real `tools` schema (so exactly one `tool_calls` reaches the agent).

### Smart: the router

`smart` makes **one** non-streamed, `temperature: 0`, JSON-only call to a fast `router` model, reads `{ "route": "simple" | "fusion" }`, and dispatches to the matching sub-strategy. Any router error, timeout, or unparseable reply falls back to the configured `default` route (default `simple`, because cost control is the whole point). The sub-strategies can be inline blocks or string references to other configured models for DRY reuse.

---

## The honest cost note (read this)

Fusion runs on **every** step. An agent loop (read → think → edit → run tests → re-read …) multiplies upstream **model API calls**:

```
upstream_calls_per_step = N_panel + 1 (judge) + 1 (synth) = 3 + 1 + 1 = 5
```

A typical coding task of **15–25 steps** therefore issues roughly **75–125 upstream model calls**.

- **Tool executions do NOT multiply** — the tools run once per step (only the synth emits the canonical `tool_calls`). Your agent's actions are unaffected.
- **Model API calls DO multiply** — by the panel + judge + synth factor. That is the cost, and it is accepted for v1.

**Mitigation (shipped, not deferred): the `smart` strategy.** Point your agent at a `smart` model and routine steps (`read_file`, `grep`, …) take the cheap `single` path (1 call), while genuinely hard steps still get full fusion. Other levers baked in: a small default panel (3), a global `max_concurrency` cap (4), tight judge/panel timeouts (60 s / 90 s, both under the ~182 s upstream ceiling), and the `fusion_planning_turn_only` knob (run fusion only on the planning turn, degrade to synth-only — 5 calls → 1 — once the conversation already has tool messages).

---

## Install

```bash
npm install
```

Requires **Node ≥ 24** (the proxy uses native `process.loadEnvFile`, top-level features, and runs TypeScript through `tsx` — there is no separate build step).

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
- **No round-robin, no semantic cache, no context-size routing** yet — these are Phase 6 / future.
- **Loopback only.** Binds `127.0.0.1` by default, no TLS, single-tenant (optional shared client token, not per-user auth). Not built for multi-tenant exposure.
- **Failover + streaming**: the chain can advance only *before* the first token; a mid-stream upstream failure surfaces as a stream error (cannot silently re-roll a partially sent response).
