# llm-fusion — Fusion Proxy

> **English** | [Русский](./README.ru.md) | [Українська](./README.ua.md)

> **Three models argue over your prompt, a judge scores their answers, and one final model writes the reply you actually see.**

llm-fusion is a small proxy you run on your own machine, in front of [Ollama Cloud](https://ollama.com). Your coding tool connects to it the way it would connect to any single model — same API, same streaming, same tool calls. The difference is hidden behind the model name: ask for `fusion-coder` and you get a panel of three models answering in parallel, a judge comparing what they said, and a synthesizer writing the final answer. Answers to hard questions come out noticeably stronger than what any one of those models produces alone.

Running that panel on every request would be expensive, so there is also `fusion-agents`: a fast router looks at each request and decides whether it deserves the full panel or just one cheap call. Reading a file gets one call. Recovering from a failed test run gets the whole panel.

No database, no build step, no accounts. Node 24, one YAML config file, and your Ollama Cloud key. It speaks both the OpenAI Chat Completions API and the Anthropic Messages API, so OpenCode, Claude Code, Continue, Cline, Aider — or your own agent loop — all work unchanged.

Familiar with **OpenRouter Fusion**? Same idea — *many models think, one answers* — but self-hosted, transparent, and built to survive long agent loops. Full comparison [below](#llm-fusion-vs-openrouter-fusion).

---

## Quick start — 5 minutes

You need two things: **Node.js 24 or newer** and an **[Ollama Cloud](https://ollama.com) API key**.

```bash
# 1. Get the code
git clone https://github.com/Lexus2016/LLM-Fusion.git llm-fusion
cd llm-fusion

# 2. Install dependencies (no build step — TypeScript runs directly via tsx)
npm install

# 3. Put your Ollama Cloud key into .env
printf 'OLLAMA_API_KEY=ollama-your-key\n' > .env

# 4. Start the proxy
npm start
```

That's it — the proxy is listening on `http://127.0.0.1:8080`. Check it works:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fusion-coder","messages":[{"role":"user","content":"Say hello"}]}'
```

### Step 5 — pick a model by task

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1` (API key: any non-empty string, e.g. `local-no-auth`) and ask for one of the three shipped presets:

| Ask for | When you are… |
|---|---|
| **`fusion-coder`** | writing code, planning a feature, auditing a change |
| **`fusion-researcher`** | researching a topic, analyzing data, writing a report |
| **`fusion-agents`** | running an autonomous agent loop for many steps |

### Or skip all of the above — one command

The bundled launchers start the proxy if it isn't running, write the client config for you, and open your tool:

```bash
./bin/fusion-opencode fusion-coder     # OpenCode TUI, wired and ready
./bin/fusion-claude fusion-agents      # Claude Code, wired and ready
```

---

## Use it with your tool

### OpenCode

The fastest path is the launcher above. To wire it manually instead, add a provider block to `opencode.json` (project-local, or global `~/.config/opencode/opencode.json`):

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
        "fusion-agents": { "name": "Fusion Agents" }
      }
    }
  }
}
```

The model ids must match the virtual names in your `fusion.yaml`. If you set `FUSION_PROXY_TOKEN` (client auth), add the token for provider id `fusion` via `opencode auth login` (or your `auth.json`) so OpenCode sends it as the Bearer token.

With the proxy running, pick the model with `-m fusion/<name>`:

```bash
opencode -m fusion/fusion-coder                         # interactive TUI, coding / planning
opencode -m fusion/fusion-agents                        # interactive TUI, autonomous routing
opencode run -m fusion/fusion-researcher "Summarize X"  # headless, one prompt
```

The `fusion-opencode` launcher does all three steps — starts the proxy if needed, writes the `fusion` provider into your global OpenCode config (idempotent, it only touches the `fusion` key), and launches OpenCode:

```bash
./bin/fusion-opencode fusion-researcher        # TUI for research / reports
./bin/fusion-opencode fusion-agents run "Fix the failing test in src/foo.ts"   # headless one-shot
```

Put it on your `PATH` to call it from any directory:

```bash
npm link            # or: ln -s "$PWD/bin/fusion-opencode" ~/.local/bin/fusion-opencode
fusion-opencode fusion-coder
```

It honors `FUSION_PROXY_URL` (default `http://127.0.0.1:8080`) and, if your proxy requires client auth, `FUSION_PROXY_TOKEN` (used as the apiKey OpenCode sends).

### Claude Code

Claude Code uses the **Anthropic Messages API**. llm-fusion exposes `POST /v1/messages` on the same base URL, so `bin/fusion-claude` starts the proxy (if needed), exports the Anthropic env vars, and launches Claude Code with the model you choose:

```bash
./bin/fusion-claude fusion-agents            # autonomous agent loops (default)
./bin/fusion-claude fusion-coder             # programming / planning
./bin/fusion-claude fusion-researcher        # research / reports
./bin/fusion-claude fusion-agents run "Fix the failing test in src/foo.ts"
```

It honors `FUSION_PROXY_URL` and `FUSION_PROXY_TOKEN` exactly like `fusion-opencode`. One difference from OpenAI-compatible clients: Claude Code uses the proxy **root** (`http://127.0.0.1:8080`), not `/v1`, because it calls `/v1/messages`.

Full setup guide, env-var reference, and troubleshooting: [`docs/claude-code.md`](./docs/claude-code.md).

### Any other OpenAI-compatible client

The proxy is a drop-in OpenAI Chat Completions endpoint, so anything that speaks OpenAI works — Continue, Cline, Aider with an OpenAI base URL, or your own loop:

- **Base URL:** `http://127.0.0.1:8080/v1`
- **API key:** any non-empty string while the proxy is unauthenticated (e.g. `local-no-auth`); the real `FUSION_PROXY_TOKEN` value if you enabled client auth.
- **Model:** `fusion-coder`, `fusion-researcher`, or `fusion-agents` — see [Which model do I use?](#which-model-do-i-use)

---

## Updating

`llm-fusion` is tracked in git, not published to npm, so "upgrading" is pulling the latest and refreshing dependencies:

```bash
git pull                          # fast-forward to the newest commit / tag
npm install                       # pick up any dependency changes
```

Then restart the proxy. Your `fusion.yaml` and `.env` are yours — `git pull` never overwrites local changes to them (the shipped `fusion.yaml` carries local model presets you may have edited, so if you edited it, either commit/stash it or keep your edits in a copy referenced via `FUSION_CONFIG`). To pin a specific release instead of tracking `main`:

```bash
git fetch --tags
git checkout v0.1.27              # or any tag from the Releases page
npm install
```

The `fusion.example.yaml` is the fully annotated reference and does get updated across releases — diff it against your working config when a new version lands to see new options. Changelog: [`CHANGELOG.md`](./CHANGELOG.md), releases: <https://github.com/Lexus2016/LLM-Fusion/releases>.

---

# Technical reference

Everything below is detail: how the pipeline works, what it costs, every config key, every endpoint. You don't need any of it to get started — the quick start above is the whole setup.

## Which model do I use?

Three **task-specialized** presets ship in `fusion.yaml`, each assembled from an empirical model shoot-out (8 models × 3 task probes — coding, research, and tool-calling):

| Call this model | For | Strategy | How it is built |
|---|---|---|---|
| **`fusion-coder`** | programming, planning, code audit | `fusion` | panel `glm-5.2` + `kimi-k2.7-code` + `mistral-large-3:675b` → judge `glm-5.2` → synth `glm-5.2` |
| **`fusion-researcher`** | research, analysis, reports | `fusion` | panel `kimi-k2.7-code` + `glm-5.2` + `gpt-oss:120b` → judge `glm-5.2` → synth `kimi-k2.7-code` |
| **`fusion-agents`** | autonomous agent loops | `smart` | router `glm-5.2`; easy steps → `glm-5.2`, hard / error-recovery steps → the `fusion-coder` panel |

Two rules came straight out of the data:

- **Coding uses fusion, not a single model.** Architecture and planning genuinely benefit from multiple viewpoints. (A pure code *audit* — just enumerating issues — is the one coding-shaped task a single model wins, and it is not representative of programming.)
- **Panels mix model lineages on purpose.** `glm`/`kimi`/`deepseek`/`minimax`/`qwen` are all Chinese labs and share blind spots; every panel adds a Western decorrelator — `mistral-large-3:675b` (Mistral) on the coder panel (since v0.1.26+: it also lifts the advertised context window from 131K to 262K), `gpt-oss:120b` (OpenAI lineage) on the researcher panel — so panel errors are less correlated; that is the whole point of a panel. (Gemini was the original decorrelator but rejects tool-call history produced by other models with a `thought_signature` error on every mid-loop step, so it was replaced.)

The original generic presets (`fusion-1`, `smart-1`, `fast-glm` / `fast-kimi` / `fast-deepseek`) still ship for ad-hoc use.

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
2. **Judge** — one structured-JSON call compares the panel answers (consensus, disagreements, unique insights, blind spots, partial coverage, hallucination_flags) and emits a **calibrated `confidence`** (`high`/`medium`/`low`) plus `fragile_claims`; the synth hedges fragile/low-confidence claims rather than laundering shared priors into false certainty. If the judge returns invalid JSON the stage degrades gracefully to the raw panel answers rather than failing.
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

### Agent-loop reliability (v0.1.16+)

Long agent runs used to die on four load-dependent failure modes; all four are now handled structurally:

- **Synth completeness guard with a judge fallback.** A "thinking" synth can declare itself done while still mid-plan (empty answer, or one inline `<think>` block with no artifact). The guard detects that on both the streaming and non-streaming paths, retries the same synth once with a strict completion nudge, and — if that retry is still unusable (or was itself cut by the token cap mid tool call) — makes ONE attempt on a different model (the judge — or a panel member when the judge IS the synth), so no single model can stall an agent loop. During the silent recovery the stream carries SSE `: keepalive` comments (every 5 s, `FUSION_SYNTH_RECOVERY_PING_MS` to tune) so clients and intermediaries don't time out.
- **Honest `stop_reason` on token-cap cuts.** A tool call truncated by `max_tokens` is reported to Anthropic clients as `stop_reason: "max_tokens"` — not as a runnable `tool_use` with broken JSON input. When the cap lands exactly *after* a tool call's JSON completed, the call stays runnable (`tool_use`), so a finished multi-minute Write is never thrown away.
- **Per-model concurrency budgets.** `upstream.per_model_concurrency` gives each real upstream model its own gate in front of the global limiter — a burst of background small-model calls (Claude Code fires 80–130/min) queues at its own gate instead of head-of-line blocking interactive turns.
- **Separated traffic classes.** The `fusion-claude` launcher defaults the Claude Code background model (`ANTHROPIC_SMALL_FAST_MODEL`) to `fast-deepseek` — a model no panel, judge, synth, router, or simple route depends on — so background bursts can't rate-limit the model that writes your files.
- **Single-route tool-turn guard (v0.1.26).** The smart `simple` passthrough — the route ~87 % of agent steps take — gets its own completeness guard. Three field-validated failure modes used to end the agent's turn with nothing executed (the "does one step, then stops until you type *continue*" stall): the model *narrating* the next action in prose instead of emitting the tool call; a large-file write cut by the output cap mid-arguments (`finish_reason:"length"`, unrunnable JSON); and the upstream terminating long generation streams (~5 min on Ollama Cloud). The guard detects all three on tool-carrying requests and runs one **live-streamed** recovery retry (nudged to emit the tool call and write large payloads in chunks), failing open to the original response. Every tool-carrying stream also logs one `tool-turn terminal state` line, so a real-session stall is diagnosable from the log alone.
- **Per-model `request_overrides` (v0.1.26).** Single models (and the smart inline `simple` slot) accept extra request-body fields merged into every upstream call — e.g. `request_overrides: { reasoning_effort: "none" }` stops a thinking model from deliberating for minutes on mechanical agent steps (A/B-measured on glm-5.2: reasoning 1692→0 chars, 6 s→2 s, tool-calling intact; `think:false` and `"low"` are ignored by Ollama Cloud). Core keys (`model`, `messages`, `stream`, `tools`, `tool_choice`) are protected. The shipped `fusion-agents` preset uses exactly this.

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
- global `max_concurrency` cap plus **per-model concurrency budgets** (`per_model_concurrency`) — a burst on one model queues at its own gate instead of head-of-line blocking interactive turns,
- tight panel/judge timeouts (90 s / 120 s in the shipped presets, both under the ~182 s upstream ceiling),
- `fusion_planning_turn_only` knob (run the full panel on every planning turn — any request whose latest message is a fresh user instruction — and degrade to synth-only — 5 calls → 1 — only on mid-loop tool-result continuations; a new task deep in a long session still gets the panel).

That is how llm-fusion keeps long agent loops affordable without sacrificing deliberation where it matters.

## llm-fusion vs OpenRouter Fusion

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

## Configuration

The proxy loads `./fusion.yaml` by default (override the path with `FUSION_CONFIG`). This is a trimmed copy of the shipped config — the three task-specialized presets plus one single model:

```yaml
upstream:
  base_url: https://ollama.com
  api_key_env: OLLAMA_API_KEY        # env var NAME holding the key — never the key itself
server:
  bind: 127.0.0.1
  port: 8080
models:
  # single — 1:1 passthrough; call this when you want one fast model, no panel
  fast-glm:
    strategy: single
    target: glm-5.2

  # fusion — panel -> judge -> synth; programming, planning, code audit
  fusion-coder:
    strategy: fusion
    panel: [glm-5.2, kimi-k2.7-code, mistral-large-3:675b]
    judge: glm-5.2
    synth: glm-5.2
    tool_mode: deliberate

  # fusion — panel tuned for research and reports, optional web grounding
  fusion-researcher:
    strategy: fusion
    panel: [kimi-k2.7-code, glm-5.2, gpt-oss:120b]
    judge: glm-5.2
    synth: kimi-k2.7-code
    tool_mode: deliberate
    web_search:
      enabled: true                  # also needs TAVILY_API_KEY in .env, otherwise fully OFF
      max_results: 3

  # smart — a router picks cheap single vs full fusion per request; agent loops
  fusion-agents:
    strategy: smart
    router: glm-5.2
    default: simple
    escalate_on_tool_error: true     # a failing tool result goes straight to fusion
    simple:
      target: glm-5.2                # mechanical steps (read / grep / run)
      request_overrides:
        reasoning_effort: "none"     # no minutes-long deliberation on mechanical steps
    fusion: fusion-coder             # hard steps -> the coder panel (string ref)
```

Call any of these by putting the name in the `model` field of your request — `fast-glm` answers with one model and no panel. The full shipped [`fusion.yaml`](./fusion.yaml) additionally carries `fast-kimi` / `fast-deepseek` / `fusion-1` / `smart-1` for ad-hoc use, tuned timeouts, per-model concurrency budgets, and the BinEval blocks. Everything not specified falls back to documented defaults. For the **fully annotated reference** — every key, every default, `failover`, `tool_mode`, `fusion_planning_turn_only`, `overrides`, inline smart sub-blocks — see [`fusion.example.yaml`](./fusion.example.yaml).

Strategy cheat-sheet for the `models:` map:

- **`single`** → `{ strategy: single, target: <model> }`
- **`failover`** → `{ strategy: failover, chain: [<m1>, <m2>, …] }`
- **`fusion`** → `{ strategy: fusion, panel: [...], judge: <model>, synth: <model> }`
- **`smart`** → `{ strategy: smart, router: <model>, default: simple|fusion, simple: <single-block-or-ref>, fusion: <fusion-block-or-ref> }`

Config is **hot-reloaded**: edit `fusion.yaml` and routing/model changes apply live (an invalid edit is rejected and the previous config kept). Changing the `upstream` block — base URL, key env, concurrency — needs a restart.

## Environment variables

The proxy reads plain environment variables. It also **auto-loads a local `.env`** at startup (Node 24 native; an absent file is fine).

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OLLAMA_API_KEY` | Yes, for live use | — | The Ollama Cloud Bearer key. The name is whatever `upstream.api_key_env` points to (default `OLLAMA_API_KEY`). Held server-side only; never sent to clients or logged. |
| `FUSION_PROXY_TOKEN` | No | unset | When `server.auth_token_env: FUSION_PROXY_TOKEN` is set in the config, clients must send `Authorization: Bearer <this value>`. Unset ⇒ the proxy is unauthenticated (localhost single-user) and warns at startup. |
| `FUSION_CONFIG` | No | `./fusion.yaml` | Path to the config file to load. |
| `LOG_PRETTY` | No | unset | `LOG_PRETTY=1` enables human-readable pretty logs (otherwise structured JSON). |
| `LOG_LEVEL` | No | `info` | pino log level (`debug`, `info`, `warn`, …). |
| `FUSION_BIND` | No | `server.bind` | Overrides the bind address without editing the config (used by the Docker image). |
| `FUSION_SYNTH_RECOVERY_PING_MS` | No | `5000` | Interval of the SSE `: keepalive` comments emitted while a synth recovery retry runs inside a stream. Positive number; anything else falls back to the default. |

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

## Running

```bash
npm run dev      # tsx watch — restarts on source changes
npm run start    # tsx — one-shot
```

Or with an inline key (no `.env`):

```bash
OLLAMA_API_KEY=ollama-... npm run start
```

On boot it prints a banner: the listen URL, the loaded virtual models and their strategies, whether client auth is on, and whether the upstream key is present.

## Endpoints

| Method & path | Purpose |
|---------------|---------|
| `POST /v1/chat/completions` | Main inference entrypoint. OpenAI-compatible; supports `stream`, `tools`, `tool_choice`, and image content blocks. Routed by the virtual `model` name. |
| `POST /v1/messages` | Anthropic Messages API entrypoint for Claude Code. Translates Anthropic content blocks to/from the internal OpenAI pipeline; supports streaming and tool use. |
| `GET /v1/models` | Lists the configured virtual models (OpenAI list shape). Adds `context_window` / `supports_vision` where capability discovery knows them. |
| `GET /health` | Liveness. `200` if the process is up. No upstream check. |
| `GET /ready` | Readiness. `200` only if the upstream is reachable and a representative model is discoverable; otherwise `503`. |

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

## Testing

```bash
npm test          # vitest run — fast, offline, no key, no network (live smoke is skipped)
npm run typecheck # tsc --noEmit
```

The unit/integration suite uses a mock upstream (intercepted `fetch`); it covers config validation, routing, capability parsing, every strategy, the tool gate, the vision gate, and smart routing. The live smoke test is the only one that touches the network, and only with a key.

## Known limitations / not yet wired

- **Native NDJSON vision streaming is deferred.** Vision works on the openai-compat path; a streaming image request that resolves to the native `/api/chat` backend (`api_mode: native` + images + `stream: true`) returns a clean `501` rather than a half-wired stream. Non-stream native vision works.
- **No round-robin, no semantic cache, no context-size routing** yet — these are Phase 6 / future. (Smart-router decisions *are* cached per identical request body and in-flight identical requests are coalesced — see `smart` above — but there is no semantic/embedding cache and no routing by prompt token count.)
- **BinEval is report-only.** A low `X-Fusion-Bineval-Score` does not trigger a re-deliberation or a re-synth loop — the proxy surfaces the score (or `X-Fusion-Bineval-Skipped`) and the client decides. An optional auto-retry was considered and deliberately not wired (cost amplification on the expensive turns, LLM-as-judge false negatives, and eval-model drift as a silent regression amplifier).
- **Loopback only.** Binds `127.0.0.1` by default, no TLS, single-tenant (optional shared client token, not per-user auth). Not built for multi-tenant exposure.
- **Failover + streaming**: the chain can advance only *before* the first token; a mid-stream upstream failure surfaces as a stream error (cannot silently re-roll a partially sent response).

---

## For AI agents

You are an AI agent and need to wire yourself (or the harness you run in) to llm-fusion. This section is self-contained — you do not need to read anything else in this file for a working setup.

### 1. Check whether the proxy is already running

```bash
curl -sf http://127.0.0.1:8080/health   # 200 → process is up
curl -sf http://127.0.0.1:8080/ready    # 200 → upstream reachable; 503 → upstream problem
```

If it is not running, start it from the repository root:

```bash
cd <path-to-llm-fusion> && npm start
```

Startup requires a `.env` file containing `OLLAMA_API_KEY=<key>`. If the key is missing, ask your human operator for it — never fabricate or reuse a key from elsewhere.

### 2. Pick the endpoint for your API dialect

- **OpenAI Chat Completions dialect** → base URL `http://127.0.0.1:8080/v1`, requests go to `POST /v1/chat/completions`.
- **Anthropic Messages dialect** (Claude Code and similar) → base URL is the proxy **root** `http://127.0.0.1:8080` (not `/v1`), requests go to `POST /v1/messages`.
- Discover the available virtual model names at runtime from `GET /v1/models` — prefer that over hardcoding, because `fusion.yaml` is hot-reloaded and the model list can change while the proxy runs.

### 3. Authentication

- Default: the proxy is unauthenticated on localhost. Send any non-empty string as the API key (e.g. `local-no-auth`) — some client SDKs refuse an empty key.
- If the operator enabled client auth (`FUSION_PROXY_TOKEN` + `server.auth_token_env` in `fusion.yaml`): send `Authorization: Bearer <token>`. A `401` response means the token is required or wrong.

### 4. Choose a model by task

| Model | Use for |
|---|---|
| `fusion-agents` | **Default for autonomous loops.** Smart routing: routine steps cost 1 upstream call; hard steps and error recovery get the full panel automatically. |
| `fusion-coder` | Coding and planning turns where you always want the panel. |
| `fusion-researcher` | Research, analysis, and report writing. |

### 5. Tool-calling contract (what you can rely on)

- The proxy emits **exactly one `tool_calls` per step** regardless of strategy — panel and judge run entirely server-side and never see your real `tools` schema, so your existing tool loop runs unchanged and side effects stay exactly-once.
- When your latest tool result looks like a failure (error / exception / non-zero exit), `fusion-agents` escalates that turn straight to full fusion — you do not need to ask for deeper reasoning yourself.
- A tool call truncated by the token cap is reported honestly (`stop_reason: "max_tokens"` on the Anthropic dialect), never as runnable-looking broken JSON.

### 6. Ready-made launchers (instead of manual wiring)

```bash
./bin/fusion-opencode <model>    # starts proxy if needed, writes OpenCode provider config, opens OpenCode
./bin/fusion-claude <model>      # starts proxy if needed, exports ANTHROPIC_* env vars, opens Claude Code
```

Both honor `FUSION_PROXY_URL` (default `http://127.0.0.1:8080`) and `FUSION_PROXY_TOKEN`. Both accept `run "<prompt>"` after the model name for a headless one-shot.

### 7. Minimal smoke probe (copy-paste)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fusion-agents","messages":[{"role":"user","content":"Reply with the single word: ok"}]}'
```

A JSON completion back means you are fully configured. Detailed Claude Code wiring lives in [`docs/claude-code.md`](./docs/claude-code.md); every config key is documented in [`fusion.example.yaml`](./fusion.example.yaml).
