# AGENTS.md — orientation for AI agents working in (or with) llm-fusion

This file is the quick brief an AI coding agent should read first. Humans: see [`README.md`](./README.md).

## What this is

`llm-fusion` (**Fusion Proxy**) is an OpenAI-compatible HTTP proxy in front of **Ollama Cloud**. One virtual model name resolves to a multi-model pipeline: `single`, `failover`, `fusion` (panel → judge → synth), or `smart` (an LLM router picks cheap vs deep per request). Single process, Node 24 + TypeScript + Hono, **no build step** (runs `.ts` via `tsx`), config in one YAML file.

## Run it

```bash
npm install                                     # Node ≥ 24
printf 'OLLAMA_API_KEY=ollama-your-key\n' > .env # Ollama Cloud key (auto-loaded at boot)
npm start                                       # proxy on http://127.0.0.1:8080
npm run dev                                      # same, with tsx watch (restart on change)
```

## Use it (which model to call)

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`, API key `local-no-auth` (or the `FUSION_PROXY_TOKEN` value if client auth is on), and pick the model by task:

| Model | Use for |
|---|---|
| `fusion-coder` | programming, planning, code audit |
| `fusion-researcher` | research, analysis, reports |
| `fusion-agents` | autonomous agent loops (smart routing; **default for long runs**) |

The proxy emits **exactly one** `tool_calls` per step in every strategy, so an existing tool-loop runs unchanged — the panel/judge are server-side and invisible. Model lineups were chosen empirically (an 8-model × 3-probe shoot-out across coding, research, and tool-calling).

OpenCode shortcut: `./bin/fusion-opencode fusion-coder` (starts the proxy + wires the provider + opens the TUI).

## Develop it (conventions)

- **Verify before claiming done.** `npm test` (vitest, offline mock upstream — no key, no network) and `npm run typecheck` (`tsc --noEmit`) must both pass.
- **Live checks** that hit the real API run only with a key: `npm run smoke`.
- **No build step / no `dist`.** Edit `src/*.ts`; `tsx` runs them directly. Do not add a compile step.
- **No typecasting.** No `as` in TypeScript — fix the types at the source.
- **Config is hot-reloaded** from `fusion.yaml`; an invalid edit is rejected and the previous config kept. Changing the `upstream:` block needs a restart. Full annotated reference: [`fusion.example.yaml`](./fusion.example.yaml).
- **Secrets:** the Ollama key lives in `.env` / the `OLLAMA_API_KEY` env var only. Never inline it in code, config, logs, or commits.

## Layout

- `src/index.ts` — entrypoint / HTTP server (Hono).
- `src/config.ts` — YAML load + zod schema + hot-reload watcher.
- `src/strategies/` — `single`, `failover`, `fusion`, `smart`.
- `src/usage.ts`, `src/attribution.ts` — upstream usage/cost accounting + per-call error attribution.
- `src/capabilities.ts` — `/api/show` capability discovery (vision/tools/context).
- `test/` — vitest suite (mock upstream) + `live.smoke.test.ts` (key-gated).
