# Using Claude Code with llm-fusion

[Claude Code](https://code.claude.com) speaks the **Anthropic Messages API**, not the OpenAI Chat Completions API. llm-fusion therefore exposes `POST /v1/messages` on the same base URL as the OpenAI-compatible `POST /v1/chat/completions`. You can point Claude Code directly at the proxy with a few environment variables.

## Quick start

With the proxy installed and your Ollama Cloud key in `.env`:

```bash
npm install
printf 'OLLAMA_API_KEY=ollama-your-key\n' > .env
./bin/fusion-claude fusion-agents
```

`fusion-claude` will:

1. Start the proxy if it is not already answering on `/health`.
2. Export the Anthropic env vars Claude Code needs.
3. Launch `claude` in the current directory.

Use the same task-specialized models as OpenCode:

```bash
./bin/fusion-claude fusion-coder      # programming / planning / audit
./bin/fusion-claude fusion-researcher # research / analysis / reports
./bin/fusion-claude fusion-agents     # autonomous agent loops (recommended default)
```

You can pass any extra arguments to Claude Code itself:

```bash
./bin/fusion-claude fusion-agents run "Fix the failing test in src/foo.ts"
```

## Manual setup

If you prefer not to use the launcher, set these environment variables and then run `claude`:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080      # proxy root — do NOT add /v1
export ANTHROPIC_MODEL=fusion-agents                  # virtual model from fusion.yaml
export ANTHROPIC_SMALL_FAST_MODEL=fast-kimi           # cheap model for quick tasks
export ANTHROPIC_AUTH_TOKEN=local-no-auth             # or your FUSION_PROXY_TOKEN
claude
```

### Why `ANTHROPIC_BASE_URL` has no `/v1`

Claude Code appends `/v1/messages` to the base URL. llm-fusion serves the Anthropic endpoint at exactly that path, so the base URL must be the proxy root:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:8080
                         └── Claude Code calls http://127.0.0.1:8080/v1/messages
```

OpenAI-compatible clients, by contrast, use `http://127.0.0.1:8080/v1` because they call `/v1/chat/completions`.

## Authentication

If your `fusion.yaml` sets `server.auth_token_env: FUSION_PROXY_TOKEN`, clients must authenticate. For Claude Code this is done via `ANTHROPIC_AUTH_TOKEN` (the Anthropic SDK sends it as the `x-api-key` header):

```bash
export FUSION_PROXY_TOKEN=choose-a-long-shared-secret
./bin/fusion-claude fusion-agents
```

The launcher reads `FUSION_PROXY_TOKEN` and passes it through. If you set `ANTHROPIC_AUTH_TOKEN` yourself, that value is used instead.

## Model mapping

The value of `ANTHROPIC_MODEL` must be a **virtual model name** defined in `fusion.yaml`. The presets that ship with the repo are:

| `ANTHROPIC_MODEL` | Use for | Strategy |
|---|---|---|
| `fusion-coder` | Programming, planning, code audit | `fusion` |
| `fusion-researcher` | Research, analysis, reports | `fusion` |
| `fusion-agents` | Autonomous agent loops (default) | `smart` |

`ANTHROPIC_SMALL_FAST_MODEL` should point to a cheap `single` model, e.g. `fast-kimi`, `fast-glm`, or `fast-deepseek`. Claude Code uses this for lightweight tasks like summarizing tool output or quick lookups. The shipped `fusion.yaml` already defines those fast singles.

## What gets translated

llm-fusion converts between Anthropic Messages API and the internal OpenAI Chat Completions pipeline on the fly:

- `user` / `assistant` / `system` messages
- `text`, `image`, `tool_use`, and `tool_result` content blocks
- `tools` and `tool_choice`
- streaming SSE (`text_delta`, `input_json_delta`, `message_stop`, etc.)
- `max_tokens`, `temperature`, `top_p`

Top-level Anthropic-only fields like `thinking` and `metadata` are currently dropped.

## Verifying it works

After `fusion-claude` starts, you should see proxy logs like:

```
llm-fusion listening on http://127.0.0.1:8080
request complete model=fusion-agents status=200 stream=true
```

If Claude Code shows an API error, check:

1. The proxy is running (`curl http://127.0.0.1:8080/health`).
2. `ANTHROPIC_BASE_URL` does **not** end in `/v1`.
3. `ANTHROPIC_MODEL` matches a name in `fusion.yaml`.
4. `ANTHROPIC_AUTH_TOKEN` matches `FUSION_PROXY_TOKEN` if client auth is enabled.

## Example shell alias

Add to `~/.zshrc` or `~/.bashrc`:

```bash
alias fclaude='cd /path/to/your/project && /path/to/llm-fusion/bin/fusion-claude fusion-agents'
```

Then `fclaude` starts Claude Code wired to the proxy from any project.
