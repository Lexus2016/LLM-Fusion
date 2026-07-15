# Provider research — what to connect after Ollama Cloud

> Date: 2026-07-15. Question: which additional upstream providers should
> llm-fusion support, and how hard is each to add? Method: three parallel
> research passes (OpenRouter, FAL-AI, broad landscape) over current vendor docs.

## TL;DR

- **Every serious LLM inference provider is a drop-in OpenAI-compatible endpoint**
  (`POST {base}/v1/chat/completions`, `Bearer` auth, SSE, tool-calling, `usage`).
  So a single generic **`openai-compat`** connector type covers ~11 providers;
  they differ only in **base URL**, **model-ID format**, optional **headers**, and
  **how an exhausted account is signalled**.
- **Build now:** the generic `openai-compat` type, with **OpenRouter** as the
  first configured example. This simultaneously unlocks DeepInfra, Together,
  Novita, Nebius, Hyperbolic, DeepSeek, Mistral, Cerebras, Groq, Baseten — **by
  config, no new code.**
- **Skip:** **FAL-AI** for chat (poor fit, see §3), **Fireworks** (flagship
  Qwen3-Coder-480B is dedicated-deploy, not serverless), **Gemini** (proprietary,
  partial compat, project-bound keys) except as an optional non-open fallback.

## 1. The exhausted-account signal (drives auto-switch quality)

This is the crux for "switch when limits/billing run out":

- **`402` = clean, unambiguous "this key is out of money" → mark connector down.**
  Providers with a clean 402: **DeepInfra, DeepSeek-direct, Novita, Nebius,
  OpenRouter** (402 fires even on free models).
- **`429` is overloaded.** It means *both* transient rate-limit *and* period/day
  cap. Must read the body: `rate_limit_exceeded` + `Retry-After` = transient
  (cool down, keep key); `insufficient_quota` / "tokens per day" / "daily" =
  exhausted-for-period (rotate). **Groq and Cerebras have no 402 at all** —
  daily-cap exhaustion arrives only as a 429 with a distinctive message.
- **`401`** = dead/rotated key → down. **`403`** = permissions/moderation → do
  **not** rotate on this alone (it's request-specific, not account death).

llm-fusion's classifier implements exactly this: `401→auth`, `402→payment`,
`429→rate_limit` (escalated to `quota` when the body says daily-cap/insufficient),
`5xx/network/timeout→cooling`, `403→passthrough`.

## 2. Provider landscape (all OpenAI-compatible unless noted)

| Provider | Base URL | Notable coding models / ID format | Cost posture | Exhausted signal |
|---|---|---|---|---|
| **OpenRouter** | `openrouter.ai/api/v1` | 300–500+ models, `provider/model` (`qwen/qwen3-coder`, `anthropic/claude-…`); `:free` tier | passthrough at cost; prepaid credits; BYOK | **402** insufficient credits; 429 rate; mid-stream = SSE `finish_reason:"error"` |
| **DeepInfra** | `api.deepinfra.com/v1/openai` | DeepSeek V4, Qwen3.5-397B, Qwen3-Coder; `deepseek-ai/DeepSeek-V3.2` | cheapest tier, prepaid PAYG | **402** clean; 429 rate/busy |
| **Together AI** | `api.together.xyz/v1` | Qwen3-Coder-480B serverless, DeepSeek V4, Kimi, GLM; HF-style `Qwen/…` | prepaid PAYG | documented codes; 429 rate distinct from balance |
| **Novita AI** | `api.novita.ai/openai` | Qwen3-Coder-480B ($0.38/$1.55) & 30B ($0.07/$0.27), DeepSeek, GLM; `qwen/qwen3-coder-480b-…` | lowest cost-per-coding-token | 402-class out-of-credits; 429 rate |
| **Nebius** | `api.studio.nebius.com/v1` (rebranding → verify) | Qwen3-Coder-480B, DeepSeek V4 Pro, GLM-5.2; **all models support tools** | cheap $0.08–1.93/M | strict schema (429/401/402-class) |
| **Hyperbolic** | `api.hyperbolic.xyz/v1` | 25+ open models, tools on 18+; HF-style | very cheap PAYG | 429/401 (catalog churn) |
| **DeepSeek** | `api.deepseek.com` | own only: `deepseek-v4-flash`, `deepseek-v4-pro`, 1M ctx | rock-bottom for its models; 5M free on signup | **402** "Insufficient Balance" |
| **Mistral** | `api.mistral.ai/v1` | `codestral-latest`, `devstral-2-latest`; EU/GDPR | PAYG | 429/401/403 |
| **Groq** | `api.groq.com/openai/v1` | gpt-oss-120b/20b, Qwen3-32B, Kimi K2; fastest | cheap; free tier | **no 402**; day-cap → 429 w/ message |
| **Cerebras** | `api.cerebras.ai/v1` | public catalog narrowed to gpt-oss-120b (+GLM preview); ~3000 tok/s | free tier; Code Pro/Max | **no 402**; 429 only |
| **Fireworks** | `api.fireworks.ai/inference/v1` | Qwen3-Coder-480B **dedicated-only** ⚠ | PAYG | standard |
| **Gemini** | `…/v1beta/openai/` | proprietary `gemini-3-pro`; **partial compat** | free AI Studio tier | quirks; project-bound keys |

Multi-account (rotate keys on same base URL) is trivial for all **except Gemini**
(keys are Google-project-scoped).

## 3. FAL-AI — POOR FIT for a chat proxy

- fal.ai is a **generative-media** platform (FLUX, video, TTS, vision); it is not
  a text-LLM house.
- Its only OpenAI-compatible chat path
  (`fal.run/openrouter/router/openai/v1/chat/completions`) is a **reseller of
  OpenRouter** — same models you'd get direct, with a `Key <FAL_KEY>` auth quirk
  (not `Bearer`). A redundant middleman for chat.
- Its real value (media) lives behind a **queue/submit-poll** paradigm
  (`fal.subscribe`, webhooks) that needs a fundamentally different adapter.
- **Verdict:** don't add fal as a chat connector. If media generation is wanted
  later, expose it as a **function-callable tool** via a separate queue adapter —
  a different project, not part of the OpenAI-compatible pool.

## 4. Recommendation — order to add

1. **OpenRouter** — build now (the configured example of `openai-compat`); widest
   catalog, clean 402, trivial multi-key.
2. **DeepInfra** — cheapest broad open-weight coding catalog + clean 402; the best
   auto-switch fit. Config-only once `openai-compat` ships.
3. **Novita** — lowest cost-per-coding-token (Qwen3-Coder 480B & 30B).
4. **Together** — reliable serverless Qwen3-Coder-480B, documented error codes.
5. **DeepSeek-direct** — cheapest access to top DeepSeek V4 coding models, clean
   402, 5M free tokens (migrate off `deepseek-chat`/`reasoner` before 2026-07-24).

Fast lane (latency): **Cerebras / Groq** are excellent for a fast single/router
model but have narrow public coding catalogs and 429-only exhaustion.

All of #2–#5 require **zero new code** — only a `connectors:` entry with the right
`base_url`, `api_key_env`, and `model_map`.
