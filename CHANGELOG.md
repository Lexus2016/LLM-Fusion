# Changelog

All notable changes to this project will be documented in this file.

## [0.1.11] - 2026-06-28

### Added

- Smart-router decision cache: an identical router request body now reuses the prior classification instead of re-paying a router round-trip every turn. Only successfully parsed decisions are cached; failures (timeout, non-OK status, unparseable JSON) fall through to the default route and are NOT cached, so a transient blip self-heals on the next identical request.
- In-flight coalescing: concurrent identical router requests share a single upstream call, so a burst of identical in-flight requests does not pay N router round-trips.
- LRU-bounded cache (max 256 entries) to keep memory predictable across long-running sessions.

## [0.1.10] - 2026-06-28

### Added

- Fusion panel gains conditional abort: slow panel members are cancelled once enough survivors have answered, cutting tail latency while preserving `min_panel_success`.
- Thinking/reasoning tags are stripped from panel answers before they reach the judge, keeping the analysis focused on substantive content.
- Smart-router prompt tuning improves routing accuracy between `simple` and `fusion` strategies.
- Anti-hallucination safeguards in the fusion pipeline:
  - Judge now emits `hallucination_flags`, cross-referencing panel experts and flagging claims supported by only one member as suspect.
  - Synth is instructed to omit or caveat flagged items and to acknowledge uncertainty when experts disagree irreconcilably.

## [0.1.9] - 2026-06-27

### Fixed

- Anthropic `/v1/messages` endpoint no longer rejects valid Claude Code requests with `messages.N.content: Invalid input`.
  - Accepts `content: null` on user/assistant/system messages and normalises it to an empty string for upstream translation.
  - Accepts `thinking` and `redacted_thinking` content blocks (e.g. from extended-thinking models) and ignores them when translating to OpenAI.
  - Accepts `tool_result` blocks with `content: null`.
  - Accepts `tool_result` blocks whose content is an array containing image blocks; images are converted to a URL/data-URI string because OpenAI `tool` messages only support string content.
  - Ensures assistant messages that end up with no content and no tool calls fall back to `content: ""`, keeping the upstream OpenAI request valid.

### Added

- Exported `AnthropicRequestSchema` for reuse in tests and diagnostics.
- Unit and integration tests covering null content, thinking blocks, and image tool results.

## [0.1.8] - 2026-06-27

### Previous release baseline

- Existing Fusion Proxy functionality up to and including the Anthropic compatibility layer, strategy dispatcher (single/fusion/smart/failover), usage accounting, auth, and capability discovery.
