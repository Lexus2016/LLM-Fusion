/**
 * Extract the first balanced top-level JSON object substring from `text`.
 *
 * Upstream models (especially "thinking" models) are inconsistent about honoring
 * `response_format: { type: "json_object" }` and "JSON only" instructions: the same
 * model may return a clean `{...}` on one call and wrap it in ```json … ``` fences,
 * or prepend a sentence, on the next. A naive `JSON.parse(content)` then fails
 * intermittently. This scans for the first `{` and returns through its matching
 * `}` (tracking string literals and escapes), so the payload survives code fences,
 * leading prose, and trailing commentary.
 *
 * Returns null when no balanced object is present (e.g. truncated output), letting
 * the caller fall back exactly as it would for unparseable content.
 */
export function extractJsonObject(text: string): string | null {
  const start = text.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/**
 * True when `value` is a plain JSON object — not null, not an array. The only
 * shape the OpenAI protocol allows for a tool call's `function.arguments`.
 */
export function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * True when `text` parses as a JSON OBJECT. A bare `JSON.parse` also accepts
 * `5`, `true`, `null` and `[1,2]`, which are not runnable tool arguments.
 *
 * This lives here because three call sites used to answer the question three
 * different ways on the same upstream bytes: the OpenAI surface emitted a
 * scalar-argument tool call as a finished, runnable turn while the Anthropic
 * surface reported `stop_reason: "max_tokens"` with an empty `input` — an
 * agent loop that retries forever against a deterministic upstream.
 */
export function isJsonObjectString(text: string): boolean {
  try {
    return isJsonObject(JSON.parse(text));
  } catch {
    return false;
  }
}
