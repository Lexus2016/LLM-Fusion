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
