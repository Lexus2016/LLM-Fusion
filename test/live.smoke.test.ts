import { describe, it, expect, afterAll } from "vitest";
import { z } from "zod";

/**
 * Phase 0 — gated LIVE upstream verification (spec §11, §16 Assumptions Register).
 *
 * Makes REAL calls to Ollama Cloud. The whole suite is SKIPPED unless
 * `OLLAMA_API_KEY` is set, so the default offline `npm test` stays green and
 * network-free. Run it explicitly with a key:
 *
 *     OLLAMA_API_KEY=ollama-... npm run smoke
 *
 * Each assumption is its own `it(...)` with a clear PASS/FAIL log. Tool-calling
 * and vision are LOGGED rather than hard-asserted (a model lacking a capability
 * is reality to record, not a suite failure) — but auth failures (401) DO fail
 * loudly. The final `afterAll` prints a summary you use to finalize `api_mode`
 * and the vision path in fusion.yaml.
 */

const KEY = process.env.OLLAMA_API_KEY;
const BASE = (process.env.OLLAMA_BASE_URL ?? "https://ollama.com").replace(/\/+$/, "");
const PANEL = ["glm-5.2", "kimi-k2.7-code", "deepseek-v4-pro"] as const;
const VISION_MODEL = process.env.FUSION_VISION_MODEL ?? "kimi-k2.7-code";
// 1x1 transparent PNG as an OpenAI image_url data URL.
const TINY_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

const TIMEOUT_MS = Number(process.env.SMOKE_TIMEOUT_MS ?? "120000");

function authHeaders(): Record<string, string> {
  return { "content-type": "application/json", authorization: `Bearer ${KEY ?? ""}` };
}

function chat(body: Record<string, unknown>): Promise<Response> {
  return fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
}

// --- Minimal zod views over the upstream payloads (no `any`) ----------------

const CompletionSchema = z
  .object({
    choices: z
      .array(
        z
          .object({
            message: z
              .object({
                content: z.union([z.string(), z.null()]).optional(),
                tool_calls: z.array(z.unknown()).optional(),
              })
              .passthrough(),
          })
          .passthrough(),
      )
      .optional(),
  })
  .passthrough();

const ShowSchema = z
  .object({
    capabilities: z.array(z.string()).optional(),
    model_info: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough();

function firstContextLength(modelInfo: Record<string, unknown> | undefined): {
  key: string | null;
  value: number | null;
} {
  if (!modelInfo) return { key: null, value: null };
  for (const [k, v] of Object.entries(modelInfo)) {
    if (k.endsWith(".context_length") && typeof v === "number") return { key: k, value: v };
  }
  return { key: null, value: null };
}

// --- Accumulated results for the final summary ------------------------------

interface ShowInfo {
  caps: string[] | undefined;
  contextKey: string | null;
  context: number | null;
}
interface Summary {
  a1: string;
  sse: string;
  tools: Record<string, string>;
  show: Record<string, ShowInfo>;
  vision: { model: string; status: number; accepted: boolean } | null;
}
const summary: Summary = { a1: "?", sse: "?", tools: {}, show: {}, vision: null };

const live = describe.skipIf(!KEY);

live("llm-fusion live smoke (Phase 0 verification gate)", () => {
  it(
    "A-1 — Bearer auth: non-stream chat to glm-5.2 returns a completion",
    async () => {
      const res = await chat({
        model: "glm-5.2",
        stream: false,
        max_tokens: 16,
        messages: [{ role: "user", content: "Reply with the single word: pong" }],
      });
      expect(res.status, "Bearer auth must be accepted on /v1/chat/completions").toBe(200);
      const parsed = CompletionSchema.safeParse(await res.json());
      const content = parsed.success ? parsed.data.choices?.[0]?.message.content : undefined;
      console.log(`[A-1] status=${res.status} content=${JSON.stringify(content)?.slice(0, 80)}`);
      expect(typeof content).toBe("string");
      summary.a1 = "PASS";
    },
    TIMEOUT_MS,
  );

  it(
    "A-1/SSE — streaming: stream:true returns SSE data chunks",
    async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "glm-5.2",
          stream: true,
          max_tokens: 16,
          messages: [{ role: "user", content: "Count: one two three" }],
        }),
      });
      expect(res.status).toBe(200);
      const contentType = res.headers.get("content-type") ?? "";
      const text = await res.text();
      const sawDataLines = /(^|\n)data:/.test(text);
      console.log(
        `[SSE] status=${res.status} content-type=${contentType} sawDataLines=${sawDataLines} bytes=${text.length}`,
      );
      expect(sawDataLines, "streaming response should contain SSE `data:` lines").toBe(true);
      summary.sse = "PASS";
    },
    TIMEOUT_MS,
  );

  it.each(PANEL)(
    "A-2 — tool-calling: %s (logs PASS/FAIL; does not hard-fail when a model lacks tools)",
    async (model) => {
      const tools = [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Get the current weather for a city.",
            parameters: {
              type: "object",
              properties: { city: { type: "string", description: "City name" } },
              required: ["city"],
            },
          },
        },
      ];
      const res = await chat({
        model,
        stream: false,
        max_tokens: 128,
        tool_choice: "auto",
        tools,
        messages: [{ role: "user", content: "What is the weather in Paris right now? Use the tool." }],
      });
      // Auth/availability must work; tool support itself is recorded, not asserted.
      expect(res.status, `${model}: upstream must not reject the key`).not.toBe(401);
      const parsed = CompletionSchema.safeParse(await res.json());
      const toolCalls = parsed.success ? parsed.data.choices?.[0]?.message.tool_calls : undefined;
      const emitted = Array.isArray(toolCalls) ? toolCalls.length : 0;
      const verdict = res.status === 200 && emitted > 0 ? "PASS" : "FAIL";
      console.log(
        `[A-2] ${model}: tool_calls ${verdict} (status=${res.status}, emitted=${emitted})`,
      );
      summary.tools[model] = verdict;
    },
    TIMEOUT_MS,
  );

  it.each(PANEL)(
    "A-3 — /api/show: %s returns capabilities[] + a *.context_length",
    async (model) => {
      const res = await fetch(`${BASE}/api/show`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model }),
      });
      console.log(`[A-3] ${model}: /api/show status=${res.status}`);
      expect(res.status).toBe(200);
      const parsed = ShowSchema.safeParse(await res.json());
      const caps = parsed.success ? parsed.data.capabilities : undefined;
      const ctx = firstContextLength(parsed.success ? parsed.data.model_info : undefined);
      console.log(
        `[A-3] ${model}: capabilities=${JSON.stringify(caps)} context(${ctx.key})=${ctx.value}`,
      );
      expect(Array.isArray(caps), `${model}: capabilities[] should be present`).toBe(true);
      summary.show[model] = { caps, contextKey: ctx.key, context: ctx.value };
    },
    TIMEOUT_MS,
  );

  it(
    "A-4/A-5 — vision: OpenAI image_url accepted by a vision model (logged; not fatal)",
    async () => {
      const res = await chat({
        model: VISION_MODEL,
        stream: false,
        max_tokens: 32,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What colour is this image? Answer in one word." },
              { type: "image_url", image_url: { url: TINY_PNG } },
            ],
          },
        ],
      });
      const status = res.status;
      const bodyText = (await res.text()).slice(0, 200);
      const accepted = status === 200;
      console.log(
        `[A-4/A-5] vision model='${VISION_MODEL}' /v1 image_url status=${status} accepted=${accepted} body=${bodyText}`,
      );
      summary.vision = { model: VISION_MODEL, status, accepted };
      // Auth must work; whether THIS model/format is supported is informational.
      expect(status, "vision request must not be an auth failure").not.toBe(401);
    },
    TIMEOUT_MS,
  );

  afterAll(() => {
    const line = "=".repeat(54);
    const out: string[] = [];
    out.push("");
    out.push(line);
    out.push("  llm-fusion — Phase 0 live-smoke summary");
    out.push(line);
    out.push(`  A-1 Bearer non-stream : ${summary.a1}`);
    out.push(`  SSE streaming         : ${summary.sse}`);
    out.push("  A-2 tool_calls per model:");
    for (const m of PANEL) out.push(`      ${m.padEnd(16)} ${summary.tools[m] ?? "?"}`);
    out.push("  A-3 /api/show:");
    for (const m of PANEL) {
      const s = summary.show[m];
      out.push(`      ${m.padEnd(16)} caps=${JSON.stringify(s?.caps)} context=${s?.context ?? "?"}`);
    }
    const v = summary.vision;
    out.push(`  A-4/A-5 vision (${v?.model}) : status=${v?.status} accepted=${v?.accepted}`);
    out.push(line);
    out.push("  Finalize in fusion.yaml:");
    out.push("   - api_mode: 'openai' if A-4/A-5 accepted=true; else 'native'.");
    out.push("   - synth: pick a model whose A-2 row is PASS (synth must emit tool_calls).");
    out.push(line);
    out.push("");
    console.log(out.join("\n"));
  });
});
