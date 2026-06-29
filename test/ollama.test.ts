import { describe, it, expect } from "vitest";
import { OllamaClient } from "../src/upstream/ollama";
import type { FetchFn } from "../src/types";

/**
 * Dynamic request_timeout_s: for streaming, the hard timeout is connection /
 * first-response only. Once the response headers arrive, the timeout is cleared
 * so a slow-but-progressing stream is NOT truncated mid-delivery. This file
 * verifies that behavior directly against OllamaClient with a mock fetch.
 */

function streamResponse(signal: AbortSignal | undefined | null, emitAfterMs: number): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      // If the (still-active) hard timeout ever aborts the fetch signal, error
      // the stream immediately — this is exactly what would truncate delivery.
      const onAbort = (): void => controller.error(new DOMException("aborted", "AbortError"));
      if (signal) {
        if (signal.aborted) {
          onAbort();
          return;
        }
        signal.addEventListener("abort", onAbort, { once: true });
      }
      // Emit a chunk AFTER the hard timeout would have fired. With the dynamic
      // timeout, the signal is NOT aborted by then, so this succeeds.
      const t = setTimeout(() => {
        controller.enqueue(enc.encode('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'));
        controller.enqueue(enc.encode("data: [DONE]\n\n"));
        controller.close();
      }, emitAfterMs);
      // Don't keep the event loop alive for the test.
      t.unref?.();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

describe("OllamaClient — dynamic request_timeout_s for streaming", () => {
  it("does NOT hard-cut a stream that delivers after the timeout (headers arrived in time)", async () => {
    const timeoutMs = 50;
    // The chunk arrives at 120ms — well past the 50ms hard timeout. With the old
    // fixed-timeout behavior the fetch signal would abort at 50ms and the stream
    // would error; with the dynamic (phase) timeout the signal is cleared once
    // headers arrive, so the chunk is delivered.
    let capturedSignal: AbortSignal | null | undefined;
    const fetchFn: FetchFn = async (_input, init) => {
      capturedSignal = init?.signal;
      return streamResponse(init?.signal, 120);
    };
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn, timeoutMs });

    const res = await client.chatCompletions({ model: "m", stream: true }, { stream: true });
    expect(res.kind).toBe("stream");
    if (res.kind !== "stream") throw new Error("expected stream");
    if (!res.body) throw new Error("expected non-null stream body");

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let text = "";
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += dec.decode(value);
    }
    expect(text).toContain('"content":"hi"');
    expect(text).toContain("data: [DONE]");
    // The fetch signal must not have been aborted by our hard timeout after headers.
    expect(capturedSignal?.aborted).toBe(false);
  });
});