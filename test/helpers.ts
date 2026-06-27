import type { FetchFn } from "../src/types";

/** Build a JSON `Response`. */
export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Build an SSE `Response` emitting `data: {...}` chunks then `data: [DONE]`. */
export function sseResponse(chunks: unknown[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      }
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

/**
 * An OK (status 200) streaming `Response` whose body errors *immediately*, before
 * any chunk is enqueued — simulates an upstream failure before the first token.
 */
export function streamErrorImmediate(message = "upstream stream failed"): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.error(new Error(message));
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

/**
 * An OK (status 200) streaming `Response` that emits `chunks` and then errors —
 * simulates a mid-stream upstream failure after the first token was forwarded.
 *
 * The error is raised on a later `pull` (not synchronously in `start`), so the
 * already-emitted chunks are delivered to the consumer before the failure —
 * `controller.error()` in `start` would instead discard any queued chunks.
 */
export function sseThenError(chunks: unknown[], message = "mid-stream failure"): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunks[i])}\n\n`));
        i += 1;
        return;
      }
      controller.error(new Error(message));
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

export interface MockRoute {
  match: (url: string, init?: RequestInit) => boolean;
  respond: (url: string, init?: RequestInit) => Response;
}

/** A mock fetch implementation routing by URL — no network, no API key. */
export function mockFetch(routes: MockRoute[]): FetchFn {
  return async (input, init) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    for (const route of routes) {
      if (route.match(url, init)) return route.respond(url, init);
    }
    return new Response(JSON.stringify({ error: `no mock route for ${url}` }), {
      status: 404,
      headers: { "content-type": "application/json" },
    });
  };
}
