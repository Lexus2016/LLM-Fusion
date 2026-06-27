import { describe, it, expect } from "vitest";
import {
  buildDataUrl,
  imageUrlToNativeBase64,
  openAiBodyToNativeChat,
  openAiMessageToNative,
  openAiMessagesToNative,
  parseDataUrl,
  requestHasImages,
} from "../src/vision";
import { OllamaClient } from "../src/upstream/ollama";
import { NativeStreamingNotImplementedError } from "../src/errors";
import { mockFetch, jsonResponse } from "./helpers";
import type { ChatCompletionRequest, ChatMessage, FetchFn } from "../src/types";
import { z } from "zod";

// "hello" base64-encoded — a stable, byte-checkable payload.
const HELLO_B64 = Buffer.from("hello").toString("base64"); // aGVsbG8=
const PNG_DATA_URL = `data:image/png;base64,${HELLO_B64}`;

describe("vision — detection", () => {
  it("detects image_url content blocks", () => {
    const withImage: ChatCompletionRequest = {
      model: "x",
      messages: [
        { role: "user", content: [{ type: "image_url", image_url: { url: PNG_DATA_URL } }] },
      ],
    };
    expect(requestHasImages(withImage)).toBe(true);
  });

  it("returns false for text-only and string-content requests", () => {
    expect(requestHasImages({ model: "x", messages: [{ role: "user", content: "hi" }] })).toBe(false);
    expect(
      requestHasImages({ model: "x", messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] }),
    ).toBe(false);
    expect(requestHasImages({ model: "x" })).toBe(false);
  });
});

describe("vision — data URL transform (byte-accurate round trip)", () => {
  it("parses a base64 data URL into mime + payload", () => {
    const parts = parseDataUrl(PNG_DATA_URL);
    expect(parts).not.toBeNull();
    expect(parts?.mime).toBe("image/png");
    expect(parts?.base64).toBe(HELLO_B64);
  });

  it("returns null for non-data URLs", () => {
    expect(parseDataUrl("https://example.com/cat.png")).toBeNull();
    expect(imageUrlToNativeBase64("https://example.com/cat.png")).toBeNull();
  });

  it("round-trips data URL -> native base64 -> data URL, byte-accurate", () => {
    const base64 = imageUrlToNativeBase64(PNG_DATA_URL);
    expect(base64).toBe(HELLO_B64); // load-bearing bytes preserved exactly
    const back = buildDataUrl(base64!, "image/png");
    expect(back).toBe(PNG_DATA_URL); // full URL reconstructed identically
    // And the decoded bytes equal the original.
    expect(Buffer.from(base64!, "base64").toString()).toBe("hello");
  });

  it("converts an OpenAI message with an image into native images[]", () => {
    const msg: ChatMessage = {
      role: "user",
      content: [
        { type: "text", text: "describe" },
        { type: "image_url", image_url: { url: PNG_DATA_URL } },
      ],
    };
    const native = openAiMessageToNative(msg);
    expect(native.role).toBe("user");
    expect(native.content).toBe("describe");
    expect(native.images).toEqual([HELLO_B64]);
    // Reconstruct the data URL from the native image — byte-accurate.
    expect(buildDataUrl(native.images![0]!, "image/png")).toBe(PNG_DATA_URL);
  });

  it("passes string content through unchanged (no images)", () => {
    const native = openAiMessageToNative({ role: "user", content: "plain text" });
    expect(native).toEqual({ role: "user", content: "plain text" });
  });

  it("transforms a full OpenAI body into native /api/chat shape", () => {
    const body: Record<string, unknown> = {
      model: "vm1",
      stream: false,
      messages: [
        { role: "user", content: [{ type: "image_url", image_url: { url: PNG_DATA_URL } }] },
      ],
    };
    const native = openAiBodyToNativeChat(body);
    const NativeSchema = z.object({
      model: z.string(),
      messages: z.array(z.object({ role: z.string(), content: z.string(), images: z.array(z.string()).optional() })),
    });
    const parsed = NativeSchema.parse(native);
    expect(parsed.messages[0]?.images).toEqual([HELLO_B64]);
  });

  it("maps an array of messages", () => {
    const out = openAiMessagesToNative([
      { role: "system", content: "sys" },
      { role: "user", content: [{ type: "image_url", image_url: { url: PNG_DATA_URL } }] },
    ]);
    expect(out[0]).toEqual({ role: "system", content: "sys" });
    expect(out[1]?.images).toEqual([HELLO_B64]);
  });
});

describe("vision — native /api/chat client path", () => {
  it("posts the native body to /api/chat and returns the JSON result (non-stream)", async () => {
    let hitUrl = "";
    let sentBody = "";
    const fetchFn: FetchFn = mockFetch([
      {
        match: (u) => u.endsWith("/api/chat"),
        respond: (u, init) => {
          hitUrl = u;
          sentBody = String(init?.body);
          return jsonResponse({ message: { role: "assistant", content: "a cat" } });
        },
      },
    ]);
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
    const result = await client.chatNative({ model: "vm1", messages: [] }, { stream: false });
    expect(hitUrl).toContain("/api/chat");
    expect(JSON.parse(sentBody).stream).toBe(false);
    expect(result.kind).toBe("json");
    if (result.kind === "json") {
      const data = z.object({ message: z.object({ content: z.string() }) }).parse(result.data);
      expect(data.message.content).toBe("a cat");
    }
  });

  it("throws a typed deferred error for native streaming", async () => {
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: mockFetch([{ match: () => true, respond: () => jsonResponse({}) }]),
    });
    await expect(client.chatNative({ model: "vm1" }, { stream: true })).rejects.toBeInstanceOf(
      NativeStreamingNotImplementedError,
    );
  });
});
