import { z } from "zod";
import type { CapabilityProvider, ChatCompletionRequest, ChatMessage } from "./types";
import { CapabilityError } from "./errors";

/**
 * Vision handling: detect OpenAI `image_url` content blocks and convert between
 * the OpenAI `image_url` shape (incl. `data:` base64 URLs) and the native Ollama
 * `images: [base64]` shape.
 *
 * The OpenAI-compat path forwards `image_url` blocks verbatim (the primary path
 * while `api_mode` defaults to `auto`/`openai`). The native `/api/chat` path
 * needs raw base64 in `images[]`; the transform helpers here are pure and
 * unit-tested so the wiring stays trivial.
 */

// --- Detection ------------------------------------------------------------

/** Whether any message carries an OpenAI `image_url` content block. */
export function requestHasImages(request: ChatCompletionRequest): boolean {
  const messages = request.messages;
  if (!messages) return false;
  for (const message of messages) {
    const content = message.content;
    if (Array.isArray(content)) {
      for (const part of content) {
        if (part.type === "image_url") return true;
      }
    }
  }
  return false;
}

// --- Capability gate (shared) ---------------------------------------------

/**
 * Capability gate for a single-model target: if the request carries image
 * blocks and the resolved real target is known-non-vision, reject with a 400
 * `CapabilityError`. Discovery is only triggered when images are actually
 * present, so the common text path adds no upstream call. Shared by the `single`
 * dispatch gate (router.ts) and the `smart` -> `simple` sub-route (smart.ts) so
 * an image smart-routed to a non-vision target fails clean instead of producing
 * an opaque upstream error.
 */
export async function assertSingleVisionCapable(
  capabilities: CapabilityProvider,
  request: ChatCompletionRequest,
  target: string,
  virtualName: string,
): Promise<void> {
  if (!requestHasImages(request)) return;
  const { capability } = await capabilities.discover(target);
  if (!capability.vision) {
    throw new CapabilityError(
      `virtual model '${virtualName}' resolves to '${target}', which does not support image input`,
    );
  }
}

// --- data: URL parsing ----------------------------------------------------

/** The decoded pieces of a base64 `data:` URL. */
export interface DataUrlParts {
  mime: string;
  base64: string;
}

// data:[<mime>][;<param>...];base64,<payload>
const DATA_URL_RE = /^data:([^;,]*)(?:;[^,;]+)*;base64,([\s\S]+)$/;

/**
 * Parse a base64 `data:` URL into its mime type and base64 payload. Returns
 * `null` for any URL that is not a base64 `data:` URL (e.g. an `https://` URL).
 */
export function parseDataUrl(url: string): DataUrlParts | null {
  const m = DATA_URL_RE.exec(url);
  if (!m) return null;
  const mime = m[1] && m[1].length > 0 ? m[1] : "application/octet-stream";
  const base64 = m[2] ?? "";
  return { mime, base64 };
}

/** Build a base64 `data:` URL from a mime type and base64 payload. */
export function buildDataUrl(base64: string, mime = "image/png"): string {
  return `data:${mime};base64,${base64}`;
}

/**
 * Extract the raw base64 payload from an OpenAI `image_url` URL for the native
 * `images[]` array. Returns `null` when the URL is not an inline base64
 * `data:` URL (remote URLs cannot be inlined without a fetch, which the native
 * path does not perform).
 */
export function imageUrlToNativeBase64(url: string): string | null {
  const parts = parseDataUrl(url);
  return parts ? parts.base64 : null;
}

// --- Message <-> native transform ----------------------------------------

/** A native Ollama `/api/chat` message: text content plus optional images. */
export interface NativeMessage {
  role: string;
  content: string;
  images?: string[];
}

const ImageUrlPartSchema = z
  .object({
    type: z.literal("image_url"),
    image_url: z.object({ url: z.string() }).passthrough(),
  })
  .passthrough();

const TextPartSchema = z
  .object({ type: z.literal("text"), text: z.string() })
  .passthrough();

/**
 * Convert one OpenAI message into native shape. A string content passes through
 * unchanged; an array content is flattened into a joined text `content` plus an
 * `images[]` array of base64 payloads (data-URL images only). A non-data image
 * URL is preserved verbatim inside the text content so information is never
 * silently dropped.
 */
export function openAiMessageToNative(message: ChatMessage): NativeMessage {
  const role = message.role;
  const content = message.content;

  if (typeof content === "string") {
    return { role, content };
  }
  if (!Array.isArray(content)) {
    return { role, content: "" };
  }

  const textChunks: string[] = [];
  const images: string[] = [];
  for (const part of content) {
    const img = ImageUrlPartSchema.safeParse(part);
    if (img.success) {
      const base64 = imageUrlToNativeBase64(img.data.image_url.url);
      if (base64 !== null) {
        images.push(base64);
      } else {
        textChunks.push(`[image: ${img.data.image_url.url}]`);
      }
      continue;
    }
    const txt = TextPartSchema.safeParse(part);
    if (txt.success) textChunks.push(txt.data.text);
  }

  const native: NativeMessage = { role, content: textChunks.join("\n") };
  if (images.length > 0) native.images = images;
  return native;
}

/** Convert an OpenAI messages array into native messages. */
export function openAiMessagesToNative(messages: ChatMessage[]): NativeMessage[] {
  return messages.map(openAiMessageToNative);
}

/**
 * Transform a full OpenAI-shaped chat body into the native `/api/chat` shape:
 * messages are converted to native (with inline base64 `images[]`); all other
 * fields pass through unchanged.
 */
export function openAiBodyToNativeChat(body: Record<string, unknown>): Record<string, unknown> {
  const rawMessages = body.messages;
  const parsed = z.array(z.unknown()).safeParse(rawMessages);
  if (!parsed.success) return { ...body };
  const messages = MessagesSchema.parse(rawMessages);
  return { ...body, messages: openAiMessagesToNative(messages) };
}

const MessagesSchema = z.array(
  z
    .object({
      role: z.string(),
      content: z
        .union([z.string(), z.array(z.object({ type: z.string() }).passthrough()), z.null()])
        .optional(),
    })
    .passthrough(),
);
