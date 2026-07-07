import { describe, it, expect } from "vitest";
import {
  createThinkTagStreamFilter,
  stripThinkingTags,
  makeReasoningPromotionTransform,
} from "../src/reasoning";

describe("createThinkTagStreamFilter", () => {
  function runFragments(fragments: string[]): string {
    const f = createThinkTagStreamFilter();
    let out = "";
    for (const frag of fragments) out += f.push(frag);
    out += f.flush();
    return out;
  }

  it("matches stripThinkingTags on a single whole fragment", () => {
    const s = "before <think>private plan</think> after";
    expect(runFragments([s])).toBe(stripThinkingTags(s));
  });

  it("strips an OPEN tag split across two fragments", () => {
    expect(runFragments(["Hello <th", "ink>secret</think> world"])).toBe("Hello  world");
  });

  it("strips a CLOSE tag split across two fragments", () => {
    expect(runFragments(["A <think>secret</thi", "nk> B"])).toBe("A  B");
  });

  it("suppresses think CONTENT that spans many fragments", () => {
    expect(runFragments(["<think>", "step 1... ", "step 2... ", "</think>", "REAL ANSWER"])).toBe("REAL ANSWER");
  });

  it("emits a false partial tag as literal text once it diverges", () => {
    // "<tho" starts like "<think>" but diverges — must come back out as text.
    expect(runFragments(["The tag <tho", "ught> is literal"])).toBe("The tag <thought> is literal");
  });

  it("strips an orphan close tag outside a think block", () => {
    expect(runFragments(["A </think> B"])).toBe("A  B");
  });

  it("is case-insensitive like stripThinkingTags", () => {
    expect(runFragments(["x <THINK>hidden</THINK> y"])).toBe("x  y");
  });

  it("handles several think blocks across mixed boundaries", () => {
    expect(
      runFragments(["a<think>1</think>b<t", "hink>2</t", "hink>c"]),
    ).toBe("abc");
  });

  it("drops an unterminated think block at stream end", () => {
    expect(runFragments(["visible ", "<think>never closed..."])).toBe("visible ");
  });
});

describe("makeReasoningPromotionTransform — think tags across delta boundaries", () => {
  async function pump(lines: string[]): Promise<string> {
    const t = makeReasoningPromotionTransform();
    const writer = t.writable.getWriter();
    const enc = new TextEncoder();
    const dec = new TextDecoder();
    const readAll = (async () => {
      let s = "";
      for await (const c of t.readable) s += dec.decode(c as Uint8Array, { stream: true });
      return s;
    })();
    for (const l of lines) await writer.write(enc.encode(l));
    await writer.close();
    return readAll;
  }
  const chunk = (content: string) => `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n`;

  it("strips a tag split across two content deltas and suppresses the block body", async () => {
    const out = await pump([chunk("Hello <th"), chunk("ink>secret</think> world"), "data: [DONE]\n"]);
    expect(out).not.toContain("secret");
    expect(out).not.toContain("ink>");
    expect(out).toContain("Hello");
    expect(out).toContain("world");
  });

  it("surfaces a false-partial tail as a valid, separately framed chunk before [DONE]", async () => {
    const withMeta = `data: ${JSON.stringify({ id: "c-1", model: "m-1", choices: [{ delta: { content: "ends with <th" } }] })}\n`;
    const out = await pump([withMeta, "data: [DONE]\n"]);
    const lines = out.split("\n");
    const doneIdx = lines.findIndex((l) => l.trim() === "data: [DONE]");
    expect(doneIdx).toBeGreaterThan(0);
    expect(lines[doneIdx - 1]).toBe(""); // blank line closes the tail event before [DONE]
    // Every data line except [DONE] must be valid JSON (no merged SSE events).
    const dataPayloads = lines
      .filter((l) => l.startsWith("data:") && l.trim() !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data:".length).trim()) as { model?: string; choices: Array<{ delta: { content?: string } }> });
    const tail = dataPayloads.find((p) => p.choices[0]?.delta.content === "<th");
    expect(tail).toBeDefined(); // the literal text is not swallowed…
    expect(tail?.model).toBe("m-1"); // …and the synthetic chunk carries the stream's metadata
  });

  it("merges a reasoning-phase false partial before the first content, not after it", async () => {
    const reasoningChunk = (r: string) => `data: ${JSON.stringify({ choices: [{ delta: { reasoning: r } }] })}\n`;
    const out = await pump([reasoningChunk("planning <thi"), chunk("REAL"), "data: [DONE]\n"]);
    const texts = out
      .split("\n")
      .filter((l) => l.startsWith("data:") && l.trim() !== "data: [DONE]")
      .map((l) => (JSON.parse(l.slice(5).trim()) as { choices: Array<{ delta: { content?: string } }> }).choices[0]?.delta.content ?? "");
    expect(texts.join("")).toBe("planning <thiREAL"); // literal narration stays in order
  });
});
