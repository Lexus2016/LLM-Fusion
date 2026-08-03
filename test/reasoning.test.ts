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

  it("drops a reasoning-phase false partial once real content arrives", async () => {
    const reasoningChunk = (r: string) => `data: ${JSON.stringify({ choices: [{ delta: { reasoning: r } }] })}\n`;
    const out = await pump([reasoningChunk("planning <thi"), chunk("REAL"), "data: [DONE]\n"]);
    const texts = out
      .split("\n")
      .filter((l) => l.startsWith("data:") && l.trim() !== "data: [DONE]")
      .map((l) => (JSON.parse(l.slice(5).trim()) as { choices: Array<{ delta: { content?: string } }> }).choices[0]?.delta.content ?? "");
    // Real content proves the reasoning was private chain-of-thought — its text
    // AND its carried false-partial tag go with it, rather than being prepended
    // to the answer the client renders.
    expect(texts.join("")).toBe("REAL");
    expect(out).not.toContain("planning");
  });

  const reasoningChunk = (r: string) => `data: ${JSON.stringify({ choices: [{ delta: { reasoning: r } }] })}\n`;
  const contentsOf = (out: string): string =>
    out
      .split("\n")
      .filter((l) => l.startsWith("data:") && l.trim() !== "data: [DONE]")
      .map((l) => (JSON.parse(l.slice(5).trim()) as { choices: Array<{ delta?: { content?: string } }> }).choices[0]?.delta?.content ?? "")
      .join("");

  it("never flushes reasoning as the answer on a tool turn (no content, tool_calls present)", async () => {
    // An agent-loop step is reasoning + a tool call and NO prose. Flushing the
    // buffer there would inject private chain-of-thought straight into the
    // visible transcript — the non-stream path has always guarded this.
    const toolChunk = `data: ${JSON.stringify({
      choices: [{ delta: { tool_calls: [{ index: 0, id: "t1", function: { name: "bash", arguments: "{}" } }] } }] })}\n`;
    const finish = `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })}\n`;
    const out = await pump([reasoningChunk("I should run ls to check."), toolChunk, finish, "data: [DONE]\n"]);
    expect(out).not.toContain("I should run ls");
    expect(contentsOf(out)).toBe("");
    expect(out).toContain('"name":"bash"'); // the tool call itself survives untouched
    expect(out).toContain("[DONE]");
  });

  it("delivers a reasoning-only answer BEFORE the finish_reason chunk", async () => {
    // Clients that stop accumulating at finish_reason would never see content
    // emitted after it.
    const finish = `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n`;
    const out = await pump([reasoningChunk("the whole "), reasoningChunk("answer"), finish, "data: [DONE]\n"]);
    expect(contentsOf(out)).toBe("the whole answer");
    const idxAnswer = out.indexOf("the whole answer");
    const idxFinish = out.indexOf('"finish_reason":"stop"');
    expect(idxAnswer).toBeGreaterThanOrEqual(0);
    expect(idxFinish).toBeGreaterThan(idxAnswer);
  });

  it("strips reasoning fields from chunks arriving after the answer started", async () => {
    const out = await pump([chunk("ANSWER"), reasoningChunk("late private thought"), "data: [DONE]\n"]);
    expect(out).not.toContain("late private thought");
    expect(out).not.toContain('"reasoning"');
    expect(contentsOf(out)).toBe("ANSWER");
  });

  it("promotes reasoning_content as well as reasoning", async () => {
    const rc = `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: "via reasoning_content" } }] })}\n`;
    const out = await pump([rc, "data: [DONE]\n"]);
    expect(contentsOf(out)).toBe("via reasoning_content");
  });

  it("a whitespace-only content delta does not discard a reasoning-only answer", async () => {
    const out = await pump([chunk(" "), reasoningChunk("the real answer"), "data: [DONE]\n"]);
    expect(contentsOf(out)).toContain("the real answer");
  });

  it("content that is entirely an inline <think> block does not discard a reasoning-only answer", async () => {
    // The latch must fire on the FILTERED text: raw content of "<think>…</think>"
    // is chain-of-thought, and latching on it would drop the buffer and then
    // strip that same content, leaving the client with nothing.
    const out = await pump([reasoningChunk("ANSWER"), chunk("<think>inline cot</think>"), "data: [DONE]\n"]);
    expect(contentsOf(out)).toBe("ANSWER");
    expect(out).not.toContain("inline cot");
  });

  it("keeps per-choice state so one choice's content cannot swallow another's answer (n>1)", async () => {
    const two = `data: ${JSON.stringify({
      choices: [
        { index: 0, delta: { content: "A0" } },
        { index: 1, delta: { reasoning: "A1" } },
      ],
    })}\n`;
    const out = await pump([two, "data: [DONE]\n"]);
    const perChoice = out
      .split("\n")
      .filter((l) => l.startsWith("data:") && l.trim() !== "data: [DONE]")
      .flatMap((l) => (JSON.parse(l.slice(5).trim()) as { choices: Array<{ index?: number; delta?: { content?: string } }> }).choices)
      .reduce<Record<number, string>>((acc, c) => {
        acc[c.index ?? 0] = (acc[c.index ?? 0] ?? "") + (c.delta?.content ?? "");
        return acc;
      }, {});
    expect(perChoice[0]).toBe("A0");
    expect(perChoice[1]).toBe("A1"); // was lost when the state was transform-wide
  });

  it("does not re-buffer reasoning that trails a choice's finish_reason", async () => {
    // Regression from the per-choice refactor: deleting the state on flush let
    // a trailing reasoning fragment recreate it with a fresh latch and reach
    // the client as content at [DONE].
    const finish = `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n`;
    const out = await pump([chunk("ANSWER"), finish, reasoningChunk("trailing cot"), "data: [DONE]\n"]);
    expect(out).not.toContain("trailing cot");
    expect(contentsOf(out)).toBe("ANSWER");
  });

  it("treats a chunk it cannot parse but that mentions tool_calls as a tool turn", async () => {
    // Failing open here would flush chain-of-thought into an agent transcript.
    const deviant = `data: {"choices":[{"delta":{"tool_calls":"not-an-array"}}],"weird":\n`;
    const out = await pump([reasoningChunk("private plan"), deviant, "data: [DONE]\n"]);
    expect(out).not.toContain("private plan");
  });
});
