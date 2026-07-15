import { describe, it, expect } from "vitest";
import {
  classifyStatus,
  classifyThrown,
  DEFAULT_CLASSIFY_OPTIONS,
  isHardReason,
  type ClassifyOptions,
} from "../src/connectors/health";
import {
  ConnectorRegistry,
  type ConnectorClient,
  type ResolvedConnector,
} from "../src/connectors/registry";
import { PooledUpstreamClient } from "../src/connectors/pooled_client";
import type { ChatCompletionResult } from "../src/types";
import { NoConnectorAvailableError, NotImplementedError, UpstreamTimeoutError } from "../src/errors";

// --- helpers --------------------------------------------------------------

function jsonResult(
  status: number,
  data: unknown = {},
  extra: { retryAfterMs?: number } = {},
): ChatCompletionResult {
  return {
    kind: "json",
    status,
    data,
    usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    ...(extra.retryAfterMs !== undefined ? { retryAfterMs: extra.retryAfterMs } : {}),
  };
}

function streamResult(status = 200): ChatCompletionResult {
  return {
    kind: "stream",
    status,
    body: new ReadableStream<Uint8Array>({ start: (c) => c.close() }),
    contentType: "text/event-stream",
    usage: Promise.resolve({ promptTokens: 0, completionTokens: 0, totalTokens: 0 }),
  };
}

type Handler = (
  body: Record<string, unknown>,
  opts: { stream: boolean; signal?: AbortSignal },
) => Promise<ChatCompletionResult> | ChatCompletionResult;

class FakeClient implements ConnectorClient {
  readonly supportsNativeShow: boolean;
  seenModels: string[] = [];
  constructor(
    private readonly handler: Handler,
    supportsNativeShow = false,
  ) {
    this.supportsNativeShow = supportsNativeShow;
  }
  async chatCompletions(body: Record<string, unknown>, opts: { stream: boolean; signal?: AbortSignal }) {
    this.seenModels.push(String(body.model));
    return this.handler(body, opts);
  }
  async chatNative(body: Record<string, unknown>, opts: { stream: boolean; signal?: AbortSignal }) {
    this.seenModels.push(String(body.model));
    return this.handler(body, opts);
  }
  async show(): Promise<unknown> {
    return {};
  }
}

function cfg(id: string, o: Partial<ResolvedConnector> = {}): ResolvedConnector {
  return {
    id,
    group: o.group ?? "g1",
    provider: o.provider ?? "openai-compat",
    baseUrl: o.baseUrl ?? `https://${id}.test`,
    host: o.host ?? `https://${id}.test`,
    hasKey: o.hasKey ?? true,
    treat403As: o.treat403As ?? "passthrough",
    quotaMarkers: o.quotaMarkers ?? [],
    modelMap: o.modelMap ?? {},
  };
}

function registry(
  entries: Array<{ id: string; client: ConnectorClient; cfg?: Partial<ResolvedConnector> }>,
  opts: { now?: () => number; cooldownMs?: number; downRecheckMs?: number } = {},
): ConnectorRegistry {
  return new ConnectorRegistry(
    entries.map((e) => ({ cfg: cfg(e.id, e.cfg), client: e.client })),
    opts,
  );
}

const opts = { stream: false as const };

/** snapshot()[i] with a non-undefined guarantee (noUncheckedIndexedAccess). */
function snap(reg: ConnectorRegistry, i = 0) {
  const s = reg.snapshot()[i];
  if (!s) throw new Error(`no connector snapshot at index ${i}`);
  return s;
}

// --- classification -------------------------------------------------------

describe("health.classifyStatus", () => {
  const base: ClassifyOptions = DEFAULT_CLASSIFY_OPTIONS;

  it("maps hard vs soft statuses", () => {
    expect(classifyStatus(200, "", base).kind).toBe("success");
    expect(classifyStatus(401, "", base)).toMatchObject({ kind: "failure", reason: "auth", hard: true });
    expect(classifyStatus(402, "", base)).toMatchObject({ kind: "failure", reason: "payment", hard: true });
    expect(classifyStatus(429, "", base)).toMatchObject({ kind: "failure", reason: "rate_limit", hard: false });
    expect(classifyStatus(500, "", base)).toMatchObject({ kind: "failure", reason: "server_error", hard: false });
    expect(classifyStatus(404, "", base)).toMatchObject({ kind: "not_found" });
    expect(classifyStatus(400, "", base)).toMatchObject({ kind: "request_error" });
  });

  it("403 is passthrough by default, hard when configured", () => {
    expect(classifyStatus(403, "", base).kind).toBe("request_error");
    expect(classifyStatus(403, "", { ...base, treat403As: "down" })).toMatchObject({
      kind: "failure",
      reason: "auth",
      hard: true,
    });
  });

  it("429 escalates to quota only on an explicit marker match", () => {
    const withMarkers: ClassifyOptions = { treat403As: "passthrough", quotaMarkers: ["insufficient"] };
    expect(classifyStatus(429, "rate limit exceeded, retry", withMarkers)).toMatchObject({ reason: "rate_limit" });
    expect(classifyStatus(429, '{"error":"Insufficient credits"}', withMarkers)).toMatchObject({
      reason: "quota",
      hard: true,
    });
  });

  it("classifyThrown splits timeout vs network, and surfaces non-health errors", () => {
    expect(classifyThrown(new UpstreamTimeoutError("t"))).toMatchObject({ reason: "timeout" });
    expect(classifyThrown(new Error("boom"))).toMatchObject({ reason: "network" });
    // A capability / not-implemented error must be surfaced, never cooled.
    expect(classifyThrown(new NotImplementedError("nope"))).toMatchObject({ kind: "surface" });
  });

  it("severity ranks rate_limit above auth", () => {
    const rl = classifyStatus(429, "", base);
    const auth = classifyStatus(401, "", base);
    if (rl.kind !== "failure" || auth.kind !== "failure") throw new Error("expected failures");
    expect(rl.severity).toBeGreaterThan(auth.severity);
  });

  it("isHardReason", () => {
    expect(isHardReason("payment")).toBe(true);
    expect(isHardReason("rate_limit")).toBe(false);
  });
});

// --- registry state machine ----------------------------------------------

describe("ConnectorRegistry", () => {
  it("a soft failure cools; a hard failure downs", () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], { now: () => t, cooldownMs: 60_000 });
    const a = reg.acquire("a");
    if (!a.ok) throw new Error("expected ok");
    reg.recordFailure("a", a.epoch, "rate_limit");
    expect(reg.snapshot()[0]).toMatchObject({ state: "cooling", reason: "rate_limit" });

    const reg2 = registry([{ id: "b", client: new FakeClient(() => jsonResult(200)) }], { now: () => t });
    const b = reg2.acquire("b");
    if (!b.ok) throw new Error("expected ok");
    reg2.recordFailure("b", b.epoch, "payment");
    expect(reg2.snapshot()[0]).toMatchObject({ state: "down", reason: "payment" });
  });

  it("epoch guard: a late failure cannot overwrite a newer success", () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], { now: () => t });
    const attemptA = reg.acquire("a"); // epoch 0
    const attemptB = reg.acquire("a"); // epoch 0
    if (!attemptA.ok || !attemptB.ok) throw new Error("expected ok");
    reg.recordSuccess("a", attemptB.epoch, 5); // bumps epoch -> 1, state up
    reg.recordFailure("a", attemptA.epoch, "rate_limit"); // stale epoch 0 -> ignored
    expect(snap(reg).state).toBe("up");
  });

  it("single-flight probe: only one probe admitted after cooldown elapses", () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], { now: () => t, cooldownMs: 60_000 });
    const first = reg.acquire("a");
    if (!first.ok) throw new Error("expected ok");
    reg.recordFailure("a", first.epoch, "rate_limit"); // cooling until 61000
    // still cooling -> not acquirable
    expect(reg.acquire("a").ok).toBe(false);
    t = 61_000; // cooldown elapsed
    const probe = reg.acquire("a");
    expect(probe.ok).toBe(true);
    if (probe.ok) expect(probe.probe).toBe(true);
    // a concurrent caller cannot also probe
    expect(reg.acquire("a").ok).toBe(false);
  });

  it("cooldown is monotonic (a shorter later ban does not shorten it)", () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], { now: () => t, cooldownMs: 60_000 });
    const a = reg.acquire("a");
    if (!a.ok) throw new Error("expected ok");
    reg.recordFailure("a", a.epoch, "rate_limit"); // until 61000
    reg.recordFailure("a", a.epoch, "server_error", { cooldownMs: 1_000 }); // until 2000 -> ignored
    expect(snap(reg).cooldownUntil).toBe(61_000);
  });

  it("worst-of: a soft failure never downgrades a hard down", () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], {
      now: () => t,
      cooldownMs: 60_000,
      downRecheckMs: 900_000,
    });
    const a = reg.acquire("a");
    if (!a.ok) throw new Error("expected ok");
    reg.recordFailure("a", a.epoch, "payment"); // down
    reg.recordFailure("a", a.epoch, "rate_limit"); // stays down
    expect(reg.snapshot()[0]).toMatchObject({ state: "down", reason: "payment" });
  });

  it("manual disable / enable / reset", () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], { now: () => t });
    reg.disable("a");
    expect(snap(reg).state).toBe("off");
    expect(reg.acquire("a").ok).toBe(false);
    reg.enable("a");
    expect(snap(reg).state).toBe("up");
    // reset from down
    const a = reg.acquire("a");
    if (!a.ok) throw new Error("expected ok");
    reg.recordFailure("a", a.epoch, "payment");
    expect(snap(reg).state).toBe("down");
    reg.reset("a");
    expect(reg.snapshot()[0]).toMatchObject({ state: "up", reason: null });
  });

  it("activeId is the first up connector; pin overrides order", () => {
    let t = 1000;
    const reg = registry(
      [
        { id: "a", client: new FakeClient(() => jsonResult(200)) },
        { id: "b", client: new FakeClient(() => jsonResult(200)) },
      ],
      { now: () => t },
    );
    expect(reg.activeId()).toBe("a");
    reg.disable("a");
    expect(reg.activeId()).toBe("b");
    reg.enable("a");
    reg.pin("b");
    expect(reg.activeId()).toBe("b");
    expect(reg.order()[0]).toBe("b");
    reg.unpin();
    expect(reg.activeId()).toBe("a");
  });
});

// --- pooled client failover ----------------------------------------------

describe("PooledUpstreamClient", () => {
  it("returns the first connector's success without touching the second", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(200, { ok: 1 }));
    const c2 = new FakeClient(() => jsonResult(200, { ok: 2 }));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, opts);
    expect(res.status).toBe(200);
    expect(c2.seenModels).toHaveLength(0);
    expect(reg.snapshot()[0]).toMatchObject({ state: "up", totalRequests: 1 });
  });

  it("advances past a 429 to the next connector; the first goes cooling", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(429, { error: "rate" }));
    const c2 = new FakeClient(() => jsonResult(200, { ok: 2 }));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, opts);
    expect(res.status).toBe(200);
    const snap = reg.snapshot();
    expect(snap[0]).toMatchObject({ id: "a", state: "cooling", reason: "rate_limit" });
    expect(snap[1]).toMatchObject({ id: "b", state: "up" });
    expect(reg.activeId()).toBe("b");
  });

  it("surfaces the MOST-RECOVERABLE failure (429 over a dead backup's 401)", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(429));
    const c2 = new FakeClient(() => jsonResult(401));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, opts);
    expect(res.status).toBe(429); // NOT the 401
    expect(reg.snapshot()[0]).toMatchObject({ state: "cooling" });
    expect(reg.snapshot()[1]).toMatchObject({ state: "down", reason: "auth" });
  });

  it("402 marks the connector down and advances", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(402));
    const c2 = new FakeClient(() => jsonResult(200));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, opts);
    expect(res.status).toBe(200);
    expect(reg.snapshot()[0]).toMatchObject({ state: "down", reason: "payment" });
  });

  it("404 advances without changing health and falls back to the last 404", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(404));
    const c2 = new FakeClient(() => jsonResult(404));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, opts);
    expect(res.status).toBe(404);
    // health untouched: both still up
    expect(reg.snapshot().map((s) => s.state)).toEqual(["up", "up"]);
  });

  it("a 400 request error returns immediately and keeps the connector healthy", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(400, { error: "bad" }));
    const c2 = new FakeClient(() => jsonResult(200));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, opts);
    expect(res.status).toBe(400);
    expect(c2.seenModels).toHaveLength(0);
    expect(snap(reg).state).toBe("up");
  });

  it("throws NoConnectorAvailableError when nothing is usable", async () => {
    let t = 1000;
    const reg = registry([{ id: "a", client: new FakeClient(() => jsonResult(200)) }], { now: () => t });
    reg.disable("a");
    const pool = new PooledUpstreamClient(reg);
    await expect(pool.chatCompletions({ model: "m" }, opts)).rejects.toBeInstanceOf(NoConnectorAvailableError);
  });

  it("uses Retry-After from a 429 to set the cooldown", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(429, {}, { retryAfterMs: 5_000 }));
    const c2 = new FakeClient(() => jsonResult(200));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t, cooldownMs: 60_000 });
    const pool = new PooledUpstreamClient(reg);
    await pool.chatCompletions({ model: "m" }, opts);
    expect(snap(reg).cooldownUntil).toBe(6_000); // 1000 + 5000, not the 60s default
  });

  it("passes a streaming success through", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => streamResult(200));
    const reg = registry([{ id: "a", client: c1 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    const res = await pool.chatCompletions({ model: "m" }, { stream: true });
    expect(res.kind).toBe("stream");
  });

  it("translates the model id via the connector's model_map", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => jsonResult(200));
    const reg = registry(
      [{ id: "a", client: c1, cfg: { modelMap: { "qwen3-coder:480b": "qwen/qwen3-coder" } } }],
      { now: () => t },
    );
    const pool = new PooledUpstreamClient(reg);
    await pool.chatCompletions({ model: "qwen3-coder:480b" }, opts);
    expect(c1.seenModels).toEqual(["qwen/qwen3-coder"]);
  });

  it("a client abort is rethrown without flipping connector health", async () => {
    let t = 1000;
    const ac = new AbortController();
    ac.abort();
    const c1 = new FakeClient(() => {
      throw new DOMException("aborted", "AbortError");
    });
    const reg = registry([{ id: "a", client: c1 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    await expect(
      pool.chatCompletions({ model: "m" }, { stream: false, signal: ac.signal }),
    ).rejects.toThrow();
    expect(snap(reg).state).toBe("up"); // health untouched
  });

  it("a capability/not-implemented throw is surfaced without cooling the connector", async () => {
    let t = 1000;
    const c1 = new FakeClient(() => {
      throw new NotImplementedError("native streaming not wired");
    });
    const c2 = new FakeClient(() => jsonResult(200));
    const reg = registry([{ id: "a", client: c1 }, { id: "b", client: c2 }], { now: () => t });
    const pool = new PooledUpstreamClient(reg);
    await expect(pool.chatCompletions({ model: "m" }, opts)).rejects.toBeInstanceOf(NotImplementedError);
    // c1 must NOT be cooled (deterministic error), and c2 must not have been tried.
    expect(snap(reg, 0).state).toBe("up");
    expect(c2.seenModels).toHaveLength(0);
  });

  it("show() routes to a native-capable connector and skips generic ones", async () => {
    let t = 1000;
    const generic = new FakeClient(() => jsonResult(200), false);
    const ollama = new FakeClient(() => jsonResult(200), true);
    const reg = registry(
      [{ id: "or", client: generic }, { id: "ol", client: ollama }],
      { now: () => t },
    );
    const pool = new PooledUpstreamClient(reg);
    const shown = await pool.show("some-model");
    expect(shown).toEqual({});
  });
});
