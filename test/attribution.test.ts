import { describe, it, expect } from "vitest";
import pino from "pino";
import type { Logger } from "pino";
import { OllamaClient } from "../src/upstream/ollama";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createResilience } from "../src/concurrency";
import type { Resilience } from "../src/concurrency";
import { singleStrategy } from "../src/strategies/single";
import { createFusionStrategy } from "../src/strategies/fusion";
import type { TimerFactory } from "../src/strategies/fusion";
import { jsonResponse } from "./helpers";
import type { ChatCompletionRequest, FetchFn, StrategyContext, UpstreamClient } from "../src/types";

/**
 * Part 1 — per-upstream-call error attribution. Every upstream failure (error,
 * timeout, 429, circuit-open) must carry: `stage`, `upstream_model`, `err_kind`,
 * `latency_ms`, and `status` when the upstream answered.
 */

interface Captured {
  logger: Logger;
  lines: Array<Record<string, unknown>>;
}

/** A pino logger whose output is captured as parsed JSON objects. */
function captureLogger(): Captured {
  const lines: Array<Record<string, unknown>> = [];
  const logger = pino(
    { level: "warn", base: undefined },
    {
      write(s: string) {
        lines.push(JSON.parse(s));
      },
    },
  );
  return { logger, lines };
}

/** All `upstream call failed` attribution lines emitted so far. */
function failures(cap: Captured): Array<Record<string, unknown>> {
  return cap.lines.filter((l) => l.msg === "upstream call failed");
}

const config = parseConfig({
  upstream: { base_url: "https://mock.test", api_key_env: "X", max_concurrency: 4 },
  models: {
    "fast-glm": { strategy: "single", target: "glm-5.2" },
    "fusion-1": { strategy: "fusion", panel: ["m1", "m2", "m3"], judge: "j", synth: "s" },
  },
});

function fetchOnce(respond: (url: string, init?: RequestInit) => Response): FetchFn {
  return async (input, init) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    return respond(url, init);
  };
}

function singleCtx(
  client: UpstreamClient,
  request: ChatCompletionRequest,
  logger: Logger,
  resilience: Resilience,
): StrategyContext {
  const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
  const entry = config.models["fast-glm"];
  if (!entry) throw new Error("missing fast-glm");
  return { request, config, client, capabilities, logger, modelConfig: entry, resilience };
}

function freshResilience(failureThreshold?: number): Resilience {
  return createResilience({
    maxConcurrency: 4,
    now: () => 1_000_000,
    sleep: async () => {},
    failureThreshold,
  });
}

describe("per-call error attribution", () => {
  it("attributes a 429 with stage, upstream model, status, err_kind, and latency_ms (single)", async () => {
    const cap = captureLogger();
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: fetchOnce(() => jsonResponse({ error: "rate limited" }, 429)),
    });
    await singleStrategy.execute(
      singleCtx(client, { model: "fast-glm", messages: [] }, cap.logger, freshResilience()),
    );
    const f = failures(cap);
    expect(f).toHaveLength(1);
    expect(f[0]).toMatchObject({
      stage: "single",
      upstream_model: "glm-5.2",
      err_kind: "rate_limit",
      status: 429,
    });
    expect(typeof f[0]?.latency_ms).toBe("number");
  });

  it("attributes a network error with no status (single)", async () => {
    const cap = captureLogger();
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: fetchOnce(() => {
        throw new Error("connection refused");
      }),
    });
    await expect(
      singleStrategy.execute(
        singleCtx(client, { model: "fast-glm", messages: [] }, cap.logger, freshResilience()),
      ),
    ).rejects.toBeTruthy();
    const f = failures(cap);
    expect(f).toHaveLength(1);
    expect(f[0]).toMatchObject({ stage: "single", upstream_model: "glm-5.2", err_kind: "error" });
    expect(f[0]?.status).toBeUndefined();
    expect(typeof f[0]?.latency_ms).toBe("number");
  });

  it("attributes a circuit-open skip with latency_ms 0 and no status (single)", async () => {
    const cap = captureLogger();
    const resilience = freshResilience(1); // 1 failure trips the breaker open
    resilience.breaker.recordFailure("glm-5.2");
    const client = new OllamaClient({
      baseUrl: "https://mock.test",
      apiKey: "k",
      fetchFn: fetchOnce(() => jsonResponse({ ok: true })),
    });
    await expect(
      singleStrategy.execute(singleCtx(client, { model: "fast-glm", messages: [] }, cap.logger, resilience)),
    ).rejects.toMatchObject({ httpStatus: 503 });
    const f = failures(cap);
    expect(f).toHaveLength(1);
    expect(f[0]).toMatchObject({
      stage: "single",
      upstream_model: "glm-5.2",
      err_kind: "circuit_open",
      latency_ms: 0,
    });
    expect(f[0]?.status).toBeUndefined();
  });

  it("attributes a slow panel member as a timeout with stage 'panel' (fusion)", async () => {
    const cap = captureLogger();
    const fastTimer: TimerFactory = () => {
      let h: ReturnType<typeof setTimeout>;
      const expired = new Promise<void>((resolve) => {
        h = setTimeout(resolve, 5);
      });
      return { expired, cancel: () => clearTimeout(h) };
    };
    const fetchFn: FetchFn = async (input, init) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      if (url.endsWith("/api/show")) return jsonResponse({ capabilities: ["completion"], model_info: {} });
      const body: { model: string } = JSON.parse(String(init?.body));
      if (body.model === "m2") return new Promise<Response>(() => {}); // never resolves -> timeout
      return jsonResponse({ choices: [{ message: { content: `ans-${body.model}` } }] });
    };
    const client = new OllamaClient({ baseUrl: "https://mock.test", apiKey: "k", fetchFn });
    const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger: cap.logger });
    const entry = config.models["fusion-1"];
    if (!entry) throw new Error("missing fusion-1");
    const testConfig = {
      ...config,
      defaults: {
        ...config.defaults,
        min_panel_success: 3,
      },
    };
    const strategy = createFusionStrategy({ timer: fastTimer });
    try {
      await strategy.execute({
        request: { model: "fusion-1", messages: [{ role: "user", content: "hi" }] },
        config: testConfig,
        client,
        capabilities,
        logger: cap.logger,
        modelConfig: entry,
      });
    } catch (err) {
      // Expected AllMembersFailedError
    }
    const timeout = failures(cap).find((l) => l.upstream_model === "m2");
    expect(timeout).toMatchObject({ stage: "panel", upstream_model: "m2", err_kind: "timeout" });
    expect(typeof timeout?.latency_ms).toBe("number");
  });
});
