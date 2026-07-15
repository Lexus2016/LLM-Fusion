import { describe, it, expect } from "vitest";
import type { MiddlewareHandler } from "hono";
import pino from "pino";
import { createPanelApp } from "../src/panel/routes";
import { createApp } from "../src/server";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createAuthMiddleware } from "../src/auth";
import {
  ConnectorRegistry,
  type ConnectorClient,
  type ResolvedConnector,
} from "../src/connectors/registry";
import type { ChatCompletionResult } from "../src/types";

const logger = pino({ level: "silent" });

class NoopClient implements ConnectorClient {
  readonly supportsNativeShow = false;
  async chatCompletions(): Promise<ChatCompletionResult> {
    return { kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
  }
  async chatNative(): Promise<ChatCompletionResult> {
    return { kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
  }
  async show(): Promise<unknown> {
    return {};
  }
}

function resolved(id: string): ResolvedConnector {
  return {
    id,
    provider: "ollama",
    baseUrl: `https://${id}.test`,
    host: `https://${id}.test`,
    hasKey: true,
    treat403As: "passthrough",
    quotaMarkers: [],
    modelMap: {},
  };
}

function makeRegistry(ids: string[]): ConnectorRegistry {
  return new ConnectorRegistry(
    ids.map((id) => ({ cfg: resolved(id), client: new NoopClient() })),
  );
}

const openAuth: MiddlewareHandler = async (_c, next) => {
  await next();
};

describe("panel routes", () => {
  it("GET /panel serves self-contained HTML", async () => {
    const app = createPanelApp({ registry: makeRegistry(["a"]), auth: openAuth, logger });
    const res = await app.request("/panel");
    expect(res.status).toBe(200);
    const html = await res.text();
    expect(html).toContain("<!doctype html>");
    expect(html).toContain("connectors");
    expect(html).not.toContain("api_key"); // no secrets in the shell
  });

  it("GET /admin/connectors returns a snapshot + activeId", async () => {
    const app = createPanelApp({ registry: makeRegistry(["a", "b"]), auth: openAuth, logger });
    const res = await app.request("/admin/connectors");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.connectors).toHaveLength(2);
    expect(body.activeId).toBe("a");
    expect(body.connectors[0]).toMatchObject({ id: "a", state: "up", active: true });
  });

  it("POST disable → connector off and active moves on", async () => {
    const reg = makeRegistry(["a", "b"]);
    const app = createPanelApp({ registry: reg, auth: openAuth, logger });
    const res = await app.request("/admin/connectors/a/disable", { method: "POST" });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.activeId).toBe("b");
    expect(body.connectors[0]).toMatchObject({ id: "a", state: "off" });
  });

  it("POST reset revives a downed connector; pin makes it active", async () => {
    const reg = makeRegistry(["a", "b"]);
    const acq = reg.acquire("a");
    if (!acq.ok) throw new Error("expected ok");
    reg.recordFailure("a", acq.epoch, "payment"); // a is down
    const app = createPanelApp({ registry: reg, auth: openAuth, logger });

    await app.request("/admin/connectors/a/reset", { method: "POST" });
    expect(reg.snapshot().find((s) => s.id === "a")?.state).toBe("up");

    const res = await app.request("/admin/connectors/b/pin", { method: "POST" });
    const body = await res.json();
    expect(body.activeId).toBe("b");
    await app.request("/admin/unpin", { method: "POST" });
    expect(reg.activeId()).toBe("a");
  });

  it("unknown action → 400, unknown connector → 404", async () => {
    const app = createPanelApp({ registry: makeRegistry(["a"]), auth: openAuth, logger });
    expect((await app.request("/admin/connectors/a/frobnicate", { method: "POST" })).status).toBe(400);
    expect((await app.request("/admin/connectors/zzz/disable", { method: "POST" })).status).toBe(404);
  });

  it("admin routes are auth-gated; the HTML shell is not", async () => {
    const auth = createAuthMiddleware(() => "secret-token");
    const app = createPanelApp({ registry: makeRegistry(["a"]), auth, logger });

    // HTML shell is ungated (no secrets in it) so a browser can load it.
    expect((await app.request("/panel")).status).toBe(200);

    // Data + actions require the token.
    expect((await app.request("/admin/connectors")).status).toBe(401);
    const ok = await app.request("/admin/connectors", {
      headers: { authorization: "Bearer secret-token" },
    });
    expect(ok.status).toBe(200);
    expect((await app.request("/admin/connectors/a/disable", { method: "POST" })).status).toBe(401);
  });

  it("/ready reflects the connector pool (up = ready, all off = degraded)", async () => {
    const reg = makeRegistry(["a", "b"]);
    const config = parseConfig({
      upstream: {},
      connectors: [
        { id: "a", base_url: "https://a.test", api_key_env: "K1" },
        { id: "b", base_url: "https://b.test", api_key_env: "K2" },
      ],
      models: { m: { strategy: "single", target: "x" } },
    });
    const client = new NoopClient();
    const capabilities = new CapabilityService({ client, getOverrides: () => config.overrides, logger });
    const app = createApp({
      getConfig: () => config,
      client,
      capabilities,
      getAuthToken: () => undefined,
      logger,
      registry: reg,
    });
    expect((await app.request("/ready")).status).toBe(200); // both up
    reg.disable("a");
    reg.disable("b");
    expect((await app.request("/ready")).status).toBe(503); // none up
  });
});
