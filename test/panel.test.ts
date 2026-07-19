import { describe, it, expect } from "vitest";
import type { MiddlewareHandler } from "hono";
import pino from "pino";
import { createPanelApp } from "../src/panel/routes";
import { createApp } from "../src/server";
import { CapabilityService } from "../src/capabilities";
import { parseConfig } from "../src/config";
import { createAuthMiddleware } from "../src/auth";
import { ProviderRouter } from "../src/connectors/provider_router";
import type { ResolvedGroup } from "../src/connectors/resolve";
import type { ConnectorClient, ResolvedConnector } from "../src/connectors/registry";
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

function resolved(id: string, group: string): ResolvedConnector {
  return {
    id,
    group,
    provider: "ollama",
    baseUrl: `https://${id}.test`,
    host: `https://${id}.test`,
    hasKey: true,
    treat403As: "passthrough",
    quotaMarkers: [],
    modelMap: {},
  };
}

function makeRouter(ids: string[], group = "g1"): ProviderRouter {
  const g: ResolvedGroup = {
    id: group,
    type: "ollama",
    accounts: ids.map((id) => ({ cfg: resolved(id, group), client: new NoopClient() })),
  };
  return new ProviderRouter([g]);
}

const openAuth: MiddlewareHandler = async (_c, next) => {
  await next();
};

describe("panel routes", () => {
  it("GET /panel serves self-contained HTML", async () => {
    const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
    const res = await app.request("/panel");
    expect(res.status).toBe(200);
    const html = await res.text();
    expect(html).toContain("<!doctype html>");
    expect(html).toContain("admin/providers");
    // The shell carries env-var NAMES only, never a secret key VALUE.
    expect(html).not.toMatch(/sk-[A-Za-z0-9]{6}/);
  });

  it("GET /admin/providers returns grouped snapshot", async () => {
    const app = createPanelApp({ router: makeRouter(["a", "b"]), auth: openAuth, logger, authEnforced: () => true });
    const res = await app.request("/admin/providers");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.providers).toHaveLength(1);
    expect(body.providers[0]).toMatchObject({ id: "g1", type: "ollama", activeId: "a" });
    expect(body.providers[0].accounts).toHaveLength(2);
    expect(body.providers[0].accounts[0]).toMatchObject({ id: "a", group: "g1", state: "up", active: true });
  });

  it("POST disable → account off and the group's active moves on", async () => {
    const router = makeRouter(["a", "b"]);
    const app = createPanelApp({ router, auth: openAuth, logger, authEnforced: () => true });
    const res = await app.request("/admin/connectors/a/disable", { method: "POST" });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.providers[0].activeId).toBe("b");
    expect(body.providers[0].accounts[0]).toMatchObject({ id: "a", state: "off" });
  });

  it("POST reset revives a downed account; pin makes it active; unpin clears it", async () => {
    const router = makeRouter(["a", "b"]);
    const reg = router.registryForAccount("a");
    if (!reg) throw new Error("registry not found");
    const acq = reg.acquire("a");
    if (!acq.ok) throw new Error("expected ok");
    reg.recordFailure("a", acq.epoch, "payment"); // a is down
    const app = createPanelApp({ router, auth: openAuth, logger, authEnforced: () => true });

    await app.request("/admin/connectors/a/reset", { method: "POST" });
    expect(reg.snapshot().find((s) => s.id === "a")?.state).toBe("up");

    const res = await app.request("/admin/connectors/b/pin", { method: "POST" });
    const body = await res.json();
    expect(body.providers[0].activeId).toBe("b");
    await app.request("/admin/connectors/b/unpin", { method: "POST" });
    expect(reg.activeId()).toBe("a");
  });

  it("unknown action → 400, unknown account → 404", async () => {
    const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger, authEnforced: () => true });
    expect((await app.request("/admin/connectors/a/frobnicate", { method: "POST" })).status).toBe(400);
    expect((await app.request("/admin/connectors/zzz/disable", { method: "POST" })).status).toBe(404);
  });

  describe("admin origin/content-type guard", () => {
    it("passes a loopback request with no Origin (curl/scripts)", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/providers", { headers: { host: "127.0.0.1:8080" } });
      expect(res.status).toBe(200);
    });

    it("403 (fail-closed) when auth is off and the Host header is absent", async () => {
      // A missing Host must not slip past the loopback pin — else a non-browser
      // client (HTTP/1.0, curl --http1.0) bypasses it on an ALLOW_OPEN deployment.
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/providers");
      expect(res.status).toBe(403);
      expect((await res.json()).error).toMatch(/loopback/);
    });

    it("403 when Origin host does not match Host", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/providers", {
        headers: { origin: "http://evil.example.com", host: "127.0.0.1:8080" },
      });
      expect(res.status).toBe(403);
      expect((await res.json()).error).toMatch(/origin/);
    });

    it("passes when Origin matches Host", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/providers", {
        headers: { origin: "http://127.0.0.1:8080", host: "127.0.0.1:8080" },
      });
      expect(res.status).toBe(200);
    });

    it("403 when auth is off and Host is not a loopback name (DNS-rebinding)", async () => {
      // The rebinding vector: evil.tld rebinds to 127.0.0.1, so Origin==Host and
      // the content-type is allowed — only pinning Host to loopback stops it.
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/providers", {
        headers: { origin: "http://evil.example.com:8080", host: "evil.example.com:8080" },
      });
      expect(res.status).toBe(403);
      expect((await res.json()).error).toMatch(/loopback/);
    });

    it("does NOT falsely pass a `127.`-prefixed non-loopback Host (F3 class)", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/providers", {
        headers: { origin: "http://127.evil.com:8080", host: "127.evil.com:8080" },
      });
      expect(res.status).toBe(403);
    });

    it("allows a non-loopback Host when auth IS enforced (front-with-your-own-auth)", async () => {
      const app = createPanelApp({
        router: makeRouter(["a"]),
        auth: openAuth,
        logger,
        authEnforced: () => true,
      });
      const res = await app.request("/admin/providers", {
        headers: { origin: "https://proxy.internal", host: "proxy.internal" },
      });
      expect(res.status).toBe(200);
    });

    it("415 on a mutating request with a non-JSON body", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/connectors/a/disable", {
        method: "POST",
        headers: { host: "127.0.0.1:8080", "content-type": "text/plain" },
        body: "action=disable",
      });
      expect(res.status).toBe(415);
    });

    it("415 on a chunked body with no content-type (transfer-encoding bypass)", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/connectors/a/disable", {
        method: "POST",
        headers: { host: "127.0.0.1:8080", "transfer-encoding": "chunked" },
      });
      expect(res.status).toBe(415);
    });

    it("allows bodyless mutations without a content-type", async () => {
      const app = createPanelApp({ router: makeRouter(["a"]), auth: openAuth, logger });
      const res = await app.request("/admin/connectors/a/disable", { method: "POST", headers: { host: "127.0.0.1:8080" } });
      expect(res.status).toBe(200);
    });
  });

  it("admin routes are auth-gated; the HTML shell is not", async () => {
    const auth = createAuthMiddleware(() => "secret-token");
    // auth IS enforced here, so the loopback Host pin is off (the token is the gate).
    const app = createPanelApp({ router: makeRouter(["a"]), auth, logger, authEnforced: () => true });

    expect((await app.request("/panel")).status).toBe(200); // ungated shell
    expect((await app.request("/admin/providers")).status).toBe(401);
    const ok = await app.request("/admin/providers", { headers: { authorization: "Bearer secret-token" } });
    expect(ok.status).toBe(200);
    expect((await app.request("/admin/connectors/a/disable", { method: "POST" })).status).toBe(401);
  });

  it("/ready reflects the provider pool (up = ready, all off = degraded)", async () => {
    const router = makeRouter(["a", "b"]);
    const config = parseConfig({
      upstream: {},
      providers: { g1: { type: "ollama", base_url: "https://g1.test", accounts: [{ id: "a", api_key_env: "K1" }, { id: "b", api_key_env: "K2" }] } },
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
      router,
    });
    expect((await app.request("/ready")).status).toBe(200); // both up
    const reg = router.registryForAccount("a");
    reg?.disable("a");
    reg?.disable("b");
    expect((await app.request("/ready")).status).toBe(503); // none up
  });
});
