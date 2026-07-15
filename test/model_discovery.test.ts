import { describe, it, expect } from "vitest";
import type { MiddlewareHandler } from "hono";
import pino from "pino";
import { OpenAiCompatClient, parseModelList } from "../src/upstream/openai_compat";
import { OllamaClient } from "../src/upstream/ollama";
import { ProviderRouter } from "../src/connectors/provider_router";
import { createConfigEditorApp } from "../src/panel/config_editor";
import { parseConfig } from "../src/config";
import type { ResolvedGroup } from "../src/connectors/resolve";
import type { ConnectorClient, ResolvedConnector } from "../src/connectors/registry";
import type { ChatCompletionResult, FetchFn } from "../src/types";

const logger = pino({ level: "silent" });
const openAuth: MiddlewareHandler = async (_c, next) => {
  await next();
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("parseModelList", () => {
  it("reads the OpenAI-compat shape { data: [{ id }] }, sorted + deduped", () => {
    const out = parseModelList({ object: "list", data: [{ id: "kimi-k2.7" }, { id: "glm-5.2" }, { id: "glm-5.2" }] });
    expect(out).toEqual(["glm-5.2", "kimi-k2.7"]);
  });

  it("reads the Ollama /api/tags shape { models: [{ name }] }", () => {
    const out = parseModelList({ models: [{ name: "glm-5.2:latest", model: "glm-5.2" }, { name: "qwen3" }] });
    expect(out).toEqual(["glm-5.2:latest", "qwen3"]);
  });

  it("tolerates a malformed body → empty list", () => {
    expect(parseModelList(null)).toEqual([]);
    expect(parseModelList("nope")).toEqual([]);
    expect(parseModelList({ data: "not-an-array" })).toEqual([]);
  });
});

describe("OpenAiCompatClient.listModels", () => {
  it("GETs /v1/models with auth and returns the ids", async () => {
    let seen: { url: string; auth?: string } | null = null;
    const fetchFn: FetchFn = async (input, init) => {
      const headers = new Headers(init?.headers);
      seen = { url: String(input), auth: headers.get("authorization") ?? undefined };
      return json({ data: [{ id: "b-model" }, { id: "a-model" }] });
    };
    const client = new OpenAiCompatClient({ baseUrl: "https://prov.test/api/v1", apiKey: "sk-secret", fetchFn });
    const models = await client.listModels();
    expect(models).toEqual(["a-model", "b-model"]);
    expect(seen!.url).toBe("https://prov.test/api/v1/v1/models");
    expect(seen!.auth).toBe("Bearer sk-secret");
  });

  it("throws on a non-OK response", async () => {
    const fetchFn: FetchFn = async () => new Response("nope", { status: 500 });
    const client = new OpenAiCompatClient({ baseUrl: "https://prov.test", apiKey: "k", fetchFn });
    await expect(client.listModels()).rejects.toThrow(/models failed/);
  });
});

describe("OllamaClient.listModels", () => {
  it("uses /v1/models when available", async () => {
    const fetchFn: FetchFn = async (input) => {
      if (String(input).endsWith("/v1/models")) return json({ data: [{ id: "glm-5.2" }] });
      return new Response("x", { status: 500 });
    };
    const client = new OllamaClient({ baseUrl: "https://ollama.com", apiKey: "k", fetchFn });
    expect(await client.listModels()).toEqual(["glm-5.2"]);
  });

  it("falls back to /api/tags when /v1/models fails", async () => {
    const hits: string[] = [];
    const fetchFn: FetchFn = async (input) => {
      const url = String(input);
      hits.push(url);
      if (url.endsWith("/v1/models")) return new Response("no openai surface", { status: 404 });
      if (url.endsWith("/api/tags")) return json({ models: [{ name: "kimi", model: "kimi" }, { name: "glm-5.2" }] });
      return new Response("x", { status: 500 });
    };
    const client = new OllamaClient({ baseUrl: "https://ollama.com", apiKey: "k", fetchFn });
    expect(await client.listModels()).toEqual(["glm-5.2", "kimi"]);
    expect(hits.some((u) => u.endsWith("/v1/models"))).toBe(true);
    expect(hits.some((u) => u.endsWith("/api/tags"))).toBe(true);
  });
});

class FakeClient implements ConnectorClient {
  readonly supportsNativeShow = false;
  constructor(private readonly models: string[]) {}
  async chatCompletions(): Promise<ChatCompletionResult> {
    return { kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
  }
  async chatNative(): Promise<ChatCompletionResult> {
    return { kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
  }
  async show(): Promise<unknown> {
    return {};
  }
  async listModels(): Promise<string[]> {
    return this.models;
  }
}

function resolved(id: string, group: string): ResolvedConnector {
  return {
    id,
    group,
    provider: "openai-compat",
    baseUrl: `https://${id}.test`,
    host: `https://${id}.test`,
    hasKey: true,
    treat403As: "passthrough",
    quotaMarkers: [],
    modelMap: {},
  };
}

describe("ProviderRouter.listGroupModels", () => {
  it("routes to a keyed account's client and returns its catalog", async () => {
    const g: ResolvedGroup = {
      id: "g1",
      type: "openai-compat",
      accounts: [{ cfg: resolved("a", "g1"), client: new FakeClient(["m1", "m2"]) }],
    };
    const router = new ProviderRouter([g]);
    expect(await router.listGroupModels("g1")).toEqual(["m1", "m2"]);
  });

  it("rejects an unknown group", async () => {
    const router = new ProviderRouter([
      { id: "g1", type: "openai-compat", accounts: [{ cfg: resolved("a", "g1"), client: new FakeClient([]) }] },
    ]);
    await expect(router.listGroupModels("nope")).rejects.toThrow(/unknown provider group/);
  });
});

function editor(listProviderModels?: ConfigEditorListModels) {
  const cfg = parseConfig({
    upstream: {},
    providers: { g1: { type: "openai-compat", base_url: "https://x.test", accounts: [{ id: "a", api_key_env: "K" }] } },
    models: { m: { strategy: "single", target: "t" } },
  });
  return createConfigEditorApp({
    getConfig: () => cfg,
    configPath: "/does/not/matter.yaml",
    auth: openAuth,
    logger,
    envHas: () => true,
    listProviderModels,
  });
}
type ConfigEditorListModels = (groupId: string, opts: { signal?: AbortSignal }) => Promise<string[]>;

describe("config editor — GET /admin/config/providers/:id/models", () => {
  it("returns the provider catalog and caches it (upstream hit once)", async () => {
    let calls = 0;
    const app = editor(async () => {
      calls += 1;
      return ["glm-5.2", "kimi-k2.7"];
    });
    const first = await app.request("/admin/config/providers/g1/models");
    expect(first.status).toBe(200);
    expect((await first.json()).models).toEqual(["glm-5.2", "kimi-k2.7"]);
    const second = await app.request("/admin/config/providers/g1/models");
    expect((await second.json()).cached).toBe(true);
    expect(calls).toBe(1);
  });

  it("degrades to an empty list + note on upstream failure (never blocks the form)", async () => {
    const app = editor(async () => {
      throw new Error("provider unreachable");
    });
    const res = await app.request("/admin/config/providers/g1/models");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.models).toEqual([]);
    expect(body.note).toMatch(/unreachable/);
  });

  it("reports when live discovery is not wired", async () => {
    const app = editor(undefined);
    const res = await app.request("/admin/config/providers/g1/models");
    expect(res.status).toBe(200);
    expect((await res.json()).note).toMatch(/not available/);
  });
});
