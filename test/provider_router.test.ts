import { describe, it, expect } from "vitest";
import pino from "pino";
import { parseConfig } from "../src/config";
import { resolveProviders } from "../src/connectors/resolve";
import { ProviderRouter } from "../src/connectors/provider_router";
import { dispatch } from "../src/router";
import type { CapabilityProvider, FetchFn, DiscoveryResult } from "../src/types";

const logger = pino({ level: "silent" });

const caps: CapabilityProvider = {
  async discover(): Promise<DiscoveryResult> {
    return { capability: { vision: false, tools: true, context: null }, source: "default" };
  },
  clear() {},
};

// Tag each response by the host it was sent to, so we can prove which provider
// group actually served a virtual model.
const taggingFetch: FetchFn = async (input) => {
  const host = new URL(String(input)).host;
  const tag = host.includes("ollama") ? "ollama-cloud" : "openrouter";
  return new Response(
    JSON.stringify({ served_by: tag, choices: [{ index: 0, message: { role: "assistant", content: "ok" } }] }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
};

const config = parseConfig({
  upstream: {},
  providers: {
    "ollama-cloud": { type: "ollama", base_url: "https://ollama.test", accounts: [{ id: "o1", api_key_env: "K1" }] },
    openrouter: { type: "openai-compat", base_url: "https://openrouter.test", accounts: [{ id: "r1", api_key_env: "K2" }] },
  },
  models: {
    "m-ollama": { strategy: "single", provider: "ollama-cloud", target: "glm-5.2" },
    "m-router": { strategy: "single", provider: "openrouter", target: "qwen/qwen3-coder" },
  },
});

function router() {
  const groups = resolveProviders(config, { env: { K1: "k1", K2: "k2" }, fetchFn: taggingFetch });
  return new ProviderRouter(groups, {});
}

async function servedBy(r: ProviderRouter, model: string): Promise<{ status: number; served?: string }> {
  const res = await dispatch({
    request: { model, messages: [{ role: "user", content: "hi" }] },
    config,
    client: r.defaultPool,
    router: r,
    capabilities: caps,
    logger,
  });
  const text = await res.text();
  try {
    return { status: res.status, served: JSON.parse(text).served_by };
  } catch {
    return { status: res.status };
  }
}

describe("provider-group routing", () => {
  it("routes each virtual model to the pool of the provider group it is bound to", async () => {
    const r = router();
    expect(await servedBy(r, "m-ollama")).toMatchObject({ status: 200, served: "ollama-cloud" });
    expect(await servedBy(r, "m-router")).toMatchObject({ status: 200, served: "openrouter" });
  });

  it("failover stays WITHIN a group — a downed group never leaks to another provider", async () => {
    const r = router();
    // Take the ollama group's only account offline.
    r.registryForAccount("o1")?.disable("o1");
    // m-ollama must NOT be served by openrouter; it fails (no usable account).
    const out = await servedBy(r, "m-ollama").catch(() => ({ status: 503, served: undefined as string | undefined }));
    expect(out.served).not.toBe("openrouter");
    // m-router is unaffected.
    expect(await servedBy(r, "m-router")).toMatchObject({ status: 200, served: "openrouter" });
  });

  it("poolFor returns the sole group's pool when no group is named", () => {
    const single = new ProviderRouter(
      resolveProviders(
        parseConfig({
          upstream: {},
          providers: { only: { type: "ollama", base_url: "https://ollama.test", accounts: [{ id: "x", api_key_env: "K1" }] } },
          models: { m: { strategy: "single", target: "glm-5.2" } },
        }),
        { env: { K1: "k1" }, fetchFn: taggingFetch },
      ),
      {},
    );
    expect(single.soleGroupId).toBe("only");
    expect(() => single.poolFor(undefined)).not.toThrow();
  });
});
