import { describe, it, expect } from "vitest";
import { parseConfig } from "../src/config";
import { connectorDefs, resolveConnectors } from "../src/connectors/resolve";
import type { FetchFn } from "../src/types";

const models = { models: { m: { strategy: "single", target: "x" } } };
const stubFetch: FetchFn = async () => new Response("{}");

function at<T>(arr: T[], i: number): T {
  const v = arr[i];
  if (v === undefined) throw new Error(`no element at ${i}`);
  return v;
}

describe("config: connectors", () => {
  it("accepts a connectors list and applies defaults", () => {
    const cfg = parseConfig({
      upstream: { max_concurrency: 4 },
      connectors: [
        { id: "a", provider: "ollama", base_url: "https://ollama.com", api_key_env: "K1" },
        {
          id: "b",
          provider: "openai-compat",
          base_url: "https://openrouter.ai/api/v1",
          api_key_env: "K2",
          model_map: { x: "y/z" },
          extra_headers: { "X-Title": "llm-fusion" },
        },
      ],
      ...models,
    });
    expect(cfg.connectors).toHaveLength(2);
    expect(at(cfg.connectors ?? [], 0).treat_403_as).toBe("passthrough");
    expect(at(cfg.connectors ?? [], 0).quota_markers).toEqual([]);
    expect(cfg.upstream.connector_cooldown_s).toBe(60);
    expect(cfg.upstream.connector_down_recheck_s).toBe(900);
  });

  it("backward compat: single upstream, no connectors → one synthesised connector", () => {
    const cfg = parseConfig({
      upstream: { base_url: "https://ollama.com", api_key_env: "OLLAMA_API_KEY" },
      ...models,
    });
    const defs = connectorDefs(cfg);
    expect(defs).toHaveLength(1);
    expect(at(defs, 0)).toMatchObject({
      id: "default",
      provider: "ollama",
      base_url: "https://ollama.com",
      api_key_env: "OLLAMA_API_KEY",
    });
  });

  it("rejects duplicate connector ids", () => {
    expect(() =>
      parseConfig({
        upstream: {},
        connectors: [
          { id: "a", base_url: "https://x.com", api_key_env: "K1" },
          { id: "a", base_url: "https://y.com", api_key_env: "K2" },
        ],
        ...models,
      }),
    ).toThrow(/duplicate connector id/);
  });

  it("rejects a config with no connector source at all", () => {
    expect(() => parseConfig({ upstream: { max_concurrency: 4 }, ...models })).toThrow(
      /no connectors configured/,
    );
  });

  it("resolveConnectors reads keys from env, derives host, and picks the client", () => {
    const cfg = parseConfig({
      upstream: {},
      connectors: [
        { id: "a", provider: "ollama", base_url: "https://ollama.com/", api_key_env: "PRESENT" },
        {
          id: "b",
          provider: "openai-compat",
          base_url: "https://openrouter.ai/api/v1",
          api_key_env: "MISSING",
          model_map: { x: "y" },
        },
      ],
      ...models,
    });
    const entries = resolveConnectors(cfg, { env: { PRESENT: "sk-1" }, fetchFn: stubFetch });
    expect(at(entries, 0).cfg).toMatchObject({ id: "a", host: "https://ollama.com", hasKey: true });
    expect(at(entries, 1).cfg).toMatchObject({ id: "b", hasKey: false, modelMap: { x: "y" } });
    expect(at(entries, 0).client.supportsNativeShow).toBe(true); // ollama
    expect(at(entries, 1).client.supportsNativeShow).toBe(false); // generic
  });
});
