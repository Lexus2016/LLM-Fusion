import { describe, it, expect } from "vitest";
import { parseConfig } from "../src/config";
import { groupDefs, resolveProviders } from "../src/connectors/resolve";
import type { FetchFn } from "../src/types";

const stubFetch: FetchFn = async () => new Response("{}");

function at<T>(arr: T[], i: number): T {
  const v = arr[i];
  if (v === undefined) throw new Error(`no element at ${i}`);
  return v;
}

describe("config: provider groups", () => {
  it("accepts a providers map and applies defaults", () => {
    const cfg = parseConfig({
      upstream: { max_concurrency: 4 },
      providers: {
        "ollama-cloud": {
          type: "ollama",
          base_url: "https://ollama.com",
          accounts: [
            { id: "acc-1", api_key_env: "K1" },
            { id: "acc-2", api_key_env: "K2" },
          ],
        },
        openrouter: {
          type: "openai-compat",
          base_url: "https://openrouter.ai/api/v1",
          accounts: [{ id: "or-1", api_key_env: "K3", model_map: { x: "y/z" } }],
        },
      },
      models: { m: { strategy: "single", provider: "ollama-cloud", target: "glm-5.2" } },
    });
    expect(Object.keys(cfg.providers ?? {})).toEqual(["ollama-cloud", "openrouter"]);
    const acc = (cfg.providers ?? {})["ollama-cloud"]?.accounts?.[0];
    expect(acc?.treat_403_as).toBe("passthrough");
    expect(acc?.quota_markers).toEqual([]);
  });

  it("backward compat: legacy single upstream → one synthesised `default` group", () => {
    const cfg = parseConfig({
      upstream: { base_url: "https://ollama.com", api_key_env: "OLLAMA_API_KEY" },
      models: { m: { strategy: "single", target: "glm-5.2" } },
    });
    const defs = groupDefs(cfg);
    expect(defs).toHaveLength(1);
    expect(at(defs, 0)).toMatchObject({ id: "default", type: "ollama", base_url: "https://ollama.com" });
    expect(at(defs, 0).accounts[0]).toMatchObject({ id: "default", api_key_env: "OLLAMA_API_KEY" });
  });

  it("a model with no `provider` is fine when there is one group, rejected when ambiguous", () => {
    // single group -> provider optional
    expect(() =>
      parseConfig({
        upstream: {},
        providers: { g1: { type: "ollama", base_url: "https://a.com", accounts: [{ id: "a", api_key_env: "K" }] } },
        models: { m: { strategy: "single", target: "x" } },
      }),
    ).not.toThrow();
    // two groups -> provider required
    expect(() =>
      parseConfig({
        upstream: {},
        providers: {
          g1: { type: "ollama", base_url: "https://a.com", accounts: [{ id: "a", api_key_env: "K1" }] },
          g2: { type: "openai-compat", base_url: "https://b.com", accounts: [{ id: "b", api_key_env: "K2" }] },
        },
        models: { m: { strategy: "single", target: "x" } },
      }),
    ).toThrow(/must set .*provider/);
  });

  it("rejects a model bound to an unknown provider group", () => {
    expect(() =>
      parseConfig({
        upstream: {},
        providers: { g1: { type: "ollama", base_url: "https://a.com", accounts: [{ id: "a", api_key_env: "K" }] } },
        models: { m: { strategy: "single", provider: "nope", target: "x" } },
      }),
    ).toThrow(/not defined in .*providers/);
  });

  it("rejects duplicate account ids across providers", () => {
    expect(() =>
      parseConfig({
        upstream: {},
        providers: {
          g1: { type: "ollama", base_url: "https://a.com", accounts: [{ id: "dup", api_key_env: "K1" }] },
          g2: { type: "ollama", base_url: "https://b.com", accounts: [{ id: "dup", api_key_env: "K2" }] },
        },
        models: { m: { strategy: "single", provider: "g1", target: "x" } },
      }),
    ).toThrow(/duplicate account id/);
  });

  it("rejects an account with no base_url when the provider sets none", () => {
    expect(() =>
      parseConfig({
        upstream: {},
        providers: { g1: { type: "ollama", accounts: [{ id: "a", api_key_env: "K" }] } },
        models: { m: { strategy: "single", provider: "g1", target: "x" } },
      }),
    ).toThrow(/no base_url/);
  });

  it("rejects a config with no provider source at all", () => {
    expect(() =>
      parseConfig({ upstream: { max_concurrency: 4 }, models: { m: { strategy: "single", target: "x" } } }),
    ).toThrow(/no providers configured/);
  });

  it("resolveProviders reads keys from env, derives host, and picks the client per group", () => {
    const cfg = parseConfig({
      upstream: {},
      providers: {
        "ollama-cloud": {
          type: "ollama",
          base_url: "https://ollama.com/",
          accounts: [{ id: "acc-1", api_key_env: "PRESENT" }],
        },
        openrouter: {
          type: "openai-compat",
          base_url: "https://openrouter.ai/api/v1",
          accounts: [{ id: "or-1", api_key_env: "MISSING", model_map: { x: "y" } }],
        },
      },
      models: { m: { strategy: "single", provider: "ollama-cloud", target: "x" } },
    });
    const groups = resolveProviders(cfg, { env: { PRESENT: "sk-1" }, fetchFn: stubFetch });
    expect(groups.map((g) => g.id)).toEqual(["ollama-cloud", "openrouter"]);
    const ollama = at(groups, 0).accounts[0];
    const or = at(groups, 1).accounts[0];
    expect(ollama?.cfg).toMatchObject({ id: "acc-1", group: "ollama-cloud", host: "https://ollama.com", hasKey: true });
    expect(ollama?.client.supportsNativeShow).toBe(true);
    expect(or?.cfg).toMatchObject({ id: "or-1", group: "openrouter", hasKey: false, modelMap: { x: "y" } });
    expect(or?.client.supportsNativeShow).toBe(false);
  });
});
