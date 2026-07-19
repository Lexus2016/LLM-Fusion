import { describe, it, expect } from "vitest";
import { pino } from "pino";
import { CapabilityService } from "../src/capabilities";
import { ProviderRouter } from "../src/connectors/provider_router";
import type { ResolvedGroup } from "../src/connectors/resolve";
import type { ConnectorClient, ResolvedConnector } from "../src/connectors/registry";
import type { ChatCompletionResult } from "../src/types";

const logger = pino({ level: "silent" });

/** Minimal client that just records which pool answered a /api/show call. */
class SpyShowClient implements ConnectorClient {
  readonly supportsNativeShow = true;
  readonly calls: string[] = [];
  readonly tag: string;

  constructor(tag: string) {
    this.tag = tag;
  }

  async chatCompletions(): Promise<ChatCompletionResult> {
    return { kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
  }

  async chatNative(): Promise<ChatCompletionResult> {
    return { kind: "json", status: 200, data: {}, usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
  }

  async show(model: string): Promise<unknown> {
    this.calls.push(model);
    return { capabilities: ["completion"], model_info: {} };
  }
}

function resolved(id: string, tag: string): ResolvedConnector {
  return {
    id,
    group: "g1",
    provider: "ollama",
    baseUrl: `https://${id}.test`,
    host: `https://${id}.test`,
    hasKey: true,
    treat403As: "passthrough",
    quotaMarkers: [],
    modelMap: {},
  };
}

function groupFor(client: SpyShowClient): ResolvedGroup {
  return {
    id: "g1",
    type: "ollama",
    accounts: [{ cfg: resolved(client.tag + "-account", client.tag), client }],
  };
}

describe("provider reload wiring", () => {
  it("capability discovery follows a live providers rebuild (no stale pool)", async () => {
    const first = new SpyShowClient("first");
    const router = new ProviderRouter([groupFor(first)]);

    // Same pattern src/index.ts uses: a delegating client that resolves the
    // CURRENT default pool on every call, never a captured instance.
    const liveClient = {
      show: (model: string, opts?: { signal?: AbortSignal }) => router.defaultPool.show(model, opts),
    };
    const caps = new CapabilityService({ client: liveClient, getOverrides: () => ({}), logger });

    await caps.discover("m");
    expect(first.calls).toEqual(["m"]);

    const second = new SpyShowClient("second");
    router.reload([groupFor(second)]);
    caps.clear(); // index.ts clears the cache on every reload
    await caps.discover("m");

    expect(second.calls).toEqual(["m"]);
    expect(first.calls).toEqual(["m"]); // old pool was not queried again
  });
});
