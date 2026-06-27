import { z } from "zod";
import type { Logger } from "pino";
import type { Capability, CapabilityProvider, DiscoveryResult } from "./types";
import type { Config } from "./config";

/**
 * Capability discovery via `POST /api/show`. Results are cached by model name
 * and refreshed on config hot-reload (`clear()`).
 *
 * Graceful degrade: if discovery fails for a model, fall back to an `overrides`
 * entry if present, else to conservative defaults `{vision:false, tools:true,
 * context:null}`. Discovery never throws out of `discover`.
 */

type OverridesMap = Config["overrides"];

const ShowSchema = z
  .object({
    capabilities: z.array(z.string()).optional(),
    model_info: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough();

export interface CapabilityServiceDeps {
  client: { show(model: string): Promise<unknown> };
  getOverrides: () => OverridesMap;
  logger: Logger;
}

export class CapabilityService implements CapabilityProvider {
  private readonly cache = new Map<string, DiscoveryResult>();

  constructor(private readonly deps: CapabilityServiceDeps) {}

  clear(): void {
    this.cache.clear();
  }

  async discover(model: string): Promise<DiscoveryResult> {
    const cached = this.cache.get(model);
    if (cached) return cached;

    let result: DiscoveryResult;
    try {
      const raw = await this.deps.client.show(model);
      result = { capability: parseShow(raw), source: "discovered" };
    } catch (err) {
      result = this.degrade(model, err);
    }
    this.cache.set(model, result);
    return result;
  }

  private degrade(model: string, err: unknown): DiscoveryResult {
    const override = this.deps.getOverrides()[model];
    if (override) {
      this.deps.logger.warn({ model }, "capability discovery missed; using configured override");
      return {
        capability: {
          vision: override.vision ?? false,
          tools: override.tools ?? true,
          context: override.context ?? null,
        },
        source: "override",
      };
    }
    this.deps.logger.warn(
      { model, reason: err instanceof Error ? err.message : String(err) },
      "capability discovery missed; using conservative defaults",
    );
    return { capability: { vision: false, tools: true, context: null }, source: "default" };
  }
}

/**
 * Parse an `/api/show` payload into a Capability.
 *
 * Context length is matched by a GENERIC `*.context_length` key (architecture
 * prefix varies: `qwen3.context_length`, `glm.context_length`, ...) — never a
 * hardcoded architecture name.
 */
export function parseShow(raw: unknown): Capability {
  const parsed = ShowSchema.parse(raw);
  const caps = parsed.capabilities ?? [];
  const vision = caps.includes("vision");
  const tools = caps.includes("tools");

  let context: number | null = null;
  if (parsed.model_info) {
    for (const [key, value] of Object.entries(parsed.model_info)) {
      if (key.endsWith(".context_length") && typeof value === "number") {
        context = value;
        break;
      }
    }
  }

  return { vision, tools, context };
}
