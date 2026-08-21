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
  /**
   * In-flight discoveries keyed by model. The cache is only written AFTER
   * `/api/show` settles, so without this map every concurrent caller for a cold
   * model issues its own request — `GET /v1/models` fans out over all virtual
   * models in parallel and a physical model shared by two presets is discovered
   * once per referencing preset. Mirrors `routerPending` in strategies/smart.ts.
   */
  private readonly pending = new Map<string, Promise<DiscoveryResult>>();
  /**
   * Bumped by `clear()`. A discovery that started before a config hot-reload may
   * not write the post-reload cache — see `clear()`.
   */
  private epoch = 0;

  constructor(private readonly deps: CapabilityServiceDeps) {}

  clear(): void {
    this.cache.clear();
    // Dropping `pending` does NOT stop an already-running discovery from
    // reaching its `cache.set`, and that result was computed against the
    // pre-reload overrides/upstream — so the epoch bump makes the late write a
    // no-op. Callers already awaiting an in-flight promise still get a result
    // (discovery never throws); only the cache write is suppressed, so the next
    // caller re-discovers under the new config.
    this.pending.clear();
    this.epoch += 1;
  }

  async discover(model: string): Promise<DiscoveryResult> {
    const cached = this.cache.get(model);
    if (cached) return cached;

    const inFlight = this.pending.get(model);
    if (inFlight) return inFlight;

    const promise = this.discoverUncached(model, this.epoch).finally(() => {
      // Delete only OUR entry: after a `clear()` the map may already hold a
      // newer discovery for the same model.
      if (this.pending.get(model) === promise) this.pending.delete(model);
    });
    this.pending.set(model, promise);
    return promise;
  }

  /** The actual `/api/show` round-trip. Isolated so `discover` can coalesce. */
  private async discoverUncached(model: string, epoch: number): Promise<DiscoveryResult> {
    let result: DiscoveryResult;
    try {
      const raw = await this.deps.client.show(model);
      result = { capability: parseShow(raw), source: "discovered" };
    } catch (err) {
      result = this.degrade(model, err);
    }
    // Only cache AUTHORITATIVE results. A `default` result is a guess produced by
    // a (possibly transient) discovery failure with no override; caching it would
    // permanently misreport capabilities until config reload, so we leave it
    // uncached and let the next call retry discovery — self-healing once the
    // upstream recovers. `discovered` and operator `override` results are stable.
    // The epoch check drops results from a discovery that a `clear()` outran.
    if (result.source !== "default" && epoch === this.epoch) {
      this.cache.set(model, result);
    }
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
