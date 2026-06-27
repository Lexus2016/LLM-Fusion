import type { ModelConfig } from "./config";
import type { RequestContext, StrategyContext } from "./types";
import { FusionError, NotFoundError } from "./errors";
import { singleStrategy } from "./strategies/single";
import { failoverStrategy } from "./strategies/failover";
import { fusionStrategy } from "./strategies/fusion";
import { smartStrategy } from "./strategies/smart";
import { assertSingleVisionCapable, requestHasImages } from "./vision";

// Re-export so existing importers keep resolving `requestHasImages` from here;
// the canonical implementation now lives in `vision.ts` (shared with fusion,
// avoiding a router <-> fusion import cycle).
export { requestHasImages };

/**
 * Resolve the incoming virtual `model` name to a config entry and dispatch to
 * its strategy (`single` | `failover` | `fusion` | `smart`). A minimal
 * capability gate blocks image input to a known-non-vision single target.
 */
export async function dispatch(ctx: RequestContext): Promise<Response> {
  const name = ctx.request.model;
  const entry = ctx.config.models[name];
  if (!entry) {
    throw new NotFoundError(`unknown virtual model '${name}'`);
  }

  const strategyCtx: StrategyContext = {
    request: ctx.request,
    config: ctx.config,
    client: ctx.client,
    capabilities: ctx.capabilities,
    logger: ctx.logger,
    resilience: ctx.resilience,
    modelConfig: entry,
  };

  switch (entry.strategy) {
    case "single":
      await enforceCapabilityGate(strategyCtx, entry.target);
      return singleStrategy.execute(strategyCtx);
    case "failover":
      return failoverStrategy.execute(strategyCtx);
    case "fusion":
      // Vision and tool gates are applied inside the fusion strategy (the panel
      // member set depends on per-member vision discovery).
      return fusionStrategy.execute(strategyCtx);
    case "smart":
      // The smart router classifies the request, then delegates to the existing
      // single/fusion executors on the resolved sub-config (best-effort router;
      // failures degrade to the configured `default` route).
      return smartStrategy.execute(strategyCtx);
    default: {
      const exhaustive: never = entry;
      throw new FusionError(`unhandled strategy: ${String(exhaustive)}`, 500, "internal_error");
    }
  }
}

/**
 * Minimal capability gate (Phase 1): if the request carries image blocks and
 * the resolved single target is known-non-vision, reject with 400. Discovery is
 * only triggered when images are actually present, so the common text path adds
 * no upstream call. Delegates to the shared `assertSingleVisionCapable` helper
 * (also used by the `smart` -> `simple` sub-route).
 */
async function enforceCapabilityGate(ctx: StrategyContext, target: string): Promise<void> {
  await assertSingleVisionCapable(ctx.capabilities, ctx.request, target, ctx.request.model);
}

/** A representative real upstream model for a virtual model (for /ready, /v1/models). */
export function representativeMember(entry: ModelConfig): string | undefined {
  switch (entry.strategy) {
    case "single":
      return entry.target;
    case "failover":
      return entry.chain[0];
    case "fusion":
      return entry.judge;
    case "smart":
      return entry.router;
    default: {
      const exhaustive: never = entry;
      return exhaustive;
    }
  }
}
