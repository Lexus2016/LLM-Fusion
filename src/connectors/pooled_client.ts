import type { Logger } from "pino";
import type { ChatCompletionResult, UpstreamClient } from "../types";
import { NoConnectorAvailableError, NotImplementedError } from "../errors";
import { ConnectorRegistry, type ConnectorClient } from "./registry";
import { bodyText, classifyStatus, classifyThrown, type ClassifyOptions } from "./health";

/**
 * `PooledUpstreamClient` — an `UpstreamClient` that fronts an ordered pool of
 * connectors (accounts/providers) and fails over between them transparently.
 * Injected as `client` in place of the single OllamaClient, so every strategy
 * (single/failover/fusion/smart) keeps calling `client.chatCompletions()`
 * unchanged; connector-level failover happens here.
 *
 * Per call: walk connectors in selection order (pinned-first, then config order);
 * for each usable one (up, or a single probe of an elapsed cooling/down), map the
 * model via the connector's `model_map` and try it. Classify the outcome and
 * either return, advance, or surface. When every connector fails, surface the
 * MOST-RECOVERABLE failure by severity (so a primary's transient 429 beats a dead
 * backup's 401), and throw `NoConnectorAvailableError` only when nothing was even
 * attempted.
 */

type Best =
  | { severity: number; result: ChatCompletionResult }
  | { severity: number; thrown: unknown };

export interface PooledClientOptions {
  logger?: Logger;
  now?: () => number;
}

export class PooledUpstreamClient implements UpstreamClient {
  private readonly registry: ConnectorRegistry;
  private readonly logger: Logger | undefined;
  private readonly now: () => number;

  constructor(registry: ConnectorRegistry, opts: PooledClientOptions = {}) {
    this.registry = registry;
    this.logger = opts.logger;
    this.now = opts.now ?? Date.now;
  }

  async chatCompletions(
    body: Record<string, unknown>,
    opts: { stream: boolean; signal?: AbortSignal },
  ): Promise<ChatCompletionResult> {
    return this.runFailover(
      this.registry.order(),
      (client, b, o) => client.chatCompletions(b, o),
      body,
      opts,
    );
  }

  async chatNative(
    body: Record<string, unknown>,
    opts: { stream: boolean; signal?: AbortSignal },
  ): Promise<ChatCompletionResult> {
    // Native /api/chat (vision) is Ollama-only — restrict the pool to
    // native-capable connectors.
    const order = this.registry.nativeShowOrder();
    if (order.length === 0) {
      throw new NotImplementedError(
        "no native (ollama) connector available for /api/chat (native vision)",
      );
    }
    return this.runFailover(order, (client, b, o) => client.chatNative(b, o), body, opts);
  }

  async show(model: string, opts: { signal?: AbortSignal } = {}): Promise<unknown> {
    // Capability discovery is best-effort and health-neutral: try each
    // native-capable connector until one answers; never flip connector health on
    // a discovery miss (the caller degrades to overrides/defaults).
    let lastErr: unknown;
    for (const id of this.registry.nativeShowOrder()) {
      const client = this.registry.clientFor(id);
      if (!client) continue;
      try {
        return await client.show(model, opts);
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr ?? new NotImplementedError("no native (ollama) connector available for /api/show");
  }

  /** The shared connector-failover loop used by both chat paths. */
  private async runFailover(
    order: string[],
    call: (
      client: ConnectorClient,
      body: Record<string, unknown>,
      opts: { stream: boolean; signal?: AbortSignal },
    ) => Promise<ChatCompletionResult>,
    body: Record<string, unknown>,
    opts: { stream: boolean; signal?: AbortSignal },
  ): Promise<ChatCompletionResult> {
    let best: Best | undefined;
    let notFound: ChatCompletionResult | undefined;

    for (const id of order) {
      const acq = this.registry.acquire(id);
      if (!acq.ok) continue;

      const cfg = this.registry.cfgFor(id);
      const client = this.registry.clientFor(id);
      if (!cfg || !client) continue;

      const mapped = mapModel(body, cfg.modelMap);
      const startedAt = this.now();
      let result: ChatCompletionResult;
      try {
        result = await call(client, mapped, opts);
      } catch (err) {
        // Client disconnect: not a connector health failure — release the probe
        // and rethrow (the request is gone).
        if (opts.signal?.aborted) {
          this.registry.recordAbandoned(id, acq.epoch);
          throw err;
        }
        const cls = classifyThrown(err);
        if (cls.kind === "surface") {
          // Capability / not-implemented / bad-request: not a health failure and
          // deterministic across connectors — release the probe and surface it.
          this.registry.recordAbandoned(id, acq.epoch);
          throw err;
        }
        if (cls.kind === "failure") {
          this.registry.recordFailure(id, acq.epoch, cls.reason, { error: errMessage(err) });
          this.logger?.warn(
            { connector: id, reason: cls.reason, err: errMessage(err) },
            "connector call threw; advancing",
          );
          best = rank(best, { severity: cls.severity, thrown: err });
        }
        continue;
      }

      if (result.status < 400) {
        this.registry.recordSuccess(id, acq.epoch, this.now() - startedAt);
        return result;
      }

      const data = result.kind === "json" ? result.data : null;
      const cls = classifyStatus(result.status, bodyText(data), classifyOptions(cfg));

      if (cls.kind === "success" || cls.kind === "request_error") {
        // The connector answered (a bad request or a passthrough 4xx): it is
        // healthy, and retrying elsewhere won't help — surface immediately.
        this.registry.recordSuccess(id, acq.epoch, this.now() - startedAt);
        return result;
      }

      if (cls.kind === "not_found") {
        // 404 / model-not-found: a model_map/routing issue on THIS connector.
        // Advance without touching health; keep it as a last-resort fallback.
        this.registry.recordAbandoned(id, acq.epoch);
        notFound = result;
        continue;
      }

      // cls.kind === "failure": mark the connector and advance. (classifyStatus
      // never yields "surface" — that comes only from a thrown error above.)
      if (cls.kind === "failure") {
        const cooldownMs = result.kind === "json" ? result.retryAfterMs : undefined;
        this.registry.recordFailure(id, acq.epoch, cls.reason, {
          error: `HTTP ${result.status}`,
          cooldownMs,
        });
        this.logger?.warn(
          { connector: id, reason: cls.reason, status: result.status },
          "connector failed; advancing",
        );
        best = rank(best, { severity: cls.severity, result });
      }
    }

    if (best) {
      if ("result" in best) return best.result;
      throw best.thrown;
    }
    if (notFound) return notFound;
    throw new NoConnectorAvailableError(
      "no upstream connector is currently usable (all disabled, cooling, or down)",
    );
  }
}

/** Translate the logical model id to the connector's upstream id when mapped. */
function mapModel(
  body: Record<string, unknown>,
  modelMap: Record<string, string>,
): Record<string, unknown> {
  const model = body.model;
  if (typeof model === "string" && Object.prototype.hasOwnProperty.call(modelMap, model)) {
    return { ...body, model: modelMap[model] };
  }
  return body;
}

function classifyOptions(cfg: { treat403As: "passthrough" | "down"; quotaMarkers: string[] }): ClassifyOptions {
  return { treat403As: cfg.treat403As, quotaMarkers: cfg.quotaMarkers };
}

/** Keep the higher-severity failure; ties keep the earlier (higher-priority) one. */
function rank(best: Best | undefined, cand: Best): Best {
  return !best || cand.severity > best.severity ? cand : best;
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
