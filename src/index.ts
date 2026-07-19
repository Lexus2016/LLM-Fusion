import { serve } from "@hono/node-server";
import { createConfigManager, findPanelContentionOverlaps } from "./config";
import type { Config } from "./config";
import { resolveProviders } from "./connectors/resolve";
import { ProviderRouter } from "./connectors/provider_router";
import { CapabilityService } from "./capabilities";
import { createApp } from "./server";
import { createLogger } from "./logging";
import { resolveAuthToken } from "./auth";
import type { UpstreamClient } from "./types";

/**
 * Entrypoint: load config (FUSION_CONFIG env or ./fusion.yaml), build the
 * Ollama client, wire capability discovery + hot-reload, and start the server.
 */
async function main(): Promise<void> {
  // Optionally load a local .env (Node 24 native; no dotenv dependency). An
  // absent file is fine — the proxy also reads plain process env / inline vars.
  // Must run before any env var is read (logger level, FUSION_CONFIG, the key).
  try {
    process.loadEnvFile();
  } catch (err) {
    // A missing .env is expected and silently ignored. Surface anything else
    // (e.g. a permission error or unreadable file) so it is not lost — the pino
    // logger does not exist yet (env must load first to pick up LOG_LEVEL /
    // LOG_PRETTY), so emit a Node process warning rather than a structured log.
    const code = err instanceof Error && "code" in err ? err.code : undefined;
    if (code !== "ENOENT") {
      process.emitWarning(`could not load .env file: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  const logger = createLogger();
  const configPath = process.env.FUSION_CONFIG ?? "./fusion.yaml";
  const manager = await createConfigManager(configPath, logger);
  const cfg = manager.config;

  // Build the provider-group router: one registry + pool per provider group
  // (Ollama Cloud, OpenRouter, …), each with its ordered accounts. Failover stays
  // within a group; a virtual model is served by its group's pool. Legacy single
  // `upstream.base_url`+`api_key_env` synthesises one `default` group. The router
  // is process-lifetime (like the resilience limiter): models/routing hot-reload,
  // and `providers:` edits rebuild the pools in place (see onReload below).
  const groups = resolveProviders(cfg, { env: process.env, logger });
  const router = new ProviderRouter(groups, {
    cooldownMs: cfg.upstream.connector_cooldown_s * 1000,
    downRecheckMs: cfg.upstream.connector_down_recheck_s * 1000,
    logger,
  });
  // NEVER capture `router.defaultPool` in a const: `ProviderRouter.reload()`
  // replaces every pool, and a captured instance would keep capability
  // discovery (and createApp's no-router fallbacks) talking to the old,
  // decommissioned accounts with the old keys. This wrapper resolves the
  // CURRENT pool on every call. (Chat traffic is unaffected either way:
  // `dispatch` resolves `router.poolFor(...)` per request.)
  const liveClient: UpstreamClient = {
    chatCompletions: (body, opts) => router.defaultPool.chatCompletions(body, opts),
    show: (model, opts) => router.defaultPool.show(model, opts),
    chatNative: (body, opts) => router.defaultPool.chatNative(body, opts),
  };

  const capabilities = new CapabilityService({
    client: liveClient,
    getOverrides: () => manager.config.overrides,
    logger,
  });
  // Signature of the provider-layer config: a change here rebuilds the router
  // live, a models-only edit does not. Serialized from the raw config (not the
  // resolved groups, which deliberately drop secret-adjacent fields) so EVERY
  // account field counts — api_key_env, extra_headers, request_timeout_s
  // included. `upstream.request_timeout_s` is the per-account fallback
  // (resolve.ts), so it must count even when `providers:` is set; the other
  // `upstream:` knobs (concurrency, cooldowns) are process-lifetime and
  // deliberately excluded. Secret values only ever sit in this in-memory
  // string, never logged.
  const providersSignature = (c: Config): string =>
    JSON.stringify(
      // Same condition as resolveProviders: an EMPTY providers map falls back
      // to the legacy upstream group, so its signature must too.
      c.providers && Object.keys(c.providers).length > 0
        ? { providers: c.providers, fallback_request_timeout_s: c.upstream.request_timeout_s }
        : {
            base_url: c.upstream.base_url ?? null,
            api_key_env: c.upstream.api_key_env ?? null,
            request_timeout_s: c.upstream.request_timeout_s,
          },
    );
  let prevProviders = providersSignature(cfg);

  manager.onReload(() => {
    // Models / routing hot-reload live. When the config editor changes the
    // `providers:` layer, rebuild the router in place so it applies without a
    // restart; a models-only edit leaves connector health untouched.
    capabilities.clear();
    try {
      const sig = providersSignature(manager.config);
      if (sig !== prevProviders) {
        router.reload(resolveProviders(manager.config, { env: process.env, logger }));
        prevProviders = sig;
        logger.info("configuration reloaded (providers rebuilt live)");
      } else {
        logger.info("configuration reloaded (models/routing)");
      }
    } catch (err) {
      logger.error(
        { err: err instanceof Error ? err.message : String(err) },
        "provider reload failed; keeping the previous router",
      );
    }
  });

  // FUSION_ALLOW_OPEN=1 is the explicit opt-out from the non-loopback
  // fail-fast below: the operator takes responsibility for access control. It
  // un-bricks a configured-but-UNSET token var (which would otherwise 500 every
  // request) — but it never disables a token that actually resolves, so an
  // operator who sets the hatch alongside a valid token keeps their auth.
  const allowOpen = Boolean(process.env.FUSION_ALLOW_OPEN);
  const getAuthToken = (): string | undefined => {
    const token = resolveAuthToken(manager.config.server.auth_token_env, process.env);
    return allowOpen && token === "" ? undefined : token;
  };
  const authOn = Boolean(getAuthToken());
  if (allowOpen && !authOn) {
    logger.warn(
      "FUSION_ALLOW_OPEN is set: client auth is DISABLED — front this proxy with your own access control",
    );
  } else if (manager.config.server.auth_token_env && !authOn) {
    // Configured-but-UNSET (a misnamed/typo'd env var) must be LOUD: auth fails
    // closed — every request gets the middleware's 500 "configured but empty" —
    // rather than the proxy silently running open while the operator believes
    // it is authenticated.
    logger.error(
      { env: manager.config.server.auth_token_env },
      "server.auth_token_env names an env var that is UNSET (misnamed?); auth fails closed — all requests get 500 until it is set",
    );
  } else if (!authOn) {
    logger.warn(
      "no client auth token configured (server.auth_token_env unset/empty); proxy is UNAUTHENTICATED — localhost single-user only",
    );
  }

  // Admin surface (/admin/* + panel) token, SEPARATE from the client token when
  // `server.admin_token_env` is set — so the widely-copied client API token does
  // not also grant config edits + restart. Falls back to the client token when no
  // dedicated admin token is configured (backward compatible); when neither
  // resolves the admin API stays loopback-only (its Host-pinning guard).
  const getAdminToken = (): string | undefined => {
    if (manager.config.server.admin_token_env) {
      return resolveAuthToken(manager.config.server.admin_token_env, process.env);
    }
    return getAuthToken();
  };
  if (manager.config.server.admin_token_env) {
    logger.info(
      { env: manager.config.server.admin_token_env, admin_auth: getAdminToken() ? "on" : "fails-closed (env unset)" },
      "admin surface uses a dedicated token (separate from the client API token)",
    );
  }

  // Rate-limit contention check (non-fatal): a `single`/`failover` model whose
  // upstream target is ALSO a live `fusion` panel member in the same provider
  // group shares one upstream rate-limit bucket AND per_model_concurrency gate
  // with that panel. Claude Code drives ANTHROPIC_SMALL_FAST_MODEL with 80-130
  // background calls/min, so pointing it at such a model can 429-starve the
  // panel mid-request. Warn only — never mutate config — so the operator can
  // retarget the small-fast model or split the provider group. See the
  // rate-limit note in bin/fusion-claude.
  for (const overlap of findPanelContentionOverlaps(manager.config)) {
    logger.warn(
      { fastModel: overlap.fastModel, target: overlap.target, fusionModel: overlap.fusionModel },
      `model '${overlap.fastModel}' (single/failover target '${overlap.target}') overlaps fusion panel member of '${overlap.fusionModel}'; ` +
        "small-fast burst traffic can 429-starve the panel (see fusion-claude rate-limit note)",
    );
  }

  // Restart handler for the panel's Restart button: boot-only settings (bind/port,
  // upstream concurrency/timeouts) need a fresh process. We respond to the HTTP
  // request first (in config_editor), then here: close the listener and exit
  // NON-ZERO so the supervisor relaunches us — launchd's KeepAlive is configured
  // `SuccessfulExit:false` (restart only on a non-zero exit), and systemd
  // `on-failure` / docker `on-failure` behave the same. A plain exit(0) would NOT
  // be relaunched by any of them. The short delay lets the 200 flush to the panel.
  let server: ReturnType<typeof serve> | undefined;
  const requestRestart = (): void => {
    logger.warn("llm-fusion: restart requested — exiting for supervisor relaunch");
    setTimeout(() => {
      const bail = () => process.exit(1);
      // Prefer a graceful listener close, but never hang on it.
      try {
        server?.close?.(bail);
      } catch {
        bail();
      }
      setTimeout(bail, 2000).unref();
    }, 150);
  };

  const app = createApp({
    getConfig: () => manager.config,
    client: liveClient,
    capabilities,
    getAuthToken,
    getAdminToken,
    logger,
    router,
    configPath,
    envHas: (name: string) => Boolean(process.env[name]),
    requestRestart,
  });

  // `server.bind` is the default; FUSION_BIND overrides it without editing the
  // mounted config (handy in Docker, where the in-image config binds 127.0.0.1
  // but the container must listen on 0.0.0.0 to be reachable from the host). An
  // empty FUSION_BIND ("") must NOT override to "" (which some servers treat as
  // "all interfaces") — `||` falls through to the configured bind, unlike `??`.
  const bind = process.env.FUSION_BIND || manager.config.server.bind;
  const { port } = manager.config.server;

  // Fail fast rather than publish an unauthenticated proxy (billed to the
  // operator's key, plus the admin API) on a routable interface — the Docker
  // image sets FUSION_BIND=0.0.0.0, so this bites exactly there. Bind loopback
  // or configure a token; FUSION_ALLOW_OPEN=1 is the explicit escape hatch for
  // deployments that front the proxy with their own auth. The whole 127.0.0.0/8
  // block is loopback (incl. IPv4-mapped ::ffff:127.*), as are the ::1 forms.
  const isLoopbackBind =
    bind === "localhost" ||
    bind === "::1" ||
    bind === "0:0:0:0:0:0:0:1" ||
    bind.startsWith("127.") ||
    bind.startsWith("::ffff:127.");
  if (!isLoopbackBind && !authOn && !allowOpen) {
    throw new Error(
      `refusing to start: bind '${bind}' is not loopback and no client auth token resolves ` +
        "(server.auth_token_env). Set the token env var, bind to 127.0.0.1, or set " +
        "FUSION_ALLOW_OPEN=1 to run an open proxy on a non-loopback interface.",
    );
  }

  // Startup banner: what is listening, which virtual models are loaded and with
  // which strategy, whether client auth is enforced, and the connector pool
  // (id, provider, and whether each connector's key resolved). No secrets logged.
  const models = Object.entries(manager.config.models).map(
    ([name, entry]) => `${name} (${entry.strategy})`,
  );
  const providers = groups.map((g) => {
    const missing = g.accounts.filter((a) => !a.cfg.hasKey).length;
    return `${g.id} (${g.type}, ${g.accounts.length} account${g.accounts.length === 1 ? "" : "s"}${missing ? `, ${missing} NO KEY` : ""})`;
  });
  logger.info(
    {
      bind,
      port,
      url: `http://${bind}:${port}`,
      panel: `http://${bind}:${port}/panel`,
      models,
      auth: authOn ? "on" : "off",
      providers,
      config: configPath,
    },
    "llm-fusion starting",
  );

  server = serve({ fetch: app.fetch, hostname: bind, port }, (info) => {
    logger.info(`llm-fusion listening on http://${bind}:${info.port}`);
  });
}

main().catch((err) => {
  console.error("fatal:", err instanceof Error ? err.message : err);
  process.exit(1);
});
