import { serve } from "@hono/node-server";
import { createConfigManager } from "./config";
import { resolveProviders } from "./connectors/resolve";
import { ProviderRouter } from "./connectors/provider_router";
import { CapabilityService } from "./capabilities";
import { createApp } from "./server";
import { createLogger } from "./logging";

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
  // but provider/account changes need a restart.
  const groups = resolveProviders(cfg, { env: process.env, logger });
  const router = new ProviderRouter(groups, {
    cooldownMs: cfg.upstream.connector_cooldown_s * 1000,
    downRecheckMs: cfg.upstream.connector_down_recheck_s * 1000,
    logger,
  });
  // Capability discovery + fallback use the first group's pool (routes /api/show
  // to an ollama account, else degrades to defaults).
  const client = router.defaultPool;

  const capabilities = new CapabilityService({
    client,
    getOverrides: () => manager.config.overrides,
    logger,
  });
  const providersSignature = (gs: typeof groups): string =>
    JSON.stringify(
      gs.map((g) => ({
        id: g.id,
        type: g.type,
        accounts: g.accounts.map((a) => ({
          id: a.cfg.id,
          baseUrl: a.cfg.baseUrl,
          treat403As: a.cfg.treat403As,
          quotaMarkers: a.cfg.quotaMarkers,
          modelMap: a.cfg.modelMap,
        })),
      })),
    );
  let prevProviders = providersSignature(groups);

  manager.onReload(() => {
    // Models / routing hot-reload live. When the config editor changes the
    // `providers:` layer, rebuild the router in place so it applies without a
    // restart; a models-only edit leaves connector health untouched.
    capabilities.clear();
    try {
      const newGroups = resolveProviders(manager.config, { env: process.env, logger });
      const sig = providersSignature(newGroups);
      if (sig !== prevProviders) {
        router.reload(newGroups);
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

  const getAuthToken = (): string | undefined => {
    const envName = manager.config.server.auth_token_env;
    return envName ? process.env[envName] : undefined;
  };
  const authOn = Boolean(getAuthToken());
  if (!authOn) {
    logger.warn(
      "no client auth token configured (server.auth_token_env unset/empty); proxy is UNAUTHENTICATED — localhost single-user only",
    );
  }

  const app = createApp({
    getConfig: () => manager.config,
    client,
    capabilities,
    getAuthToken,
    logger,
    router,
    configPath,
    envHas: (name: string) => Boolean(process.env[name]),
  });

  // `server.bind` is the default; FUSION_BIND overrides it without editing the
  // mounted config (handy in Docker, where the in-image config binds 127.0.0.1
  // but the container must listen on 0.0.0.0 to be reachable from the host). An
  // empty FUSION_BIND ("") must NOT override to "" (which some servers treat as
  // "all interfaces") — `||` falls through to the configured bind, unlike `??`.
  const bind = process.env.FUSION_BIND || manager.config.server.bind;
  const { port } = manager.config.server;

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

  serve({ fetch: app.fetch, hostname: bind, port }, (info) => {
    logger.info(`llm-fusion listening on http://${bind}:${info.port}`);
  });
}

main().catch((err) => {
  console.error("fatal:", err instanceof Error ? err.message : err);
  process.exit(1);
});
