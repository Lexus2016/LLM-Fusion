import { serve } from "@hono/node-server";
import { createConfigManager } from "./config";
import { OllamaClient } from "./upstream/ollama";
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
  } catch {
    // No .env present; rely on the ambient environment.
  }

  const logger = createLogger();
  const configPath = process.env.FUSION_CONFIG ?? "./fusion.yaml";
  const manager = await createConfigManager(configPath, logger);
  const cfg = manager.config;

  const apiKey = process.env[cfg.upstream.api_key_env];
  if (!apiKey) {
    logger.warn(
      { env: cfg.upstream.api_key_env },
      "upstream API key env var is unset; upstream calls will be unauthenticated and will likely fail",
    );
  }

  // The upstream block (base_url / key / timeout) is read once at boot; routing
  // and model changes hot-reload live, but upstream changes need a restart.
  const client = new OllamaClient({
    baseUrl: cfg.upstream.base_url,
    apiKey,
    timeoutMs: cfg.upstream.request_timeout_s * 1000,
  });

  const capabilities = new CapabilityService({
    client,
    getOverrides: () => manager.config.overrides,
    logger,
  });
  manager.onReload(() => capabilities.clear());

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
  });

  // `server.bind` is the default; FUSION_BIND overrides it without editing the
  // mounted config (handy in Docker, where the in-image config binds 127.0.0.1
  // but the container must listen on 0.0.0.0 to be reachable from the host). An
  // empty FUSION_BIND ("") must NOT override to "" (which some servers treat as
  // "all interfaces") — `||` falls through to the configured bind, unlike `??`.
  const bind = process.env.FUSION_BIND || manager.config.server.bind;
  const { port } = manager.config.server;

  // Startup banner: what is listening, which virtual models are loaded and with
  // which strategy, whether client auth is enforced, and whether the upstream
  // key is present. No secret values are logged.
  const models = Object.entries(manager.config.models).map(
    ([name, entry]) => `${name} (${entry.strategy})`,
  );
  logger.info(
    {
      bind,
      port,
      url: `http://${bind}:${port}`,
      models,
      auth: authOn ? "on" : "off",
      upstream_key: apiKey ? "present" : "MISSING",
      upstream: cfg.upstream.base_url,
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
