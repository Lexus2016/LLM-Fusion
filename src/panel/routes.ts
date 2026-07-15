import { Hono } from "hono";
import type { MiddlewareHandler } from "hono";
import type { Logger } from "pino";
import type { Config } from "../config";
import type { ProviderRouter } from "../connectors/provider_router";
import { createConfigEditorApp } from "./config_editor";
import { PANEL_HTML } from "./page";

/**
 * Local connector panel + admin API, mounted on the main app.
 *
 *  - `GET  /panel`                        → the dashboard HTML (no secrets, ungated
 *                                           so a browser can load it; the DATA below
 *                                           carries auth).
 *  - `GET  /admin/providers`              → grouped JSON snapshot (auth-gated when a
 *                                           token is configured).
 *  - `POST /admin/connectors/:id/:action` → disable | enable | reset | pin | unpin,
 *                                           by (globally-unique) account id.
 *
 * The mutating routes and the JSON snapshot go through the same auth middleware
 * as `/v1/*`; the HTML shell itself carries no connector data.
 */

export interface PanelDeps {
  router: ProviderRouter;
  auth: MiddlewareHandler;
  logger: Logger;
  /** Config accessors for the editor (`/admin/config*`). Optional so bare unit
   *  tests can build the monitor-only panel without a config file. */
  getConfig?: () => Config;
  configPath?: string;
  envHas?: (name: string) => boolean;
}

const ACTIONS = new Set(["disable", "enable", "reset", "pin", "unpin"]);

export function createPanelApp(deps: PanelDeps): Hono {
  const app = new Hono();

  app.get("/panel", (c) => c.html(PANEL_HTML));

  // No-YAML config editor (create/edit/delete providers + models), when wired.
  if (deps.getConfig && deps.configPath && deps.envHas) {
    app.route(
      "/",
      createConfigEditorApp({
        getConfig: deps.getConfig,
        configPath: deps.configPath,
        auth: deps.auth,
        logger: deps.logger,
        envHas: deps.envHas,
      }),
    );
  }

  app.get("/admin/providers", deps.auth, (c) =>
    c.json({ ...deps.router.snapshot(), now: Date.now() }),
  );

  app.post("/admin/connectors/:id/:action", deps.auth, (c) => {
    const id = c.req.param("id");
    const action = c.req.param("action");
    if (!ACTIONS.has(action)) {
      return c.json({ error: `unknown action '${action}'` }, 400);
    }
    const registry = deps.router.registryForAccount(id);
    if (!registry) return c.json({ error: `unknown account '${id}'` }, 404);
    let ok = true;
    if (action === "disable") ok = registry.disable(id);
    else if (action === "enable") ok = registry.enable(id);
    else if (action === "reset") ok = registry.reset(id);
    else if (action === "pin") ok = registry.pin(id);
    else if (action === "unpin") registry.unpin();
    if (!ok) return c.json({ error: `unknown account '${id}'` }, 404);
    deps.logger.info({ account: id, action }, "panel: account action");
    return c.json({ ...deps.router.snapshot(), now: Date.now() });
  });

  return app;
}
