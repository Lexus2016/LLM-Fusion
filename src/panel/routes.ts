import { Hono } from "hono";
import type { MiddlewareHandler } from "hono";
import type { Logger } from "pino";
import type { ConnectorRegistry } from "../connectors/registry";
import { PANEL_HTML } from "./page";

/**
 * Local connector panel + admin API, mounted on the main app.
 *
 *  - `GET  /panel`                       → the dashboard HTML (no secrets, ungated
 *                                          so a browser can load it; the DATA below
 *                                          is what carries auth).
 *  - `GET  /admin/connectors`            → JSON snapshot (auth-gated when a token is
 *                                          configured).
 *  - `POST /admin/connectors/:id/:action`→ disable | enable | reset | pin.
 *  - `POST /admin/unpin`                 → clear the active-connector pin.
 *
 * All mutating routes and the JSON snapshot go through the same auth middleware
 * as `/v1/*`; the HTML shell itself carries no connector data.
 */

export interface PanelDeps {
  registry: ConnectorRegistry;
  auth: MiddlewareHandler;
  logger: Logger;
}

const ACTIONS = new Set(["disable", "enable", "reset", "pin"]);

export function createPanelApp(deps: PanelDeps): Hono {
  const app = new Hono();

  app.get("/panel", (c) => c.html(PANEL_HTML));

  app.get("/admin/connectors", deps.auth, (c) =>
    c.json({
      connectors: deps.registry.snapshot(),
      activeId: deps.registry.activeId() ?? null,
      now: Date.now(),
    }),
  );

  app.post("/admin/connectors/:id/:action", deps.auth, (c) => {
    const id = c.req.param("id");
    const action = c.req.param("action");
    if (!ACTIONS.has(action)) {
      return c.json({ error: `unknown action '${action}'` }, 400);
    }
    let ok = false;
    if (action === "disable") ok = deps.registry.disable(id);
    else if (action === "enable") ok = deps.registry.enable(id);
    else if (action === "reset") ok = deps.registry.reset(id);
    else if (action === "pin") ok = deps.registry.pin(id);
    if (!ok) return c.json({ error: `unknown connector '${id}'` }, 404);
    deps.logger.info({ connector: id, action }, "panel: connector action");
    return c.json({
      ok: true,
      connectors: deps.registry.snapshot(),
      activeId: deps.registry.activeId() ?? null,
      now: Date.now(),
    });
  });

  app.post("/admin/unpin", deps.auth, (c) => {
    deps.registry.unpin();
    deps.logger.info("panel: cleared active-connector pin");
    return c.json({
      ok: true,
      connectors: deps.registry.snapshot(),
      activeId: deps.registry.activeId() ?? null,
      now: Date.now(),
    });
  });

  return app;
}
