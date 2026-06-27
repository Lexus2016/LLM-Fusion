import { readFile } from "node:fs/promises";
import { watch } from "node:fs";
import type { FSWatcher } from "node:fs";
import { dirname, basename } from "node:path";
import { parse as parseYaml } from "yaml";
import type { Logger } from "pino";
import { z } from "zod";

/**
 * Full configuration schema for the Fusion Proxy.
 *
 * The schema is intentionally minimal-by-default with strong defaults: a working
 * config fits in ~10 lines (see `fusion.yaml`). All four strategy variants
 * (`single` | `failover` | `fusion` | `smart`) are modelled here as a zod
 * discriminated union so later phases only add *executors*, never schema.
 * Unknown keys are rejected (`.strict()`) with a precise path.
 */

const SingleModelSchema = z
  .object({
    strategy: z.literal("single"),
    target: z.string().min(1),
  })
  .strict();

const FailoverModelSchema = z
  .object({
    strategy: z.literal("failover"),
    chain: z.array(z.string().min(1)).min(1),
  })
  .strict();

const FusionModelSchema = z
  .object({
    strategy: z.literal("fusion"),
    panel: z.array(z.string().min(1)).min(1),
    judge: z.string().min(1),
    synth: z.string().min(1),
    tool_mode: z.enum(["deliberate", "bypass"]).default("deliberate"),
    fusion_planning_turn_only: z.boolean().default(false),
    // Per-model override of `defaults.promote_reasoning_to_content`. When unset
    // the global default applies.
    promote_reasoning_to_content: z.boolean().optional(),
  })
  .strict();

/** Inline single-strategy block usable inside a `smart` model's `simple` slot. */
const SimpleBlockSchema = z.object({ target: z.string().min(1) }).strict();

/** Inline fusion-strategy block usable inside a `smart` model's `fusion` slot. */
const FusionBlockSchema = z
  .object({
    panel: z.array(z.string().min(1)).min(1),
    judge: z.string().min(1),
    synth: z.string().min(1),
    promote_reasoning_to_content: z.boolean().optional(),
  })
  .strict();

const SmartModelSchema = z
  .object({
    strategy: z.literal("smart"),
    router: z.string().min(1),
    default: z.enum(["simple", "fusion"]).default("simple"),
    // Agent-loop escalation: when the latest tool result in the conversation
    // looks like a failure (error, exception, non-zero exit, test failure), the
    // model is recovering from an error — the step that benefits most from
    // deliberation — so smart routes straight to `fusion`, skipping the router
    // round-trip. Set `false` to always defer to the router instead.
    escalate_on_tool_error: z.boolean().default(true),
    // Either an inline strategy block OR a string naming another configured model.
    simple: z.union([SimpleBlockSchema, z.string().min(1)]),
    fusion: z.union([FusionBlockSchema, z.string().min(1)]),
  })
  .strict();

export const ModelSchema = z.discriminatedUnion("strategy", [
  SingleModelSchema,
  FailoverModelSchema,
  FusionModelSchema,
  SmartModelSchema,
]);

const OverrideSchema = z
  .object({
    tools: z.boolean().optional(),
    vision: z.boolean().optional(),
    context: z.number().int().nullable().optional(),
  })
  .strict();

/**
 * Optional per-model price, USD per 1M tokens. When a model involved in a
 * request has an entry, the proxy computes `cost_usd` for the request; absent
 * pricing leaves cost null. Keyed by the REAL upstream model id (the same id
 * used in panel/judge/synth/target/chain), not the virtual model name.
 */
const PricingEntrySchema = z
  .object({
    input_per_mtok: z.number().nonnegative(),
    output_per_mtok: z.number().nonnegative(),
  })
  .strict();

const UpstreamSchema = z
  .object({
    base_url: z.string().url(),
    api_key_env: z.string().min(1),
    api_mode: z.enum(["auto", "openai", "native"]).default("auto"),
    max_concurrency: z.number().int().min(1).default(4),
    // Strictly below the ~182 s Ollama Cloud server-side ceiling (A-6).
    request_timeout_s: z.number().int().positive().lt(182).default(170),
  })
  .strict();

const ServerSchema = z
  .object({
    bind: z.string().min(1).default("127.0.0.1"),
    port: z.number().int().min(1).max(65535).default(8080),
    auth_token_env: z.string().min(1).optional(),
  })
  .strict();

const DefaultsSchema = z
  .object({
    panel_member_timeout_s: z.number().int().positive().default(90),
    judge_timeout_s: z.number().int().positive().default(60),
    min_panel_success: z.number().int().min(1).default(1),
    // When true (default), reasoning-only upstream replies (final text in the
    // `reasoning`/`reasoning_content` field with empty `content`) are normalized
    // so clients that render only `message.content` still see the answer. A
    // fusion model may override this per-model.
    promote_reasoning_to_content: z.boolean().default(true),
  })
  .strict();

export const ConfigSchema = z
  .object({
    upstream: UpstreamSchema,
    server: ServerSchema.default({}),
    defaults: DefaultsSchema.default({}),
    models: z.record(z.string(), ModelSchema),
    overrides: z.record(z.string(), OverrideSchema).default({}),
    // Optional cost accounting. Absent/empty -> cost_usd stays null.
    pricing: z.record(z.string(), PricingEntrySchema).optional(),
  })
  .strict()
  .superRefine((cfg, ctx) => {
    // A `smart` model may reference other configured models by name for its
    // `simple` / `fusion` slots; those references must resolve.
    for (const [name, entry] of Object.entries(cfg.models)) {
      if (entry.strategy !== "smart") continue;
      checkSmartReference(cfg.models, ctx, name, "simple", entry.simple);
      checkSmartReference(cfg.models, ctx, name, "fusion", entry.fusion);
    }
  });

function checkSmartReference(
  models: Record<string, ModelConfig>,
  ctx: z.RefinementCtx,
  modelName: string,
  role: "simple" | "fusion",
  ref: SimpleModelReference | FusionModelReference,
): void {
  if (typeof ref !== "string") return;
  const target = models[ref];
  if (!target) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["models", modelName, role],
      message: `smart model '${modelName}' references unknown model '${ref}' for '${role}'`,
    });
    return;
  }
  // A smart model may not chain to another smart model (incl. itself); the
  // router would recurse with no base case.
  if (target.strategy === "smart") {
    const selfNote = ref === modelName ? " (self-reference)" : "";
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["models", modelName, role],
      message: `smart model '${modelName}' references '${ref}' for '${role}', but '${ref}' is itself a 'smart' model; smart models cannot reference other smart models${selfNote}`,
    });
    return;
  }
  // Referenced model must be strategy-compatible: simple -> single, fusion -> fusion.
  const expected = role === "simple" ? "single" : "fusion";
  if (target.strategy !== expected) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["models", modelName, role],
      message: `smart model '${modelName}' references '${ref}' for '${role}', which must point to a '${expected}' model but '${ref}' has strategy '${target.strategy}'`,
    });
  }
}

export type Config = z.infer<typeof ConfigSchema>;
export type ModelConfig = z.infer<typeof ModelSchema>;
export type SingleModelConfig = z.infer<typeof SingleModelSchema>;
export type FailoverModelConfig = z.infer<typeof FailoverModelSchema>;
export type FusionModelConfig = z.infer<typeof FusionModelSchema>;
export type SmartModelConfig = z.infer<typeof SmartModelSchema>;
export type OverrideConfig = z.infer<typeof OverrideSchema>;
export type PricingConfig = z.infer<typeof PricingEntrySchema>;
export type SimpleBlockConfig = z.infer<typeof SimpleBlockSchema>;
export type FusionBlockConfig = z.infer<typeof FusionBlockSchema>;
export type SimpleModelReference = SimpleBlockConfig | string;
export type FusionModelReference = FusionBlockConfig | string;

/** Validate an already-parsed object against the schema. Throws on invalid input. */
export function parseConfig(raw: unknown): Config {
  return ConfigSchema.parse(raw);
}

/** Read + parse + validate a YAML config file. */
export async function loadConfigFile(path: string): Promise<Config> {
  const text = await readFile(path, "utf8");
  const raw: unknown = parseYaml(text);
  return parseConfig(raw);
}

export interface ConfigManager {
  readonly config: Config;
  onReload(listener: () => void): void;
  close(): void;
}

/**
 * Load a config file and watch it for changes (basic hot-reload via `fs.watch`).
 * On a valid change the in-memory config is swapped atomically; on an invalid
 * change the previous config is kept and the error is logged.
 */
export async function createConfigManager(path: string, logger: Logger): Promise<ConfigManager> {
  let current = await loadConfigFile(path);
  const listeners: Array<() => void> = [];
  let timer: NodeJS.Timeout | undefined;

  async function reload(): Promise<void> {
    try {
      current = await loadConfigFile(path);
      logger.info("configuration reloaded");
      for (const listener of listeners) listener();
    } catch (err) {
      logger.error(
        { err: err instanceof Error ? err.message : String(err) },
        "configuration reload rejected; keeping previous config",
      );
    }
  }

  // Watch the containing DIRECTORY, not the file. An atomic save (write-temp +
  // rename over the path, which most editors and our own file tooling do) replaces
  // the file's inode; a file-level `fs.watch` then goes deaf on some platforms
  // (notably macOS, where it kqueues the inode), so hot-reload fires only once per
  // process and every later edit is silently missed. A directory watch survives the
  // replace (the directory inode is stable); we filter events down to our file.
  const dir = dirname(path);
  const base = basename(path);
  let watcher: FSWatcher | undefined;
  try {
    watcher = watch(dir, { persistent: false }, (_event, filename) => {
      // `filename` can be null on some platforms; when present, ignore siblings.
      if (filename !== null && filename !== base) return;
      // Debounce: a single save emits several events.
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        void reload();
      }, 120);
    });
  } catch (err) {
    logger.error(
      { err: err instanceof Error ? err.message : String(err) },
      "config watch failed to arm; hot-reload disabled",
    );
  }

  return {
    get config() {
      return current;
    },
    onReload(listener: () => void) {
      listeners.push(listener);
    },
    close() {
      if (timer) clearTimeout(timer);
      try {
        watcher?.close();
      } catch {
        /* already closed */
      }
    },
  };
}
