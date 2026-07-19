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
    provider: z.string().min(1).optional(),
    target: z.string().min(1),
    // Per-model override of `defaults.promote_reasoning_to_content`.
    promote_reasoning_to_content: z.boolean().optional(),
    // Extra request-body fields merged into every upstream call for this model
    // (e.g. { reasoning_effort: "none" } to suppress a thinking model's
    // deliberation on mechanical agent steps). Core keys (model, messages,
    // stream, tools, tool_choice) are protected and cannot be overridden.
    request_overrides: z.record(z.string(), z.unknown()).optional(),
  })
  .strict();

const FailoverModelSchema = z
  .object({
    strategy: z.literal("failover"),
    provider: z.string().min(1).optional(),
    chain: z.array(z.string().min(1)).min(1),
    // Per-model override of `defaults.promote_reasoning_to_content`.
    promote_reasoning_to_content: z.boolean().optional(),
  })
  .strict();

const BinevalDimensionSchema = z
  .object({
    dimension: z.string().min(1),
    question: z.string().min(1),
  })
  .strict();

const BinevalSchema = z
  .object({
    enabled: z.boolean().default(false),
    // Model to run the evaluator. Defaults to the fusion judge model.
    model: z.string().min(1).optional(),
    // Overall score below this marks the output as low-quality (surfaced in headers).
    threshold: z.number().min(0).max(1).default(0.7),
    // Per-evaluation timeout. Defaults to the global judge timeout.
    timeout_s: z.number().int().positive().lt(182).optional(),
    // Custom binary questions. When absent, the built-in DEFAULT_DIMENSIONS are used.
    dimensions: z.array(BinevalDimensionSchema).min(1).optional(),
  })
  .strict();

/**
 * Optional web grounding for a fusion panel: one Tavily search before the panel
 * fans out, results injected as prose context. Requires TAVILY_API_KEY in the
 * environment; without it the feature stays OFF even when enabled here. Shared
 * between the top-level fusion model schema and the inline smart-fusion block so
 * web grounding does not depend on whether `smart` references a fusion model by
 * string or defines an inline block.
 */
const WebSearchSchema = z
  .object({
    enabled: z.boolean().default(false),
    max_results: z.number().int().min(1).max(10).default(3),
    timeout_s: z.number().int().positive().lt(60).default(20),
    max_context_chars: z.number().int().positive().default(4000),
    // Skip web grounding when the request is already this large (chars, ≈4
    // chars/token), so the added context can't overflow a smaller-context panel
    // member mid-loop. Short planning turns still ground.
    max_prompt_chars: z.number().int().positive().default(80000),
  })
  .optional();

const FusionModelSchema = z
  .object({
    strategy: z.literal("fusion"),
    provider: z.string().min(1).optional(),
    panel: z.array(z.string().min(1)).min(1),
    judge: z.string().min(1),
    synth: z.string().min(1),
    tool_mode: z.enum(["deliberate", "bypass"]).default("deliberate"),
    fusion_planning_turn_only: z.boolean().default(false),
    // Per-model override of `defaults.promote_reasoning_to_content`. When unset
    // the global default applies.
    promote_reasoning_to_content: z.boolean().optional(),
    web_search: WebSearchSchema,
    // Optional adversarial panel member: the name of a model ALREADY listed in
    // `panel` that should run with a red-team/contrarian prompt (find flaws, hidden
    // assumptions, edge cases) instead of just answering. Addresses the "fake
    // consensus" / shared-priors failure mode by forcing one seat to disagree on
    // purpose. Validated to be a panel member (see superRefine below).
    adversarial: z.string().min(1).optional(),
    // Optional BinEval post-synth quality evaluation (non-streaming only).
    bineval: BinevalSchema.optional(),
  })
  .strict();

/** Inline single-strategy block usable inside a `smart` model's `simple` slot. */
const SimpleBlockSchema = z
  .object({
    target: z.string().min(1),
    // Same semantics as SingleModelSchema.request_overrides — flows through to
    // the resolved single config so the smart simple route can e.g. suppress a
    // thinking model's deliberation ({ reasoning_effort: "none" }).
    request_overrides: z.record(z.string(), z.unknown()).optional(),
  })
  .strict();

/** Inline fusion-strategy block usable inside a `smart` model's `fusion` slot. */
const FusionBlockSchema = z
  .object({
    panel: z.array(z.string().min(1)).min(1),
    judge: z.string().min(1),
    synth: z.string().min(1),
    promote_reasoning_to_content: z.boolean().optional(),
    web_search: WebSearchSchema,
    bineval: BinevalSchema.optional(),
  })
  .strict();

const SmartModelSchema = z
  .object({
    strategy: z.literal("smart"),
    provider: z.string().min(1).optional(),
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

/**
 * A base_url that carries the upstream API key as `Authorization: Bearer`. A
 * compromised/edited base_url exfiltrates that key to whatever host it names —
 * and the panel's no-YAML editor can rewrite it — so we constrain it at the
 * schema seam: HTTPS only (the key never crosses the wire in cleartext), with an
 * explicit exception for a loopback http:// host (local Ollama on 127.0.0.1 /
 * localhost / ::1), and NO embedded userinfo (`https://user:pass@host` is both a
 * credential-leak and an SSRF-obfuscation vector). Defense-in-depth behind the
 * admin-API auth/Origin/Host guards, not a substitute for them.
 */
const LOOPBACK_HOSTNAME = new RegExp(
  `^(127\\.(25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.(25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.(25[0-5]|2[0-4]\\d|1?\\d?\\d)|localhost|::1|0:0:0:0:0:0:0:1)$`,
  "i",
);
const baseUrlSchema = z
  .string()
  .url()
  .refine(
    (u) => {
      let parsed: URL;
      try {
        parsed = new URL(u);
      } catch {
        return false;
      }
      if (parsed.username || parsed.password) return false; // no embedded credentials
      if (parsed.protocol === "https:") return true;
      // Plain http is allowed ONLY for a loopback host (local Ollama). Node's
      // URL.hostname keeps the brackets on an IPv6 host (`[::1]`), so strip them
      // before matching — otherwise `http://[::1]:11434` fails the exception.
      const host = parsed.hostname.replace(/^\[|\]$/g, "");
      return parsed.protocol === "http:" && LOOPBACK_HOSTNAME.test(host);
    },
    { message: "base_url must be https:// (http:// only for a loopback host) and must not embed credentials" },
  );

const UpstreamSchema = z
  .object({
    // base_url / api_key_env describe the DEFAULT single connector. They are
    // optional when a top-level `connectors:` list is present (which supersedes
    // them); required otherwise (validated in superRefine).
    base_url: baseUrlSchema.optional(),
    api_key_env: z.string().min(1).optional(),
    api_mode: z.enum(["auto", "openai", "native"]).default("auto"),
    max_concurrency: z.number().int().min(1).default(4),
    // Per-model concurrency budgets (keyed by REAL upstream model name): a
    // burst on one model queues at its own gate instead of head-of-line
    // blocking every other model in the global queue. Unset = a model may use
    // the full global budget (behavior identical to a single global limiter).
    per_model_concurrency: z.record(z.number().int().min(1)).optional(),
    per_model_concurrency_default: z.number().int().min(1).optional(),
    // Strictly below the ~182 s Ollama Cloud server-side ceiling (A-6).
    request_timeout_s: z.number().int().positive().lt(182).default(170),
    // Connector-pool failover tuning (see connectors + the panel):
    //  - cooldown for a SOFT failure (rate_limit/server/network/timeout) before a
    //    connector is probed again;
    //  - recheck window for a HARD failure (auth/payment/quota) — the "billing
    //    ended" case — before an automatic probe. 0 = never auto-probe (manual
    //    reset only).
    connector_cooldown_s: z.number().int().positive().default(60),
    connector_down_recheck_s: z.number().int().min(0).default(900),
  })
  .strict();

/**
 * A single account of a provider — one credential/endpoint the proxy fails over
 * across WITHIN its provider group (same provider = same models). Holds only the
 * env-var NAME of the key, never the key itself.
 */
const AccountSchema = z
  .object({
    id: z.string().min(1),
    api_key_env: z.string().min(1),
    // Override the provider group's `base_url` for this one account (rare).
    base_url: baseUrlSchema.optional(),
    // Per-account override of the pool-wide `upstream.request_timeout_s`.
    request_timeout_s: z.number().int().positive().lt(182).optional(),
    // Logical→upstream model-id map (e.g. `qwen3-coder:480b`→`qwen/qwen3-coder`).
    // Identity when a model is absent from the map.
    model_map: z.record(z.string(), z.string()).optional(),
    // Extra request headers (e.g. OpenRouter ranking headers HTTP-Referer/X-Title).
    extra_headers: z.record(z.string(), z.string()).optional(),
    // How to treat a 403: `passthrough` (client error, account stays healthy) or
    // `down` (some gateways use 403 for no-credits/disabled).
    treat_403_as: z.enum(["passthrough", "down"]).default("passthrough"),
    // Opt-in lowercased substrings that escalate a 429 body from a transient
    // rate-limit to `quota` (account down). Empty = never escalate on text.
    quota_markers: z.array(z.string().min(1)).default([]),
  })
  .strict();

/**
 * A provider group — one upstream provider (Ollama Cloud, OpenRouter, …) and its
 * ordered list of interchangeable ACCOUNTS. Failover happens WITHIN a group, so
 * every account of a group serves the SAME models: a fusion built on this group
 * stays consistent when one account degrades and the next takes over. A virtual
 * model is bound to exactly one provider group (see each model's `provider`), so
 * it never silently jumps to a provider with a different model catalog.
 */
const ProviderSchema = z
  .object({
    // `ollama` = native-capable (adds /api/show discovery + /api/chat vision).
    // `openai-compat` = the generic OpenAI `/v1/chat/completions` provider that
    // covers OpenRouter, DeepInfra, Together, Novita, Nebius, Groq, DeepSeek, …
    type: z.enum(["ollama", "openai-compat"]).default("ollama"),
    // Group-level default base URL; an account may override it. Required unless
    // every account sets its own `base_url` (validated in superRefine).
    base_url: baseUrlSchema.optional(),
    accounts: z.array(AccountSchema).min(1),
  })
  .strict();

const ServerSchema = z
  .object({
    bind: z.string().min(1).default("127.0.0.1"),
    port: z.number().int().min(1).max(65535).default(8080),
    auth_token_env: z.string().min(1).optional(),
    // Optional SEPARATE token for the admin surface (/admin/* + the panel/WebUI),
    // naming the env var that holds it. When set, the admin API authenticates with
    // THIS token instead of the client `auth_token_env` — so the widely-copied
    // client API token (in every LLM client config) does not also grant config
    // edits + restart. When UNSET the admin surface falls back to `auth_token_env`
    // (backward compatible), and when neither resolves it is loopback-only.
    admin_token_env: z.string().min(1).optional(),
  })
  .strict();

const DefaultsSchema = z
  .object({
    panel_member_timeout_s: z.number().int().positive().default(90),
    judge_timeout_s: z.number().int().positive().default(60),
    // Stage timeout for the smart router's single classification call. Bounds the
    // router independently of the full upstream request timeout, so a slow router
    // cannot hang the whole request; on timeout it degrades to the default route.
    router_timeout_s: z.number().int().positive().default(30),
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
    // Provider groups: each is one upstream provider (Ollama Cloud, OpenRouter, …)
    // and its ordered accounts, which the proxy fails over across WITHIN the group
    // (same models). When absent, a single `default` provider is synthesised from
    // the legacy `upstream.base_url` + `upstream.api_key_env` (backward compatible).
    providers: z.record(z.string().min(1), ProviderSchema).optional(),
  })
  .strict()
  .superRefine((cfg, ctx) => {
    // Provider groups: either explicit `providers:`, or the single `default`
    // provider implied by `upstream.base_url` + `upstream.api_key_env`.
    const groupIds = new Set<string>();
    if (cfg.providers && Object.keys(cfg.providers).length > 0) {
      const seenAccounts = new Set<string>();
      for (const [gid, group] of Object.entries(cfg.providers)) {
        groupIds.add(gid);
        group.accounts.forEach((acc, i) => {
          if (seenAccounts.has(acc.id)) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              path: ["providers", gid, "accounts", i, "id"],
              message: `duplicate account id '${acc.id}'; account ids must be unique across ALL providers`,
            });
          }
          seenAccounts.add(acc.id);
          if (!acc.base_url && !group.base_url) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              path: ["providers", gid, "accounts", i, "base_url"],
              message: `account '${acc.id}' has no base_url and provider '${gid}' sets none; set one on either`,
            });
          }
        });
      }
    } else if (cfg.upstream.base_url && cfg.upstream.api_key_env) {
      groupIds.add("default");
    } else {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["providers"],
        message:
          "no providers configured: provide a top-level `providers:` map, or set both `upstream.base_url` and `upstream.api_key_env`",
      });
    }
    // Each model is bound to one provider group. An explicit `provider` must
    // resolve; when omitted it defaults to the sole group (error if ambiguous).
    const soleGroup = groupIds.size === 1 ? [...groupIds][0] : undefined;
    for (const [name, entry] of Object.entries(cfg.models)) {
      if (entry.provider !== undefined) {
        if (!groupIds.has(entry.provider)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["models", name, "provider"],
            message: `model '${name}' is bound to provider '${entry.provider}', which is not defined in \`providers\``,
          });
        }
      } else if (soleGroup === undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["models", name, "provider"],
          message: `model '${name}' must set \`provider\` (there are multiple provider groups; it is ambiguous which one serves it)`,
        });
      }
    }
    // A `smart` model runs its `simple`/`fusion` routes through ITS OWN provider
    // group's pool (dispatch resolves the client from the smart model). So a
    // string-referenced sub-model MUST resolve to the SAME provider group, else
    // its members would be sent to the wrong provider's accounts (404s).
    for (const [name, entry] of Object.entries(cfg.models)) {
      if (entry.strategy !== "smart") continue;
      const smartGroup = entry.provider ?? soleGroup;
      for (const role of ["simple", "fusion"] as const) {
        const slot = entry[role];
        if (typeof slot !== "string") continue; // inline block inherits the smart group
        const ref = cfg.models[slot];
        if (!ref) continue; // unknown-ref already reported elsewhere
        const refGroup = ref.provider ?? soleGroup;
        if (smartGroup !== undefined && refGroup !== undefined && smartGroup !== refGroup) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["models", name, role],
            message: `smart model '${name}' (provider '${smartGroup}') references '${slot}' bound to a different provider group '${refGroup}'; a smart model and its routes must share one provider group`,
          });
        }
      }
    }
    // A `smart` model may reference other configured models by name for its
    // `simple` / `fusion` slots; those references must resolve.
    for (const [name, entry] of Object.entries(cfg.models)) {
      if (entry.strategy !== "smart") continue;
      checkSmartReference(cfg.models, ctx, name, "simple", entry.simple);
      checkSmartReference(cfg.models, ctx, name, "fusion", entry.fusion);
    }
    // A fusion model's `adversarial` member must be one of its panel members, and the
    // panel must not list the same model twice — a duplicate means two identical upstream
    // calls and, for the adversarial slot, only the first copy is protected from early
    // abort (the second is treated as a normal member and can be cancelled).
    for (const [name, entry] of Object.entries(cfg.models)) {
      if (entry.strategy !== "fusion") continue;
      const adv = entry.adversarial;
      if (adv !== undefined && !entry.panel.includes(adv)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["models", name, "adversarial"],
          message: `fusion model '${name}' sets adversarial='${adv}', but '${adv}' is not listed in its panel; adversarial must be an existing panel member`,
        });
      }
      const seen = new Set<string>();
      for (const member of entry.panel) {
        if (seen.has(member)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["models", name, "panel"],
            message: `fusion model '${name}' lists panel member '${member}' more than once; each member must be unique`,
          });
        }
        seen.add(member);
      }
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
  // Referenced model must be strategy-compatible: simple -> single/failover, fusion -> fusion.
  const allowed = role === "simple" ? ["single", "failover"] : ["fusion"];
  if (!allowed.includes(target.strategy)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["models", modelName, role],
      message: `smart model '${modelName}' references '${ref}' for '${role}', which must point to a '${allowed.join(" or ")}' model but '${ref}' has strategy '${target.strategy}'`,
    });
  }
}

export type Config = z.infer<typeof ConfigSchema>;
export type ModelConfig = z.infer<typeof ModelSchema>;
export type ProviderConfig = z.infer<typeof ProviderSchema>;
export type AccountConfig = z.infer<typeof AccountSchema>;
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

/**
 * A rate-limit contention hazard surfaced by {@link findPanelContentionOverlaps}:
 * a `single`/`failover` virtual model resolves to an upstream model that is ALSO
 * a `panel` member of a `fusion` model in the SAME provider group. Because
 * contention is per-provider-model, the two share one upstream rate-limit bucket
 * AND one `per_model_concurrency` gate — so a burst of small-fast traffic through
 * the single/failover model (e.g. Claude Code's 80-130 background small-model
 * calls/min via `ANTHROPIC_SMALL_FAST_MODEL`) can 429-starve the live panel
 * mid-request. See the rate-limit note in `bin/fusion-claude`.
 */
export interface PanelContentionOverlap {
  /** Virtual model (strategy `single` | `failover`) whose target/chain resolves to `target`. */
  fastModel: string;
  /** The shared REAL upstream model id (`single.target` or one `failover.chain` entry). */
  target: string;
  /** Fusion model whose `panel` lists `target` as a member. */
  fusionModel: string;
}

/**
 * Detect rate-limit contention overlaps in a parsed config: every case where a
 * `single`/`failover` virtual model's upstream target is ALSO a `fusion` panel
 * member bound to the SAME provider group. Panel members, `single.target`, and
 * `failover.chain` entries all live in the same namespace (REAL upstream model
 * ids), so they compare directly. Judge/synth overlap is a separate, milder case
 * and is deliberately NOT reported — only panel-member contention, which starves
 * a live panel, matters here.
 *
 * Pure (no I/O); safe to call at startup on a validated config. Returns one entry
 * per (fastModel, target, fusionModel) pair; an empty array means no contention.
 */
export function findPanelContentionOverlaps(cfg: Config): PanelContentionOverlap[] {
  // Resolve each model's effective provider group exactly as the schema does:
  // an explicit `provider`, else the sole group when there is exactly one. Two
  // models contend only when they resolve to the same upstream provider group.
  const groupIds =
    cfg.providers && Object.keys(cfg.providers).length > 0
      ? new Set(Object.keys(cfg.providers))
      : cfg.upstream.base_url && cfg.upstream.api_key_env
        ? new Set(["default"])
        : new Set<string>();
  const soleGroup = groupIds.size === 1 ? [...groupIds][0] : undefined;
  const groupOf = (entry: ModelConfig): string | undefined => entry.provider ?? soleGroup;

  const overlaps: PanelContentionOverlap[] = [];
  for (const [fastModel, entry] of Object.entries(cfg.models)) {
    // Upstream models this single/failover virtual model resolves to.
    let targets: readonly string[];
    if (entry.strategy === "single") targets = [entry.target];
    else if (entry.strategy === "failover") targets = entry.chain;
    else continue;
    const fastGroup = groupOf(entry);

    for (const [fusionModel, other] of Object.entries(cfg.models)) {
      if (other.strategy !== "fusion") continue;
      if (groupOf(other) !== fastGroup) continue; // contention is per-provider-model
      for (const target of targets) {
        if (other.panel.includes(target)) {
          overlaps.push({ fastModel, target, fusionModel });
        }
      }
    }
  }
  return overlaps;
}

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
