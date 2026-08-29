import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { parse as parseYaml } from "yaml";
import { z } from "zod";
import {
  ConfigSchema,
  parseConfig,
  findPanelContentionOverlaps,
  findUnmatchedConcurrencyKeys,
} from "../src/config";

/**
 * fusion.example.yaml is the file users copy to start from. Three properties
 * have to hold mechanically, because "someone will notice" demonstrably does
 * not: the shipped example drifted for months while describing models that had
 * been retired, a panel composition that no longer existed, and a default
 * (`ANTHROPIC_SMALL_FAST_MODEL` guidance) that had become actively harmful.
 *
 *   1. it parses;
 *   2. it triggers ZERO startup warnings — an example that warns teaches the
 *      shape it warns about;
 *   3. it MENTIONS every key the schema accepts, so a new key cannot be added
 *      to the schema without documenting it here.
 */

const EXAMPLE_PATH = new URL("../fusion.example.yaml", import.meta.url);
const text = readFileSync(EXAMPLE_PATH, "utf8");

/** Every distinct key NAME anywhere in the schema tree (leaf and container). */
function schemaKeyNames(schema: z.ZodTypeAny): Set<string> {
  const names = new Set<string>();
  const seen = new Set<unknown>();

  const strip = (s: z.ZodTypeAny): z.ZodTypeAny => {
    let cur = s;
    for (;;) {
      const def = (cur as { _def?: { typeName?: string; innerType?: z.ZodTypeAny; schema?: z.ZodTypeAny } })._def;
      const t = def?.typeName;
      if (t === "ZodOptional" || t === "ZodDefault" || t === "ZodNullable") {
        cur = def!.innerType!;
        continue;
      }
      if (t === "ZodEffects") {
        cur = def!.schema!;
        continue;
      }
      return cur;
    }
  };

  const walk = (s: z.ZodTypeAny): void => {
    const cur = strip(s);
    if (seen.has(cur)) return;
    seen.add(cur);
    const def = cur._def as {
      typeName?: string;
      shape?: () => Record<string, z.ZodTypeAny>;
      type?: z.ZodTypeAny;
      valueType?: z.ZodTypeAny;
      options?: z.ZodTypeAny[];
    };
    switch (def.typeName) {
      case "ZodObject": {
        const shape = (cur as unknown as z.ZodObject<z.ZodRawShape>).shape;
        for (const [k, v] of Object.entries(shape)) {
          names.add(k);
          walk(v);
        }
        return;
      }
      case "ZodArray":
        return walk(def.type!);
      case "ZodRecord":
        return walk(def.valueType!);
      case "ZodDiscriminatedUnion":
      case "ZodUnion":
        for (const o of def.options ?? []) walk(o);
        return;
      default:
        return;
    }
  };

  walk(schema);
  return names;
}

describe("fusion.example.yaml", () => {
  it("parses and validates against the schema", () => {
    expect(() => parseConfig(parseYaml(text))).not.toThrow();
  });

  it("produces no panel-contention warnings", () => {
    // Every `single`/`failover` target must stay off every live panel: the
    // example is what a user copies for their small-fast model.
    const cfg = parseConfig(parseYaml(text));
    expect(findPanelContentionOverlaps(cfg)).toEqual([]);
  });

  it("has no per_model_concurrency key that gates nothing", () => {
    const cfg = parseConfig(parseYaml(text));
    expect(findUnmatchedConcurrencyKeys(cfg)).toEqual([]);
  });

  it("declares an adversarial member that is actually on its panel", () => {
    const cfg = parseConfig(parseYaml(text));
    for (const [name, m] of Object.entries(cfg.models)) {
      if (m.strategy !== "fusion" || !m.adversarial) continue;
      expect(m.panel, `${name}.adversarial must be a panel member`).toContain(m.adversarial);
    }
  });

  it("mentions every key the schema accepts", () => {
    const documented = new Set<string>();
    for (const line of text.split("\n")) {
      // Matches both live keys and commented-out ones (`# quota_markers: []`),
      // including inline-map keys written as `{ target: …, judge: … }`.
      for (const m of line.matchAll(/(?:^|[\s#{,])([a-z_][a-z0-9_]*)\s*:/g)) {
        documented.add(m[1]!);
      }
    }
    const missing = [...schemaKeyNames(ConfigSchema)].filter((k) => !documented.has(k)).sort();
    expect(missing, `fusion.example.yaml does not mention: ${missing.join(", ")}`).toEqual([]);
  });
});
