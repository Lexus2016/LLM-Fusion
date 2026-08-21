import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { isLoopbackBind, assertBindIsSafe, parseAllowOpen } from "../src/index";

/**
 * Regression lock for the non-loopback fail-fast in `src/index.ts` (the
 * `isLoopbackBind` check + the `refusing to start` throw, ~lines 216-237).
 *
 * That guard is the proxy's last line of defence: without it a misconfigured
 * deployment publishes an UNAUTHENTICATED proxy — plus the admin/config API —
 * on a routable interface, billed to the operator's upstream key. It had zero
 * test coverage, so any refactor that inverted the condition shipped green.
 *
 * BLACK BOX, deliberately. Every case in the first suite spawns the real
 * entrypoint with real env vars and inspects exit code / stdout / stderr, so it
 * covers the whole wiring — FUSION_BIND vs config, token resolution, the
 * escape hatch, the exit code — not just the predicate. It is what caught the
 * mutations; keep it.
 *
 * `isLoopbackBind` / `assertBindIsSafe` are now exported from `src/index.ts`
 * (which only calls `main()` when it IS the process entrypoint, so importing it
 * here is inert). The second suite below unit-tests them directly: same guard,
 * microseconds instead of a process, and it can reach the address-parsing edge
 * cases that are impractical to spawn for.
 *
 * The observable boundary is the startup banner: `logger.info(..., "llm-fusion
 * starting")` is emitted immediately AFTER the guard and BEFORE `serve()`.
 *   - banner seen        → the guard let the process through ("allowed to start")
 *   - "refusing to start" on stderr + exit 1 → the guard refused
 * Using the banner (not a successful listen) keeps the non-loopback "allowed"
 * cases honest on CI boxes where 192.168.1.10 / ::ffff:127.0.0.1 are not
 * assignable — those fail in `serve()`, well past the code under test.
 */

const REPO = resolve(__dirname, "..");
const ENTRY = join(REPO, "src", "index.ts");
const TSX = join(REPO, "node_modules", ".bin", "tsx");

const SPAWN_TIMEOUT_MS = 30_000;

interface Outcome {
  /** Process exit code (null if we had to kill it — i.e. it started and stayed up). */
  code: number | null;
  stdout: string;
  stderr: string;
  /** The guard let it through: the post-guard startup banner was emitted. */
  started: boolean;
  /** The guard threw: `main().catch` printed the message and exited non-zero. */
  refused: boolean;
  /** `bind` field of the startup banner — what the process actually resolved. */
  bannerBind?: string;
}

interface Scenario {
  name: string;
  /** `server.bind` written into the temp fusion.yaml (default 127.0.0.1 if omitted). */
  configBind?: string;
  /** `server.auth_token_env` written into the temp fusion.yaml. */
  authTokenEnv?: string;
  /** Extra env for the child. `""` values are passed through as empty strings. */
  env?: Record<string, string>;
}

/**
 * Base env for the child: the real environment minus anything that could make a
 * case pass or fail for the wrong reason (a developer's exported FUSION_* vars,
 * LOG_PRETTY spawning a pino worker, LOG_LEVEL silencing the banner).
 */
function baseEnv(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v === undefined) continue;
    if (k.startsWith("FUSION_")) continue;
    if (k === "LOG_LEVEL" || k === "LOG_PRETTY") continue;
    out[k] = v;
  }
  out.LOG_LEVEL = "info";
  return out;
}

function yamlFor(s: Scenario, port: number): string {
  const auth = s.authTokenEnv ? `  auth_token_env: ${s.authTokenEnv}\n` : "";
  return (
    `upstream:\n  base_url: https://ollama.com\n  api_key_env: OLLAMA_API_KEY_UNSET_IN_TEST\n` +
    `server:\n  bind: ${JSON.stringify(s.configBind ?? "127.0.0.1")}\n  port: ${port}\n${auth}` +
    `models:\n  fast:\n    strategy: single\n    target: glm-5.2\n`
  );
}

const dirs: string[] = [];

/**
 * Spawn the real entrypoint once and resolve as soon as the guard's verdict is
 * observable (banner or refusal), then kill it. cwd is a throwaway temp dir so
 * `process.loadEnvFile()` cannot pick up the repo's own `.env`.
 */
function run(s: Scenario, port: number): Promise<Outcome> {
  const dir = mkdtempSync(join(tmpdir(), "llm-fusion-bind-"));
  dirs.push(dir);
  const cfgPath = join(dir, "fusion.yaml");
  writeFileSync(cfgPath, yamlFor(s, port));

  const env = { ...baseEnv(), FUSION_CONFIG: cfgPath, ...(s.env ?? {}) };

  return new Promise<Outcome>((resolvePromise, rejectPromise) => {
    const child = spawn(TSX, [ENTRY], { cwd: dir, env, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (code: number | null): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const all = `${stdout}${stderr}`;
      let bannerBind: string | undefined;
      for (const line of all.split("\n")) {
        if (!line.includes("llm-fusion starting")) continue;
        try {
          const parsed: unknown = JSON.parse(line);
          if (parsed && typeof parsed === "object" && "bind" in parsed) {
            const b = (parsed as Record<string, unknown>).bind;
            if (typeof b === "string") bannerBind = b;
          }
        } catch {
          /* not JSON (pretty mode) — leave bannerBind undefined */
        }
      }
      resolvePromise({
        code,
        stdout,
        stderr,
        started: all.includes("llm-fusion starting"),
        refused: all.includes("refusing to start"),
        bannerBind,
      });
      child.kill("SIGKILL");
    };

    const timer = setTimeout(() => {
      finish(null);
    }, SPAWN_TIMEOUT_MS - 5_000);

    child.stdout.on("data", (b: Buffer) => {
      stdout += b.toString();
      // The banner is emitted right after the guard, before serve() — the
      // process would otherwise stay up forever holding the test open.
      if (stdout.includes("llm-fusion starting")) finish(null);
    });
    child.stderr.on("data", (b: Buffer) => {
      stderr += b.toString();
    });
    child.on("error", rejectPromise);
    child.on("exit", (code) => finish(code));
  });
}

/** Cases that must be ALLOWED past the guard (banner reached). */
const ALLOWED: Scenario[] = [
  { name: "loopback 127.0.0.1, no auth", configBind: "127.0.0.1" },
  { name: "loopback localhost, no auth", configBind: "localhost" },
  { name: "loopback ::1, no auth", configBind: "::1" },
  { name: "loopback 0:0:0:0:0:0:0:1 (expanded IPv6), no auth", configBind: "0:0:0:0:0:0:0:1" },
  { name: "loopback 127.0.0.5 (whole 127.0.0.0/8 is loopback), no auth", configBind: "127.0.0.5" },
  { name: "loopback ::ffff:127.0.0.1 (IPv4-mapped), no auth", configBind: "::ffff:127.0.0.1" },
  {
    name: "non-loopback 0.0.0.0 WITH a resolving auth token",
    configBind: "0.0.0.0",
    authTokenEnv: "FUSION_TEST_TOKEN",
    env: { FUSION_TEST_TOKEN: "s3cret" },
  },
  {
    name: "non-loopback 192.168.1.10 WITH a resolving auth token",
    configBind: "192.168.1.10",
    authTokenEnv: "FUSION_TEST_TOKEN",
    env: { FUSION_TEST_TOKEN: "s3cret" },
  },
  {
    name: "non-loopback 0.0.0.0, no auth, FUSION_ALLOW_OPEN=1 (explicit escape hatch)",
    configBind: "0.0.0.0",
    env: { FUSION_ALLOW_OPEN: "1" },
  },
  {
    name: "FUSION_BIND=0.0.0.0 over a loopback config, no auth, FUSION_ALLOW_OPEN=1",
    configBind: "127.0.0.1",
    env: { FUSION_BIND: "0.0.0.0", FUSION_ALLOW_OPEN: "1" },
  },
];

/** Cases that must be REFUSED (throw → `main().catch` → exit 1). */
const REFUSED: Scenario[] = [
  { name: "non-loopback 0.0.0.0, no auth, no escape hatch", configBind: "0.0.0.0" },
  { name: "non-loopback 192.168.1.10, no auth, no escape hatch", configBind: "192.168.1.10" },
  { name: "non-loopback :: (IPv6 any), no auth, no escape hatch", configBind: "::" },
  {
    name: "FUSION_BIND=0.0.0.0 overriding a loopback config, no auth",
    configBind: "127.0.0.1",
    env: { FUSION_BIND: "0.0.0.0" },
  },
  {
    name: "non-loopback + auth_token_env naming an UNSET var (fails closed, not open)",
    configBind: "0.0.0.0",
    authTokenEnv: "FUSION_TOKEN_THAT_IS_NEVER_SET",
  },
  {
    name: "non-loopback + auth_token_env set to the EMPTY string (fails closed)",
    configBind: "0.0.0.0",
    authTokenEnv: "FUSION_TEST_TOKEN",
    env: { FUSION_TEST_TOKEN: "" },
  },
  {
    name: "non-loopback, no auth, FUSION_ALLOW_OPEN='' (empty is NOT an opt-out)",
    configBind: "0.0.0.0",
    env: { FUSION_ALLOW_OPEN: "" },
  },
  {
    // The one spelling an operator reaches for to turn the hatch OFF. Under the old
    // `Boolean(process.env.FUSION_ALLOW_OPEN)` it turned it ON and published an
    // unauthenticated proxy on 0.0.0.0. Spawned rather than unit-asserted because
    // this is precisely the wiring — env var to guard — that a unit test cannot see.
    name: "non-loopback, no auth, FUSION_ALLOW_OPEN=0 (a negative spelling must NOT open the proxy)",
    configBind: "0.0.0.0",
    env: { FUSION_ALLOW_OPEN: "0" },
  },
];

/**
 * The `||` vs `??` case, called out separately so a refactor of
 * `process.env.FUSION_BIND || manager.config.server.bind` to `??` fails loudly
 * and by name. With `??`, an empty FUSION_BIND becomes the literal "" — which
 * several servers treat as "all interfaces" — instead of falling through to the
 * configured bind.
 */
const EMPTY_BIND_FALLTHROUGH: Scenario = {
  name: "FUSION_BIND='' falls through to the configured loopback bind",
  configBind: "127.0.0.1",
  env: { FUSION_BIND: "" },
};
const EMPTY_BIND_KEEPS_CONFIG: Scenario = {
  name: "FUSION_BIND='' falls through to a configured NON-loopback bind (still refuses, names it)",
  configBind: "0.0.0.0",
  env: { FUSION_BIND: "" },
};

const outcomes = new Map<string, Outcome>();

beforeAll(async () => {
  const all = [...ALLOWED, ...REFUSED, EMPTY_BIND_FALLTHROUGH, EMPTY_BIND_KEEPS_CONFIG];
  // Parallel, but capped: each case spawns a real `tsx` process (~0.6s of
  // startup), and nothing here shares state (own temp dir, own config, own
  // port). The cap keeps this file from saturating the box and destabilising
  // timing-sensitive tests running alongside it in the same suite.
  const results: Outcome[] = [];
  const LANES = 6;
  for (let i = 0; i < all.length; i += LANES) {
    const batch = all.slice(i, i + LANES);
    results.push(...(await Promise.all(batch.map((s, j) => run(s, 18400 + i + j)))));
  }
  all.forEach((s, i) => {
    const r = results[i];
    if (r) outcomes.set(s.name, r);
  });
}, 60_000);

afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

function got(name: string): Outcome {
  const o = outcomes.get(name);
  if (!o) throw new Error(`no recorded outcome for '${name}'`);
  return o;
}

describe("non-loopback bind guard (src/index.ts)", () => {
  it.each(ALLOWED.map((s) => [s.name] as const))("ALLOWS start: %s", (name) => {
    const o = got(name);
    expect(o.refused, `unexpected refusal\nstderr:\n${o.stderr}`).toBe(false);
    expect(o.started, `never reached the startup banner\nstdout:\n${o.stdout}\nstderr:\n${o.stderr}`).toBe(
      true,
    );
  });

  it.each(REFUSED.map((s) => [s.name] as const))("REFUSES to start: %s", (name) => {
    const o = got(name);
    expect(o.started, `reached the startup banner — the guard did NOT fire\nstdout:\n${o.stdout}`).toBe(
      false,
    );
    expect(o.refused, `no 'refusing to start' on stderr\nstderr:\n${o.stderr}`).toBe(true);
    expect(o.code, "must exit non-zero so a supervisor does not treat it as a clean stop").toBe(1);
  });

  it("the refusal message names the offending bind and all three remedies", () => {
    const o = got("non-loopback 0.0.0.0, no auth, no escape hatch");
    expect(o.stderr).toContain("bind '0.0.0.0' is not loopback");
    expect(o.stderr).toContain("server.auth_token_env");
    expect(o.stderr).toContain("127.0.0.1");
    expect(o.stderr).toContain("FUSION_ALLOW_OPEN=1");
  });

  it("FUSION_BIND='' must fall back to the configured bind, NOT to all-interfaces (|| not ??)", () => {
    const o = got(EMPTY_BIND_FALLTHROUGH.name);
    // With `??` the bind becomes "" — not loopback, no auth, no hatch — so the
    // guard refuses and this assertion fails. With `||` it is 127.0.0.1.
    expect(o.refused, `refused with an empty FUSION_BIND — '||' was probably changed to '??'\nstderr:\n${o.stderr}`).toBe(
      false,
    );
    expect(o.started).toBe(true);
    // Strongest form: the process must have RESOLVED the configured bind, not "".
    expect(o.bannerBind).toBe("127.0.0.1");
  });

  it("FUSION_BIND='' with a non-loopback config still refuses, and names the CONFIG bind", () => {
    const o = got(EMPTY_BIND_KEEPS_CONFIG.name);
    expect(o.refused).toBe(true);
    expect(o.code).toBe(1);
    // With `??` this would read `bind ''` — the message pins the fall-through.
    expect(o.stderr).toContain("bind '0.0.0.0' is not loopback");
    expect(o.stderr).not.toContain("bind '' is not loopback");
  });

  it("an unauthenticated loopback start still warns loudly that the proxy is open", () => {
    const o = got("loopback 127.0.0.1, no auth");
    expect(`${o.stdout}${o.stderr}`).toContain("UNAUTHENTICATED");
  });

  it("FUSION_ALLOW_OPEN warns that client auth is disabled", () => {
    const o = got("non-loopback 0.0.0.0, no auth, FUSION_ALLOW_OPEN=1 (explicit escape hatch)");
    expect(`${o.stdout}${o.stderr}`).toContain("client auth is DISABLED");
  });
});

/**
 * Fast unit suite over the exported seam. Complements — does not replace — the
 * spawn suite above: this one pins the ADDRESS PARSING, which is where the
 * fail-safe was actually weak.
 */
describe("isLoopbackBind (unit)", () => {
  const LOOPBACK = [
    "127.0.0.1",
    "localhost",
    "::1",
    "0:0:0:0:0:0:0:1",
    "127.0.0.5",
    "::ffff:127.0.0.1",
  ];

  it.each(LOOPBACK)("treats %s as loopback (must match the spawned ALLOWED cases)", (bind) => {
    expect(isLoopbackBind(bind)).toBe(true);
  });

  it("covers the whole 127.0.0.0/8 block, not just 127.0.0.1", () => {
    expect(isLoopbackBind("127.0.0.2")).toBe(true);
    expect(isLoopbackBind("127.1.2.3")).toBe(true);
    expect(isLoopbackBind("127.255.255.254")).toBe(true);
    expect(isLoopbackBind("::ffff:127.0.0.5")).toBe(true);
  });

  it("accepts the other spellings of ::1", () => {
    expect(isLoopbackBind("0000:0000:0000:0000:0000:0000:0000:0001")).toBe(true);
    expect(isLoopbackBind("::0.0.0.1")).toBe(true); // same 128 bits as ::1
    expect(isLoopbackBind("::1")).toBe(true);
  });

  /**
   * SECURITY REGRESSION LOCK. The guard used to prefix-match strings
   * (`bind.startsWith("127.")`, `bind.startsWith("::ffff:127.")`), so a
   * HOSTNAME that merely begins like a loopback address was accepted as
   * loopback — and an unauthenticated proxy started on whatever that name
   * resolves to. Each of these is a real, registrable hostname shape.
   */
  it.each([
    "127.evil.com",
    "::ffff:127.0.0.1.attacker.tld",
    "127.0.0.1.nip.io",
    "127.0.0.1.example.com",
    "localhost.evil.com",
    "127.",
    "::ffff:127.",
  ])("REJECTS the loopback-lookalike hostname %s (not an IP → not loopback)", (bind) => {
    expect(isLoopbackBind(bind)).toBe(false);
  });

  it.each(["0.0.0.0", "::", "192.168.1.10", ""])("REJECTS non-loopback %s", (bind) => {
    expect(isLoopbackBind(bind)).toBe(false);
  });

  it("rejects other routable / near-miss addresses", () => {
    expect(isLoopbackBind("10.0.0.1")).toBe(false);
    expect(isLoopbackBind("128.0.0.1")).toBe(false);
    expect(isLoopbackBind("12.7.0.1")).toBe(false);
    expect(isLoopbackBind("1270.0.0.1")).toBe(false);
    expect(isLoopbackBind("::2")).toBe(false);
    expect(isLoopbackBind("::ffff:128.0.0.1")).toBe(false);
    expect(isLoopbackBind("2001:db8::1")).toBe(false);
    expect(isLoopbackBind("fe80::1")).toBe(false);
    expect(isLoopbackBind("   127.0.0.1   ")).toBe(false); // no trimming: not an IP
    expect(isLoopbackBind("LOCALHOST")).toBe(false); // the literal is exact, as before
  });
});

describe("assertBindIsSafe (unit)", () => {
  it("allows a loopback bind with no auth and no escape hatch", () => {
    expect(() => {
      assertBindIsSafe("127.0.0.1", false, false);
    }).not.toThrow();
  });

  it("allows a non-loopback bind when auth resolves", () => {
    expect(() => {
      assertBindIsSafe("0.0.0.0", true, false);
    }).not.toThrow();
  });

  it("allows a non-loopback bind under the explicit escape hatch", () => {
    expect(() => {
      assertBindIsSafe("0.0.0.0", false, true);
    }).not.toThrow();
  });

  it("throws for non-loopback + no auth + no hatch, naming the bind and all three remedies", () => {
    // Same substrings the spawn suite asserts on stderr — locked in both places
    // so a reword cannot pass by fixing only one.
    expect(() => {
      assertBindIsSafe("0.0.0.0", false, false);
    }).toThrow("refusing to start");
    expect(() => {
      assertBindIsSafe("0.0.0.0", false, false);
    }).toThrow("bind '0.0.0.0' is not loopback");
    expect(() => {
      assertBindIsSafe("0.0.0.0", false, false);
    }).toThrow("server.auth_token_env");
    expect(() => {
      assertBindIsSafe("0.0.0.0", false, false);
    }).toThrow("FUSION_ALLOW_OPEN=1");
  });

  /**
   * The security fix at the guard level: a lookalike hostname must be refused,
   * not silently published. Fails against the old prefix-matching predicate.
   */
  it.each(["127.evil.com", "::ffff:127.0.0.1.attacker.tld", "127.0.0.1.nip.io"])(
    "refuses to start on the loopback-lookalike hostname %s",
    (bind) => {
      expect(() => {
        assertBindIsSafe(bind, false, false);
      }).toThrow(`bind '${bind}' is not loopback`);
    },
  );

  /**
   * These assert the PRODUCTION parser, not JS `Boolean()` semantics. The previous
   * version of this block computed `Boolean(value)` in the test body and asserted
   * on that, so it stayed green no matter what `src/index.ts` did — it locked
   * nothing. It also documented the defect it was locking: `Boolean("0") === true`
   * meant `FUSION_ALLOW_OPEN=0`, the operator's attempt to CLOSE the hatch, opened
   * an unauthenticated proxy on a routable interface.
   */
  it.each(["1", "true", "TRUE", "yes", "on", " 1 "])(
    "FUSION_ALLOW_OPEN=%s opts in to the open proxy",
    (value) => {
      expect(parseAllowOpen(value)).toBe(true);
      expect(() => {
        assertBindIsSafe("0.0.0.0", false, parseAllowOpen(value));
      }).not.toThrow();
    },
  );

  it.each(["0", "false", "off", "no", "", "  ", "maybe"])(
    "FUSION_ALLOW_OPEN=%s does NOT open the proxy",
    (value) => {
      expect(parseAllowOpen(value)).toBe(false);
      expect(() => {
        assertBindIsSafe("0.0.0.0", false, parseAllowOpen(value));
      }).toThrow("refusing to start");
    },
  );

  it("an unset FUSION_ALLOW_OPEN is NOT an opt-out", () => {
    expect(parseAllowOpen(undefined)).toBe(false);
    expect(() => {
      assertBindIsSafe("0.0.0.0", false, parseAllowOpen(undefined));
    }).toThrow("refusing to start");
  });
});
