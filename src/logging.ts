import pino from "pino";
import type { Logger } from "pino";

/**
 * pino logger with key masking. Any `Authorization` header or key-like field is
 * redacted. Prompt-content logging is OFF by default — message contents are
 * never logged anywhere in the proxy.
 */

const REDACT_PATHS = [
  "authorization",
  "Authorization",
  "*.authorization",
  "*.Authorization",
  "headers.authorization",
  "headers.Authorization",
  "req.headers.authorization",
  "apiKey",
  "api_key",
  "*.apiKey",
  "*.api_key",
  "token",
  "authToken",
  "*.token",
  "*.authToken",
  // Hyphenated header variants (bracket notation — fast-redact rejects bare
  // hyphens). Defense-in-depth: the net should not have holes if future code
  // logs an object carrying these keys.
  '["x-api-key"]',
  '*["x-api-key"]',
  'headers["x-api-key"]',
  '["api-key"]',
  '*["api-key"]',
];

export interface LoggerOptions {
  level?: string;
}

export function createLogger(opts: LoggerOptions = {}): Logger {
  const level = opts.level ?? process.env.LOG_LEVEL ?? "info";
  const options = {
    level,
    base: undefined,
    redact: { paths: REDACT_PATHS, censor: "[redacted]" },
  };
  // Pretty transport is opt-in (LOG_PRETTY=1) so it never spawns a worker
  // thread during tests.
  if (process.env.LOG_PRETTY === "1") {
    return pino({ ...options, transport: { target: "pino-pretty", options: { colorize: true } } });
  }
  return pino(options);
}
