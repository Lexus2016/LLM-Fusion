#!/usr/bin/env bash
# launchd wrapper for the llm-fusion proxy.
#   - sources nvm for Node 24
#   - rotates .launchd.log before it grows unbounded
#   - execs the proxy (tsx runs the TypeScript directly; no build step)
#
# The proxy itself loads .env via Node's native process.loadEnvFile().
# Override the config with FUSION_CONFIG (e.g. ./fusion.tuned.yaml).
set -euo pipefail

# Resolve the repo from THIS script's location instead of hardcoding it. The
# path was previously written as `_Projects/LLM-Fusion` while the directory is
# `_Projects/llm-fusion`; that only worked because macOS APFS is
# case-insensitive by default and would break on a case-sensitive volume, in a
# container, or after an rsync to Linux.
REPO="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# --- log rotation ------------------------------------------------------------
# launchd appends to StandardOutPath forever; without this the file grew to
# ~6 MB in three weeks and nothing ever truncated it. Keep one previous
# generation, rotate at 32 MB. Runs before the exec so the live process always
# holds a freshly-opened descriptor.
LOG="$REPO/.launchd.log"
MAX_LOG_BYTES=$((32 * 1024 * 1024))
if [ -f "$LOG" ]; then
  size=$(wc -c < "$LOG" | tr -d ' ')
  if [ "$size" -gt "$MAX_LOG_BYTES" ]; then
    mv -f "$LOG" "$LOG.1"
    : > "$LOG"
  fi
fi

export NVM_DIR="$HOME/.nvm"
# shellcheck source=/dev/null
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 24.16.0 >/dev/null 2>&1 || true

exec node node_modules/tsx/dist/cli.mjs src/index.ts
