#!/bin/bash
set -eu
if [ -n "${!CODEX_*}" ]; then
  echo 'STOP: inherited Codex environment; launch manually in an independent offline Terminal.' >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
if [ "$(uname -s)" != Darwin ] || [ ! -x /usr/bin/sandbox-exec ]; then
  echo 'STOP: macOS sandbox-exec is required; no unsandboxed fallback.' >&2
  exit 2
fi
printf '%s\n' 'Human-only offline tool: disconnect networking and close cloud sync / Codex first.'
printf '%s\n' 'Code, assets and output must be outside all Codex workspaces. Keep every output local.'
if [ "$#" -eq 0 ]; then
  read -r -p 'Absolute isolated asset directory (contains tactical/ and broadcast/): ' DATA_PATH
  read -r -p 'Absolute NEW isolated output directory: ' OUTPUT_PATH
  set -- --data-dir "$DATA_PATH" --output-dir "$OUTPUT_PATH"
fi
PYTHON_BIN="${MTAH_OFFLINE_PYTHON:-$CODE_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo 'STOP: preinstall dependencies and set MTAH_OFFLINE_PYTHON to an existing local Python.' >&2
  exit 2
fi
cd "$CODE_DIR"
exec /usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' \
 /usr/bin/env -i HOME="$HOME" PATH='/usr/bin:/bin' PYTHONNOUSERSITE=1 \
 PYTHONDONTWRITEBYTECODE=1 MTAH_OFFLINE_SANDBOX=macos-deny-network \
 MTAH_FORBIDDEN_WORKSPACE="${MTAH_FORBIDDEN_WORKSPACE:-$HOME/Documents/ChatGPT}" \
 "$PYTHON_BIN" -m solution.offline_assets reference "$@"
