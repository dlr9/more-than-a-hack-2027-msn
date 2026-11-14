#!/bin/bash
# R7 named entry point; the reviewed R6 wrapper enforces environment and network isolation.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec /bin/bash "$SCRIPT_DIR/run_offline_reference.command" "$@"
