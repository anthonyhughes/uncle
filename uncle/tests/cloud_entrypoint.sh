#!/usr/bin/env bash
# One isolated validation worker, never a production experiment launcher.
set -euo pipefail
SOURCE="${1:?supply the checkout parent of uncle/}"
RESULTS="${2:?supply an external cloud results directory}"
exec bash "$SOURCE/uncle/tests/run_cloud_gate.sh" "$SOURCE" "$RESULTS"
