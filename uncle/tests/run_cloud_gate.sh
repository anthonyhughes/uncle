#!/usr/bin/env bash
# Run only on a cloud worker. The optional sibling historical checkout supplies
# migration-test oracles; the standalone package itself never imports them.
set -euo pipefail
SOURCE="${1:?supply the checkout parent of uncle/}"
RESULTS="${2:?supply an external cloud results directory}"
PY=/mnt/disks/unerring/envs/guo-py313/bin/python
mkdir -p "$RESULTS"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$RESULTS/matplotlib"
export PYTHONPATH="$SOURCE"
cd "$SOURCE/uncle"
"$PY" -m pip freeze > "$RESULTS/environment.txt"
"$PY" -m pytest tests -q -ra -p no:cacheprovider --junitxml="$RESULTS/tests.xml" 2>&1 | tee "$RESULTS/tests.log"
PYTHONPATH="$SOURCE" "$PY" -m pytest tests -q -ra -m 'not migration and not real_data' -p no:cacheprovider --junitxml="$RESULTS/standalone.xml" 2>&1 | tee "$RESULTS/standalone.log"
