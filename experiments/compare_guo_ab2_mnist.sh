#!/usr/bin/env bash
# Compare Guo and A+B2 on full MNIST 3-vs-8 for removal sizes 1 through 10.
#
# One shared operating point: the MNIST training ridge from the real-data
# parity fixture, Guo sigma 1, and A+B2 geometry multiplier 1 with radius
# factor 1. The ten removal requests estimate the no-retraining rate. They
# are not the removal sizes. Existing data, config, points, run shard, CSV,
# fresh-run ledger, summary, and plots are left in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

EXP="$ROOT/experiments/guo_ab2_mnist"
NPZ="$EXP/mnist_3v8.npz"
CONFIG="$EXP/config.json"
POINTS="$EXP/points.csv"
RUN="$EXP/run"
CSV="$ROOT/experiments/guo_ab2_mnist.csv"
LEDGER="$EXP/ce_increase_ledger.csv"
SUMMARY="$EXP/summary"
PLOTS="$EXP/plots"
PY=(uv run --with pandas --with scikit-learn --with matplotlib python)

refuse() {
  echo "refusing to replace $1" >&2
  exit 1
}

if [[ -e "$CSV" ]]; then
  refuse "$CSV"
fi
if [[ -e "$LEDGER" ]]; then
  refuse "$LEDGER"
fi
if [[ -e "$SUMMARY" ]]; then
  refuse "$SUMMARY"
fi
if [[ -e "$PLOTS" ]]; then
  refuse "$PLOTS"
fi
mkdir -p "$EXP"

if [[ -e "$NPZ" ]]; then
  echo "using existing $NPZ"
else
  "${PY[@]}" - "$NPZ" <<'PY'
import sys
from pathlib import Path

import numpy as np

from data.mnist import load_mnist

destination = Path(sys.argv[1])
if destination.exists():
    raise SystemExit(f"refusing to replace {destination}")
split = load_mnist()
test = split.test
if test is None:
    raise SystemExit("MNIST split has no test set")
with destination.open("xb") as stream:
    np.savez(
        stream,
        X_train=np.asarray(split.train.features),
        y_train=np.asarray(split.train.labels),
        X_test=np.asarray(test.features),
        y_test=np.asarray(test.labels),
    )
print(
    f"wrote {destination} train={tuple(split.train.features.shape)} "
    f"test={tuple(test.features.shape)}"
)
PY
fi

if [[ -e "$CONFIG" ]]; then
  echo "using existing $CONFIG"
else
  "${PY[@]}" - "$NPZ" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from uncle.logistic import fit, metrics

npz_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
if config_path.exists():
    raise SystemExit(f"refusing to replace {config_path}")
lam = 0.0013456462345536545
with np.load(npz_path, allow_pickle=False) as saved:
    train_x = np.asarray(saved["X_train"], dtype=np.float64)
    train_y = np.asarray(saved["y_train"], dtype=np.float64)
    test_x = np.asarray(saved["X_test"], dtype=np.float64)
    test_y = np.asarray(saved["y_test"], dtype=np.float64)
theta = fit(train_x, train_y, lam, polish=True)
test_ce, test_accuracy = metrics(theta, test_x, test_y)
config = {
    "dataset": "mnist",
    "rng_dataset": "mnist",
    "data_npz": str(npz_path.resolve()),
    "privacy": {"epsilon": 1.0, "delta": 1e-4},
    "master_seed": 2026101301,
    "num_seeds": 1,
    "removal_requests": 10,
    "ks": list(range(1, 11)),
    "reference_metrics": {"test_ce": test_ce, "test_accuracy": test_accuracy},
    "uncle_lambdas": [lam],
    "guo_lambdas": [lam],
    "geometry_multipliers": [1.0],
    "safety_factors": [1.0],
    "sigmas": [1.0],
    "s_inf_values": [10.0, 1.0, 0.1, 0.01, 0.001],
    "uncle_variant": "theta0-A-B2-augmented-v1",
}
payload = json.dumps(config, indent=2) + "\n"
loaded = json.loads(payload)
if (loaded["privacy"]["epsilon"], loaded["privacy"]["delta"]) != (1.0, 1e-4):
    raise SystemExit("privacy budget did not round-trip through JSON")
with config_path.open("x") as stream:
    stream.write(payload)
print(
    f"wrote {config_path} reference_ce={test_ce} "
    f"reference_accuracy={test_accuracy}"
)
PY
fi

if [[ -e "$POINTS" ]]; then
  echo "using existing $POINTS"
else
  "${PY[@]}" -m uncle plan "$CONFIG" --methods uncle guo --out "$POINTS"
fi

"${PY[@]}" - "$POINTS" <<'PY'
import sys
from pathlib import Path

import pandas as pd

points = pd.read_csv(Path(sys.argv[1]), float_precision="round_trip")
ks = set(range(1, 11))
if (
    len(points) != 20
    or set(points.method) != {"uncle", "guo"}
    or set(points.k.astype(int)) != ks
    or any(set(group.k.astype(int)) != ks for _, group in points.groupby("method"))
):
    raise SystemExit("point table is not one Guo point and one A+B2 point for k=1..10")
print(f"point table has {len(points)} rows")
PY

if [[ -e "$RUN" && ! -d "$RUN" ]]; then
  refuse "$RUN"
fi
if [[ -d "$RUN" ]]; then
  if [[ ! -f "$RUN/complete.json" ]]; then
    refuse "$RUN"
  fi
  status="$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN/complete.json")"
  if [[ "$status" != "pass" ]]; then
    refuse "$RUN"
  fi
  echo "using existing $RUN"
else
  "${PY[@]}" -m uncle run "$CONFIG" "$POINTS" --out "$RUN"
fi

"${PY[@]}" - "$RUN" "$CSV" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
if csv_path.exists():
    raise SystemExit(f"refusing to replace {csv_path}")
manifest = json.loads((run_dir / "manifest.json").read_text())
complete = json.loads((run_dir / "complete.json").read_text())
config = manifest["config"]
if complete["status"] != "pass":
    raise SystemExit(f"run did not pass: {run_dir}")
if config["uncle_variant"] != "theta0-A-B2-augmented-v1":
    raise SystemExit("run is not the A+B2 variant")
rows = pd.read_csv(run_dir / "runs.csv", float_precision="round_trip")
rows["algorithm"] = rows["method"].map({"uncle": "ab2", "guo": "guo"})
if rows["algorithm"].isna().any() or set(rows["algorithm"]) != {"ab2", "guo"}:
    raise SystemExit("unexpected method in runs.csv")
rows["k"] = rows["k"].astype(int)
rows["ce_increase"] = rows["test_ce"] - rows["ce_reference"]
rows["accuracy_drop"] = rows["accuracy_reference"] - rows["test_accuracy"]
columns = [
    "algorithm",
    "k",
    "lam",
    "sigma",
    "geometry_multiplier",
    "radius_factor",
    "test_ce",
    "test_accuracy",
    "success_rate",
    "ce_reference",
    "accuracy_reference",
    "ce_increase",
    "accuracy_drop",
]
table = rows[columns].sort_values(["algorithm", "k"]).reset_index(drop=True)
ks = set(range(1, 11))
metrics = table[["test_ce", "test_accuracy", "success_rate", "ce_increase", "accuracy_drop"]]
if (
    len(table) != 20
    or any(set(group.k) != ks for _, group in table.groupby("algorithm"))
    or not np.isfinite(metrics.to_numpy(dtype=float)).all()
    or not table.success_rate.between(0, 1).all()
):
    raise SystemExit("runs.csv is not a complete Guo vs A+B2 comparison for k=1..10")
with csv_path.open("x", newline="") as stream:
    table.to_csv(stream, index=False)
print(f"wrote {csv_path} rows={len(table)}")
PY

"${PY[@]}" scripts/write_run_ledger.py \
  --csv "$CSV" \
  --dataset mnist \
  --out "$LEDGER"
"${PY[@]}" -m uncle summarize "$RUN" --out "$SUMMARY"
"${PY[@]}" -m uncle plot "$SUMMARY/summary.csv" --dataset mnist --out "$PLOTS"
"${PY[@]}" scripts/plot_run_panel.py \
  --ledger "$LEDGER" \
  --out "$PLOTS/mnist_ce_increase"

echo "$CSV"
echo "$LEDGER"
echo "$PLOTS"
