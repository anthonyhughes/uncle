#!/usr/bin/env python3
"""Write a fresh-run CE ledger from a Guo versus UNCLE comparison CSV.

Method names match figure_1.METHODS. This ledger has no 50% retraining
threshold and no CE allowance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ALGORITHM_TO_METHOD = {
    "ab2": "uncle-augmented",
    "guo": "Guo-v4+v5",
}
REQUIRED_COLUMNS = ("algorithm", "k", "ce_increase", "success_rate")
LEDGER_COLUMNS = ("dataset", "method", "k", "ce_increase", "success_rate", "status")
ATTAINED = "attained"
MISSING = "missing"


def comparison_to_ledger(table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Map one comparison table onto ledger rows."""
    if not dataset:
        raise ValueError("dataset name is empty")
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing_columns:
        raise ValueError(f"comparison CSV is missing columns: {missing_columns}")
    present = set(table.algorithm)
    missing = sorted(set(ALGORITHM_TO_METHOD) - present)
    extra = sorted(present - set(ALGORITHM_TO_METHOD))
    if missing or extra:
        raise ValueError(
            f"comparison CSV algorithms must be ab2 and guo; missing {missing}, extra {extra}"
        )
    removal_size = pd.to_numeric(table.k, errors="coerce")
    increase = pd.to_numeric(table.ce_increase, errors="coerce")
    success = pd.to_numeric(table.success_rate, errors="coerce")
    if (
        removal_size.isna().any()
        or success.isna().any()
        or not np.isfinite(removal_size.to_numpy(dtype=float)).all()
        or not np.isfinite(success.to_numpy(dtype=float)).all()
        or not np.equal(removal_size, np.rint(removal_size)).all()
        or (removal_size < 1).any()
        or not success.between(0, 1).all()
    ):
        raise ValueError("k must be a positive integer and success_rate must be between 0 and 1")
    cleaned = table.assign(
        k=removal_size.astype(int),
        ce_increase=increase.astype(float),
        success_rate=success.astype(float),
    )
    if cleaned.duplicated(["algorithm", "k"]).any():
        raise ValueError("comparison CSV repeats an algorithm at one removal size")
    sizes = [set(group.k) for _, group in cleaned.groupby("algorithm", sort=False)]
    if sizes[0] != sizes[1]:
        raise ValueError("ab2 and guo must use the same removal sizes")
    finite = np.isfinite(cleaned.ce_increase.to_numpy(dtype=float))
    ledger = pd.DataFrame({
        "dataset": dataset,
        "method": cleaned.algorithm.map(ALGORITHM_TO_METHOD),
        "k": cleaned.k,
        "ce_increase": cleaned.ce_increase.where(finite),
        "success_rate": cleaned.success_rate,
        "status": np.where(finite, ATTAINED, MISSING),
    })
    method_order = list(ALGORITHM_TO_METHOD.values())
    ledger["method"] = pd.Categorical(ledger["method"], method_order, ordered=True)
    return (
        ledger.sort_values(["method", "k"])
        .reset_index(drop=True)
        .astype({"method": str})[list(LEDGER_COLUMNS)]
    )


def write_run_ledger(csv_path: Path, out_path: Path, dataset: str = "mnist") -> Path:
    """Write the ledger. An existing destination is left unchanged."""
    destination = Path(out_path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace {destination}")
    ledger = comparison_to_ledger(
        pd.read_csv(csv_path, float_precision="round_trip"), dataset
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="") as stream:
        ledger.to_csv(stream, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="mnist")
    args = parser.parse_args()
    print(write_run_ledger(args.csv, args.out, dataset=args.dataset))


if __name__ == "__main__":
    main()
