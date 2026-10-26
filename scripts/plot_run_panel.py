#!/usr/bin/env python3
"""Draw one panel of test CE increase against samples removed.

Colors and markers follow figure_1.METHODS. The input is the fresh-run ledger
from write_run_ledger.py. The PDF sits outside the byte-for-byte check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.write_run_ledger import ALGORITHM_TO_METHOD, ATTAINED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


SERIES = {
    "uncle-augmented": ("UNCLE", "#0072B2", "o", "-", 7.0),
    "Guo-v4+v5": ("Guo et al. 2019", "#D55E00", "s", "--", 5.3),
}
REQUIRED_COLUMNS = ("dataset", "method", "k", "ce_increase", "success_rate", "status")
LABELS = {"mnist": "MNIST"}
X_TICKS = (1, 3, 10)
X_MIN_FACTOR = 0.85
X_MAX_FACTOR = 1.17
Y_PAD_FRACTION = 0.055
LINE_WIDTH = 2.9
ZERO_LINE_WIDTH = 1.4
PANEL_SIZE = (6.4, 5.0)
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 12
LEGEND_SIZE = 12
X_LABEL = "Samples removed"
Y_LABEL = "Test CE increase"


def load_ledger(ledger_path: Path) -> pd.DataFrame:
    """Return one attained CE increase per method and removal size."""
    table = pd.read_csv(ledger_path, float_precision="round_trip")
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing_columns:
        raise ValueError(f"ledger is missing columns: {missing_columns}")
    if set(SERIES) != set(ALGORITHM_TO_METHOD.values()):
        raise ValueError("panel series and ledger method names differ")
    present = set(table.method)
    missing = sorted(set(SERIES) - present)
    extra = sorted(present - set(SERIES))
    if missing or extra:
        raise ValueError(f"ledger methods must be the fresh-run pair; missing {missing}, extra {extra}")
    if table.dataset.nunique() != 1:
        raise ValueError("ledger must contain one dataset")
    if not table.status.eq(ATTAINED).all():
        raise ValueError("ledger rows must have status attained")
    removal_size = pd.to_numeric(table.k, errors="coerce")
    increase = pd.to_numeric(table.ce_increase, errors="coerce")
    if (
        removal_size.isna().any()
        or increase.isna().any()
        or not np.isfinite(removal_size.to_numpy(dtype=float)).all()
        or not np.isfinite(increase.to_numpy(dtype=float)).all()
        or not np.equal(removal_size, np.rint(removal_size)).all()
        or (removal_size < 1).any()
    ):
        raise ValueError("k must be a positive integer and ce_increase must be finite")
    cleaned = table.assign(k=removal_size.astype(int), ce_increase=increase.astype(float))
    if cleaned.duplicated(["method", "k"]).any():
        raise ValueError("ledger repeats a method at one removal size")
    sizes = [set(group.k) for _, group in cleaned.groupby("method", sort=False)]
    if sizes[0] != sizes[1]:
        raise ValueError("both methods must use the same removal sizes")
    if max(sizes[0]) <= 1:
        raise ValueError("cannot form a logarithmic axis from a single removal size of 1")
    return cleaned


def plot_run_panel(ledger_path: Path, out_stem: Path) -> list[Path]:
    """Write ``out_stem`` with ``.png`` and ``.pdf`` suffixes."""
    comparison = load_ledger(ledger_path)
    stem = Path(out_stem)
    paths = [stem.with_suffix(suffix) for suffix in (".png", ".pdf")]
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"refusing to replace {existing}")
    stem.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(matplotlib.rcParamsDefault):
        figure, axis = plt.subplots(figsize=PANEL_SIZE)
        try:
            _draw(axis, figure, comparison)
            for path in paths:
                figure.savefig(path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
    return paths


def _draw(axis, figure, comparison: pd.DataFrame) -> None:
    axis.axhline(0.0, color="black", linestyle="-", linewidth=ZERO_LINE_WIDTH, zorder=0)
    for method, (label, color, marker, style, marker_size) in SERIES.items():
        group = comparison.loc[comparison.method.eq(method)].sort_values("k")
        axis.plot(
            group.k.to_numpy(dtype=int),
            group.ce_increase.to_numpy(dtype=float),
            color=color,
            marker=marker,
            markerfacecolor="white",
            linestyle=style,
            linewidth=LINE_WIDTH,
            markersize=marker_size,
            zorder=3,
        )
    largest = int(comparison.k.max())
    x_min, x_max = X_MIN_FACTOR, X_MAX_FACTOR * largest
    axis.set_xscale("log")
    axis.set_xlim(x_min, x_max)
    ticks = [tick for tick in X_TICKS if x_min <= tick <= x_max]
    axis.set_xticks(ticks, labels=[f"{tick:g}" for tick in ticks])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    values = comparison.ce_increase.to_numpy(dtype=float)
    low = min(0.0, float(values.min()))
    high = max(0.0, float(values.max()))
    padding = max((high - low) * Y_PAD_FRACTION, 1e-6)
    axis.set_ylim(low - padding, high + padding)
    dataset = str(comparison.dataset.iloc[0])
    axis.set_title(LABELS.get(dataset, dataset), fontsize=TITLE_SIZE, pad=8)
    axis.set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    axis.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE)
    axis.tick_params(labelsize=TICK_SIZE)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.43, linewidth=0.9)
    handles = [
        Line2D(
            [], [], label=label, color=color, marker=marker, markerfacecolor="white",
            linestyle=style, linewidth=LINE_WIDTH, markersize=marker_size,
        )
        for label, color, marker, style, marker_size in SERIES.values()
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Path stem for the PNG and PDF")
    args = parser.parse_args()
    for path in plot_run_panel(args.ledger, args.out):
        print(path)


if __name__ == "__main__":
    main()
