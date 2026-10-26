#!/usr/bin/env python3
"""Regenerate submitted Figure 7 from its frozen 30-draw calibration means.

The logarithmic plot uses the arithmetic mean, not the descriptive percentile
columns also present in the historical input ledger. No fitting or reference
selection occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from common import prepare_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
import numpy as np
import pandas as pd

DATASETS = {
    "mnist": "MNIST", "imdb": "IMDb", "autotruck": "AutoTruck",
    "deeploc_esm": "DeepLoc", "catdog": "CatDog", "esc50": "ESC50",
    "molhiv_molformer": "MolHIV\u2013Transformer",
    "molhiv_morgan": "MolHIV\u2013Fingerprints", "dogfish": "DogFish",
}
INPUT = Path("diagnostics/retention-nine-dense30-20260922a/plot_variants/fraction_percentiles.csv")


def render(artifact_root: Path, out_dir: Path) -> Path:
    """Render the original log-y CE-increase plot from its plotted point ledger."""
    out_dir = prepare_output(artifact_root, out_dir)
    source = artifact_root / INPUT
    if hashlib.sha256(source.read_bytes()).hexdigest() != (
        "26dc0dff78024d63cfd98f55fafe8ef3d1ca1a4837d8e622fd7c3fb48f2fe8c8"
    ):
        raise ValueError("The frozen Figure 7 mean-coordinate ledger has changed")
    summary = pd.read_csv(source, float_precision="round_trip")
    if (set(summary.dataset) != set(DATASETS)
            or summary.duplicated(["dataset", "removed_fraction"]).any()
            or not summary.n_draws.eq(30).all()
            or not summary.groupby("dataset").size().eq(101).all()):
        raise ValueError("Expected nine datasets with 101 fractions and 30 draws")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    fig, axes = plt.subplots(3, 3, figsize=(17.4, 12.8))
    for ax, (dataset, label) in zip(axes.flat, DATASETS.items()):
        frame = summary[summary.dataset.eq(dataset)].sort_values("removed_fraction")
        values = frame.ce_increase_mean.to_numpy().copy()
        values[values <= 0] = np.nan  # The original log plot omits nonpositive means.
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.plot(100 * frame.removed_fraction, values, color="#0072B2", linewidth=2.8,
                marker="o", markersize=4, markerfacecolor="white")
        ax.set_title(label, fontsize=22, pad=12)
        ax.set_xlim(-1, 101)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(which="major", alpha=.45, linewidth=1)
        ax.grid(which="minor", axis="y", alpha=.16, linewidth=.6)
    fig.supylabel("Increase in test cross-entropy", fontsize=24, x=.008)
    fig.supxlabel("Training samples removed (%)", fontsize=24, y=.005)
    fig.tight_layout(rect=(.025, .03, 1, 1), h_pad=1.6, w_pad=1.4)
    # The original saved PNG precedes the PDF; retain that order.
    fig.savefig(out_dir / "figure_7.png", dpi=180, bbox_inches="tight")
    output = out_dir / "figure_7.pdf"
    fig.savefig(output, dpi=180, bbox_inches="tight", metadata={"CreationDate": datetime(
        2026, 9, 22, 15, 44, 40, tzinfo=timezone.utc)})
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.artifact_root, args.out_dir))
