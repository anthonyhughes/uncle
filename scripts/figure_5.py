#!/usr/bin/env python3
"""Regenerate submitted Figure 5, the illustrative Pareto tradeoff.

Plotting-only extraction of scripts/plot_pareto_design_schematic.py. All
coordinates are illustrative, not experimental observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from common import prepare_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(artifact_root: Path, out_dir: Path) -> Path:
    """Draw the original schematic; artifact_root constrains output location."""
    out_dir = prepare_output(artifact_root, out_dir)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    frontier = np.array([[.26, .88], [.49, .77], [.69, .57], [.85, .32], [.92, .13]])
    dominated = np.array([[.20, .72], [.33, .66], [.40, .48], [.57, .37], [.69, .21]])
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    ax.scatter(*dominated.T, color=".73", s=37, label="Other settings", zorder=2)
    ax.plot(*frontier.T, color="#285f8f", marker="o", markersize=6,
            linewidth=2.3, label="Nondominated settings", zorder=3)
    ax.axhline(.50, color=".3", linestyle="--", linewidth=1.1)
    ax.text(.08, .53, "Minimum certification rate", color=".3", fontsize=10)
    ax.scatter([frontier[2, 0]], [frontier[2, 1]], s=118, facecolors="none",
               edgecolors="#b14d22", linewidth=2, zorder=4)
    ax.annotate("Best feasible setting", xy=frontier[2], xytext=(.56, .91),
                fontsize=11, color="#9a421c",
                arrowprops={"arrowstyle": "-", "color": "#9a421c"})
    ax.set(xlim=(.05, 1), ylim=(0, 1), xticks=[], yticks=[],
           xlabel="Test Accuracy", ylabel="Certification Rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.tight_layout(pad=.5)
    output = out_dir / "figure_5.pdf"
    # Preserve the historical PDF metadata literally. Matplotlib 3.7.3's
    # datetime serializer mishandles negative aware timezone offsets.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Bad value for infodict keyword CreationDate.*",
                                category=UserWarning)
        fig.savefig(output, dpi=220, metadata={"CreationDate": "D:20260920183643-04'00'"})
    fig.savefig(out_dir / "figure_5.png", dpi=220)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.artifact_root, args.out_dir))
