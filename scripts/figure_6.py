#!/usr/bin/env python3
"""Reproduce submitted Figure 6 from the frozen plotted Guo parameter ledgers.

The plotted coordinates and frontier membership are read from the September 20
outputs, not recomputed or reselected. No model fitting or cloud access occurs.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import warnings

from common import prepare_output

import numpy as np
import pandas as pd


ORDER = (
    "mnist", "imdb", "autotruck", "deeploc_esm", "catdog", "esc50",
    "molhiv_molformer", "molhiv_morgan", "dogfish",
)
LABELS = dict(zip(ORDER, (
    "MNIST", "IMDb", "AutoTruck", "DeepLoc", "CatDog", "ESC50",
    "MolHIV–Transformer", "MolHIV–Fingerprints", "DogFish",
)))
INPUT_DIRECTORY = Path("diagnostics/experiment-design-appendix-20260920/figures")
INPUT_SHA256 = {
    "guo_sampled_parameters.csv":
        "7bae59d3359ca3370b3426aba5b6c0feac43144d857b9bdc04fd49221921622a",
    "guo_selected_pareto_parameters.csv":
        "3ee6e7e0d54ee86ccd05fb4d96a27c42d1a6f3ece35d3d0b8aaff61c158e3845",
}
# Preserve the literal original PDF metadata, not the reproduction's wall clock.
# Matplotlib 3.7.3 misserializes negative datetime offsets (e.g. -04 as -20);
# its string serialization preserves the correct PDF timestamp verbatim.
CREATION_DATE = "D:20260920180733-04'00'"


def render(artifact_root: Path, out_dir: Path) -> Path:
    """Render Figure 6 with its original styles, point order and coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullLocator

    out_dir = prepare_output(artifact_root, out_dir)
    directory = Path(artifact_root) / INPUT_DIRECTORY
    for filename, expected in INPUT_SHA256.items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen Figure 6 input has changed: {directory / filename}")
    grid = pd.read_csv(directory / "guo_sampled_parameters.csv", float_precision="round_trip")
    selected = pd.read_csv(directory / "guo_selected_pareto_parameters.csv",
                           float_precision="round_trip")
    if set(grid.dataset) != set(ORDER) or set(selected.dataset) != set(ORDER):
        raise ValueError("Figure 6 requires all nine frozen datasets")
    if not grid.k.eq(10).all() or not selected.k.eq(10).all():
        raise ValueError("The submitted Figure 6 uses k=10")
    out_dir = Path(out_dir)
    if out_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Generated figures must remain outside the authored repository")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9.2, "axes.labelsize": 9, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": .6, "savefig.dpi": 300})
    fig, axes = plt.subplots(3, 3, figsize=(6.55, 4.05))
    for ax, dataset in zip(axes.flat, ORDER):
        local = grid.loc[grid.dataset == dataset]
        for stage, color, marker, size in (("coarse", "#849CB0", "o", 5.0),
                                          ("refinement", "#D55E00", "+", 11.0)):
            group = local.loc[local.stage == stage]
            ax.scatter(group.lam, group.sigma, color=color, marker=marker, s=size,
                       linewidths=.65 if stage == "refinement" else 0, alpha=.85, zorder=2)
        frontier = selected.loc[selected.dataset == dataset]
        ax.scatter(frontier.lam, frontier.sigma, facecolors="none", edgecolors="#004C3F",
                   s=20, linewidths=.8, zorder=3)
        ax.set_title(LABELS[dataset], pad=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.23, linewidth=.4)
        ax.tick_params(width=.6, length=2.4, pad=2)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticks([.001, .1, 10.])
        ax.set_ylim(.0005, 650)
        lo, hi = np.log10(local.lam.min()), np.log10(local.lam.max())
        ax.set_xlim(10 ** (lo-.08*(hi-lo)), 10 ** (hi+.04*(hi-lo)))
    fig.supxlabel(r"Regularization coefficient $\lambda$", y=.075, fontsize=9)
    fig.supylabel(r"Objective-noise scale $\sigma$", x=.009, fontsize=9)
    handles = [Line2D([], [], marker="o", linestyle="none", color="#849CB0", markersize=3, label="Coarse grid"),
               Line2D([], [], marker="+", linestyle="none", color="#D55E00", markersize=4, label="Refinement"),
               Line2D([], [], marker="o", linestyle="none", color="#004C3F", markerfacecolor="none", markersize=4,
                      label="Selected Pareto settings")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.53, .002),
               columnspacing=1.5, handletextpad=.35)
    fig.subplots_adjust(left=.102, right=.988, top=.945, bottom=.18, hspace=.60, wspace=.31)
    fig.savefig(out_dir / "figure_6.png", bbox_inches="tight", pad_inches=.025)
    output = out_dir / "figure_6.pdf"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Bad value for infodict keyword CreationDate",
                                category=UserWarning)
        fig.savefig(output, bbox_inches="tight", pad_inches=.025,
                    metadata={"CreationDate": CREATION_DATE})
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.artifact_root, args.out_dir))


if __name__ == "__main__":
    main()
