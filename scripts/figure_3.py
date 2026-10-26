#!/usr/bin/env python3
"""Re-render submission Figure 3 from its frozen plotting-coordinate ledger.

This isolated renderer freezes the September 21 raised-regularization view:
native 1x, eight regularization factors, deletion count 100, and the original
budget-intersection midpoint statistics. It does not refit models, recompute
frontiers, or import mutable experiment/plotting code from elsewhere in the
repository. The historical creation date is retained as reproduction metadata.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import warnings

from common import prepare_output

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd


INPUT_RELATIVE = Path(
    "diagnostics/ab2-nine-raised-ridge-final-20260921/plots/summary_values.csv"
)
INPUT_SHA256 = "5430a8a8579a47260d186351c8b162b5bbcd88f13fa15cadffd028f6a80246fe"
ORIGINAL_PDF_SHA256 = "bc80496c4c7733aaf5230618edad893252664ab7a51eb4c50216a1b005ed92ec"
FACTORS = (1, 2, 3, 5, 10, 30, 100, 300)
ORDER = (
    "mnist", "imdb", "autotruck", "deeploc_esm", "catdog", "esc50",
    "molhiv_molformer", "molhiv_morgan", "dogfish",
)
LABELS = dict(zip(ORDER, (
    "MNIST", "IMDb", "AutoTruck", "DeepLoc", "CatDog", "ESC50",
    "MolHIV\N{EN DASH}Transformer", "MolHIV\N{EN DASH}Fingerprints", "DogFish",
)))
METHODS = {
    "uncle-augmented": ("UNCLE", "#0072B2", "o", "-"),
    "Guo-v4+v5": ("Guo et al. 2019", "#D55E00", "s", "--"),
}
# Preserve the PDF date string verbatim. Passing a timezone-aware datetime to
# Matplotlib 3.7.3 would trigger its UTC-offset serialization bug (-04 -> -20).
CREATION_DATE = "D:20260921173207-04'00'"


def render(artifact_root: Path, out_dir: Path) -> Path:
    """Write ``figure_3.pdf`` and its PNG companion from verified saved values."""
    out_dir = prepare_output(artifact_root, out_dir)
    source = Path(artifact_root) / INPUT_RELATIVE
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError(f"submission Figure 3 input has changed: {source}")
    summary = pd.read_csv(source, float_precision="round_trip")
    selected = summary[summary.metric.eq("accuracy50") & summary.k.eq(100)]
    if (
        len(selected) != len(ORDER) * len(FACTORS) * len(METHODS)
        or selected.duplicated(["dataset", "factor", "method"]).any()
        or not selected.stage.eq("fresh").all()
    ):
        raise ValueError("the frozen Figure 3 grid is incomplete or not fresh")
    out_dir = Path(out_dir)
    repository = Path(__file__).resolve().parents[1]
    if out_dir.resolve().is_relative_to(repository):
        raise ValueError("generated figures must remain outside the authored repository")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each entrypoint starts from submission-time Matplotlib defaults, including
    # Type-3 PDF fonts. Other figures' rcParams must not leak into this rendering.
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        fig, axes = plt.subplots(3, 3, figsize=(16.8, 12.6))
        for ax, dataset in zip(axes.flat, ORDER):
            panel = selected[selected.dataset.eq(dataset)]
            baseline = panel.groupby("factor").accuracy_reference.first().reindex(FACTORS) * 100
            ax.plot(FACTORS, baseline, color="black", linewidth=2.4, zorder=2)
            plotted = []
            for method, (_, color, marker, style) in METHODS.items():
                series = panel[panel.method.eq(method)].set_index("factor").reindex(FACTORS)
                values = series.plotted_value.to_numpy(float)
                plotted.extend(values[np.isfinite(values)])
                ax.plot(
                    FACTORS, values, color=color, marker=marker, linestyle=style,
                    linewidth=3., markersize=7 if method == "uncle-augmented" else 5.5,
                    markerfacecolor="white", markeredgewidth=1.7, zorder=3,
                )
            ax.set_title(LABELS[dataset], fontsize=23, pad=12)
            ax.set_xscale("log")
            ax.set_xlim(.83, 350)
            ax.set_xticks([1, 3, 10, 30, 100, 300])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.tick_params(labelsize=17)
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(alpha=.46, linewidth=1.)
            ax.set_axisbelow(True)
            finite = np.asarray(plotted + baseline.tolist(), dtype=float)
            finite = finite[np.isfinite(finite)]
            low, high = float(finite.min()), float(finite.max())
            pad = max((high - low) * .09, .25)
            ax.set_ylim(max(0., low - pad), min(100., high + pad))
        fig.supylabel(
            r"Test accuracy at $\geq 50\%$ certification (%)", fontsize=25, x=.008,
        )
        fig.supxlabel(r"Regularization factor", fontsize=25, y=.052)
        handles = [
            Line2D([], [], label=label, color=color, marker=marker, linestyle=style,
                   markerfacecolor="white", linewidth=3.)
            for label, color, marker, style in METHODS.values()
        ]
        handles.append(Line2D([], [], label="No unlearning", color="black", linewidth=2.4))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=21,
                   frameon=False, bbox_to_anchor=(.5, -.002))
        fig.tight_layout(rect=(.027, .06, 1, 1), h_pad=1.6, w_pad=1.25)
        fig.savefig(out_dir / "figure_3.png", dpi=185, bbox_inches="tight")
        pdf = out_dir / "figure_3.pdf"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Bad value for infodict keyword CreationDate\\.",
                category=UserWarning,
            )
            fig.savefig(pdf, dpi=185, bbox_inches="tight",
                        metadata={"CreationDate": CREATION_DATE})
        plt.close(fig)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.artifact_root, args.out_dir))


if __name__ == "__main__":
    main()
