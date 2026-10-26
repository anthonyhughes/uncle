#!/usr/bin/env python3
"""Reproduce submission Figure 1 from frozen, already-computed summary points.

This is a plotting-only snapshot of the September 24 mean-allowance renderer.
It does not import mutable repository helpers, recompute frontier cuts, fit a
model, or substitute results from the later finer-k campaign. Generated files
must remain outside the authored source workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import warnings

from common import prepare_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


PACKAGE = Path("diagnostics/uncle-nine-mean-retention-refresh-20260924")
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
CREATION_DATE = "D:20260924111422-04'00'"


def _limits(points: pd.DataFrame, budgets: dict[str, float]) -> dict:
    """Original v1 clipping: first sampled joint crossing of 3*allowance."""
    result = {}
    for dataset in ORDER:
        budget = float(budgets[dataset])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError(f"{dataset}: invalid mean CE allowance")
        local = points.loc[points.dataset.eq(dataset)]
        paired = local.pivot(index="k", columns="method", values="midpoint").sort_index()
        if set(paired.columns) != set(METHODS):
            raise ValueError(f"{dataset}: incomplete comparison")
        attained = local.pivot(index="k", columns="method", values="status").eq("attained").all(axis=1)
        finite = paired.apply(lambda col: col.map(math.isfinite)).all(axis=1)
        crossing = paired.index[attained & finite & paired.gt(3. * budget).all(axis=1)]
        xmax = int(crossing[0] if len(crossing) else paired.index.max())
        if xmax <= 1:
            raise ValueError("Cannot form a nondegenerate logarithmic axis")
        result[dataset] = (1., float(xmax), 0., 3. * budget)
    return result


def render(artifact_root: Path, out_dir: Path) -> Path:
    """Write figure_1.pdf and figure_1.png, returning the PDF path.

    ``artifact_root`` is the external ``project_files/unerring`` directory.
    The original completed package supplies the summary CSV and provenance;
    every loaded scientific input is checked against its recorded SHA-256.
    """
    out_dir = prepare_output(artifact_root, out_dir)
    source = Path(artifact_root).resolve() / PACKAGE
    out = Path(out_dir).resolve()
    workspace = Path(__file__).resolve().parents[1]
    if out.is_relative_to(workspace):
        raise ValueError("Generated figures must be outside the source workspace")
    manifest_path = source / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != (
        "cd40fdff124d658c58a8aca64c656dba88c1057f9398e94c8eaf8a5e6784b387"
    ):
        raise ValueError("The frozen Figure 1 manifest has changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["ce_allowance_statistic"] != "mean":
        raise ValueError("Expected the completed mean-allowance submission package")
    for name in ("plots/ce_increase_midpoints.csv", "tables/removal50_thresholds.csv"):
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != manifest["output_sha256"][name]:
            raise ValueError(f"Frozen input SHA-256 mismatch: {name}")
    points = pd.read_csv(source / "plots/ce_increase_midpoints.csv",
                         float_precision="round_trip")
    points = points.loc[points.threshold_pct.eq(50)].copy()
    if set(points.dataset) != set(ORDER):
        raise ValueError("Expected all nine submission datasets")
    thresholds = pd.read_csv(source / "tables/removal50_thresholds.csv",
                             float_precision="round_trip")
    budgets = manifest["ce_allowances"]
    np.testing.assert_allclose(
        thresholds.ce_allowance, thresholds.dataset.map(budgets), rtol=0, atol=0)
    panel_limits = _limits(points, budgets)
    out.mkdir(parents=True, exist_ok=True)

    # Defaults are frozen by the required Matplotlib version rather than by any
    # mutable repository/global style. This also isolates repeated runner calls.
    with plt.rc_context(matplotlib.rcParamsDefault):
        fig, axes = plt.subplots(3, 3, figsize=(15.4, 12.4))
        for ax, dataset in zip(axes.flat, ORDER):
            local = points.loc[points.dataset.eq(dataset)]
            ks = manifest["panel_ks"][dataset]
            ax.axhline(0., color="black", linestyle="-", linewidth=1.4, zorder=0)
            reference = float(budgets[dataset])
            ax.axhline(reference, color="0.35", linestyle="--", linewidth=1.9,
                       zorder=2, label="_nolegend_")
            for index, (method, (_, color, marker, style)) in enumerate(METHODS.items()):
                group = local.loc[local.method.eq(method)].set_index("k").reindex(ks)
                ax.plot(ks, group.midpoint.to_numpy(float), color=color, marker=marker,
                        markerfacecolor="white", linestyle=style, linewidth=2.9,
                        markersize=7 if index == 0 else 5.3, zorder=3)
                missing = group.status.ne("attained").to_numpy()
                ax.scatter(np.asarray(ks)[missing], np.full(missing.sum(), .97-.06*index),
                           transform=ax.get_xaxis_transform(), marker="x", color=color, s=40)
            ax.set_title(LABELS[dataset], fontsize=23, pad=12)
            ax.set_xscale("log")
            if max(ks) <= 100:
                ax.set_xticks([1, 3, 10, 30, 100])
                ax.xaxis.set_major_formatter(ScalarFormatter())
            else:
                ax.set_xticks([1, 10, 100, 1000, 3000],
                              labels=["1", "10", "100", "1k", "3k"])
            ax.set_xlim(.85, 1.17*max(ks))
            values = local.midpoint.to_numpy(float)
            finite = np.append(values[np.isfinite(values)], reference)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            lo, hi = finite.min(), finite.max()
            padding = max((hi-lo)*.055, 1e-6)
            ax.set_ylim(lo-padding, hi+padding)
            xmin, xmax, ymin, ymax = panel_limits[dataset]
            ticks = [1, 3, 10, 30, 100] if xmax <= 100 else [1, 10, 100, 1000]
            ticks = [value for value in ticks if xmin <= value <= xmax]
            labels = [f"{value/1000:g}k" if value >= 1000 else f"{value:g}" for value in ticks]
            ax.set_xticks(ticks, labels=labels)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=16)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(alpha=.43, linewidth=.9)
        fig.supylabel(r"Test CE increase at $\leq 50\%$ retrain", fontsize=23, x=.004)
        fig.supxlabel("Samples removed", fontsize=24, y=.048)
        handles = [
            Line2D([], [], label=label, color=color, marker=marker, markerfacecolor="white",
                   linestyle=style, linewidth=2.9)
            for label, color, marker, style in METHODS.values()
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=21,
                   bbox_to_anchor=(.5, -.002))
        fig.tight_layout(rect=(.025, .046, 1, 1), h_pad=1.7, w_pad=1.4)
        fig.savefig(out / "figure_1.png", dpi=180, bbox_inches="tight")
        pdf = out / "figure_1.pdf"
        # Matplotlib 3.7.3 serializes explicit negative-offset datetime objects
        # incorrectly (-04 becomes -20). Its PDF writer accepts this original
        # literal date string, but warns that it is not a datetime. Suppress only
        # that known metadata warning; preserve the original PDF bytes exactly.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Bad value for infodict keyword CreationDate\..*",
                category=UserWarning,
            )
            fig.savefig(pdf, dpi=180, bbox_inches="tight",
                        metadata={"CreationDate": CREATION_DATE})
        plt.close(fig)
    return pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(render(arguments.artifact_root, arguments.out_dir))
