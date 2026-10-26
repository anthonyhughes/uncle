#!/usr/bin/env python3
"""Reproduce submitted Figure 4 from the frozen September 22 plotted ledgers.

This is an isolated copy of the original plotting recipe. It neither refits
models nor imports the mutable experiment implementation. ``artifact_root`` is
the external ``project_files/unerring`` directory, not its diagnostics child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
INPUT_DIRECTORY = Path(
    "diagnostics/uncle-nine-test-selected-refresh-20260922/reference_search"
)
INPUT_SHA256 = {
    "baseline_sampled_regularizations.csv":
        "10719277168dd3344c15ab282427fd3a33bfa274fdd6490e2d142dbc9c274084",
    "baseline_failed_regularizations.csv":
        "6242cee27c5f140cdbf0bf5050de5904b731b91fcb6c5b1cd23219990ae2ff75",
    "figure_manifest.json":
        "5727b31f9f25e5f833ddf00c09fd4aa9867538ca238b7d397815a30e5292acdc",
}
# Preserve the literal original PDF metadata, not the reproduction's wall clock.
# Matplotlib 3.7.3 misserializes negative datetime offsets (e.g. -04 as -20);
# its string serialization preserves the correct PDF timestamp verbatim.
CREATION_DATE = "D:20260922171546-04'00'"


def _panels(artifact_root: Path) -> list[dict]:
    directory = artifact_root / INPUT_DIRECTORY
    for filename, expected in INPUT_SHA256.items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen Figure 4 input has changed: {directory / filename}")
    manifest = json.loads((directory / "figure_manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or tuple(manifest["panel_order"]) != ORDER:
        raise ValueError("Figure 4 requires the complete September 22 nine-panel receipt")
    metrics = pd.read_csv(directory / "baseline_sampled_regularizations.csv",
                          float_precision="round_trip")
    failures = pd.read_csv(directory / "baseline_failed_regularizations.csv",
                           float_precision="round_trip")
    selections = {panel["dataset"]: panel["selected"]
                  for panel in manifest["source_panels"]}
    return [
        dict(
            dataset=dataset,
            metrics=metrics.loc[metrics.dataset == dataset].copy(),
            failures=failures.loc[failures.dataset == dataset].to_dict(orient="records"),
            selected=selections[dataset],
        )
        for dataset in ORDER
    ]


def render(artifact_root: Path, out_dir: Path) -> Path:
    """Render Figure 4 from saved numeric observations and return its PDF path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator, ScalarFormatter

    out_dir = prepare_output(artifact_root, out_dir)
    panels = _panels(Path(artifact_root))
    out_dir = Path(out_dir)
    if out_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Generated figures must remain outside the authored repository")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": .8, "savefig.dpi": 300})
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 7.6))
    for ax, panel in zip(axes.flat, panels):
        data = panel["metrics"]
        ordered = data.sort_values("lam")
        ax.plot(ordered.lam, ordered.test_ce, color=".75", linewidth=.8, zorder=1)
        for stage, color, marker, size in (("coarse", "#0072B2", "o", 20),
                                           ("fine", "#E69F00", ".", 9)):
            group = data.loc[data.stage == stage]
            ax.scatter(group.lam, group.test_ce, color=color, marker=marker,
                       s=size, linewidths=0, zorder=3 if stage == "coarse" else 2)
        failures = panel["failures"]
        if failures:
            ax.scatter([row["lam"] for row in failures], [.035] * len(failures),
                       transform=ax.get_xaxis_transform(), color="#C00000", marker="x",
                       s=25, linewidths=1.2, zorder=5)
        selected = panel["selected"]
        ax.scatter([selected["lam"]], [selected["test_ce"]], marker="*", s=90,
                   facecolor="#004C3F", edgecolor="white", linewidth=.5, zorder=6)
        ax.set_title(LABELS[panel["dataset"]], pad=6)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.30, linewidth=.6)
        ax.margins(y=.12)
        limits = list(data.lam) + [row["lam"] for row in failures]
        lo, hi = np.log10(min(limits)), np.log10(max(limits))
        ax.set_xlim(10 ** (lo-.035*(hi-lo)), 10 ** (hi+.035*(hi-lo)))
    fig.supxlabel(r"Regularization coefficient $\lambda$", fontsize=15, y=.064)
    fig.supylabel("Test cross-entropy", fontsize=15, x=.007)
    handles = [Line2D([], [], color="#0072B2", marker="o", linestyle="none", markersize=5,
                      label="Coarse grid (20)"),
               Line2D([], [], color="#E69F00", marker=".", linestyle="none", markersize=7,
                      label="Fine grid (1,000)"),
               Line2D([], [], color="#004C3F", marker="*", linestyle="none", markersize=10,
                      label="Test-CE minimum"),
               Line2D([], [], color="#C00000", marker="x", linestyle="none", markersize=6,
                      label="Failed fit")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.53, .004), columnspacing=1.2, handletextpad=.35)
    fig.tight_layout(rect=(.025, .10, 1, 1), h_pad=1.9, w_pad=1.6)
    fig.savefig(out_dir / "figure_4.png", bbox_inches="tight", pad_inches=.05)
    output = out_dir / "figure_4.pdf"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Bad value for infodict keyword CreationDate",
                                category=UserWarning)
        fig.savefig(output, bbox_inches="tight", pad_inches=.05,
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
