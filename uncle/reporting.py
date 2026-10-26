"""Small paper plots from seed summaries; all outputs use an explicit directory."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .design import pareto_mask

if TYPE_CHECKING:
    from matplotlib.figure import Figure


LABELS = {"uncle": "UNCLE", "guo": "Guo et al. 2019", "sekhari": "Sekhari Algorithm 1"}
COLORS = {"uncle": "#0072B2", "guo": "#D55E00", "sekhari": "#009E73"}
DATASETS = {"mnist": "MNIST", "dogfish": "DogFish", "catdog": "CatDog", "imdb": "IMDB"}
NOTES = {
    "uncertainty": "Error bars are pointwise +/- one SEM across independent algorithm seeds; ten requests form one seed cluster.",
    "selection": "Utility-based choices are descriptive; SEM bars are not simultaneous or post-selection confidence intervals.",
    "sekhari": "A no-retraining rate of one is by construction for successfully computed Algorithm 1 outputs, not an empirical certificate pass rate.",
}


def _pyplot() -> ModuleType:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _errors(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in frame:
        return None
    values = frame[name].to_numpy(dtype=float)
    return values if np.isfinite(values).any() else None


def _save(figure: Figure, stem: Path, metadata: dict[str, object]) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    stem.with_suffix(".json").write_text(json.dumps({**NOTES, **metadata}, indent=2) + "\n", encoding="utf-8")
    return paths


def plot_pareto(summary: pd.DataFrame, output_dir: str | Path, *, dataset: str) -> list[Path]:
    """Write one CE/accuracy figure per k, with optional pointwise SEM bars.

    Frozen selection flags, when supplied, preserve the original chosen points.
    Otherwise each curve is the descriptive frontier of the supplied means.
    """
    plt, output_dir = _pyplot(), Path(output_dir)
    paths = []
    for k, rows in summary.groupby("k", sort=True):
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
        for axis, metric, cost, sem, label in (
            (axes[0], "ce", "mean_ce_cost", "sem_ce_cost", "Cross-entropy increase"),
            (axes[1], "accuracy", "mean_accuracy_drop", "sem_accuracy_drop", "Accuracy drop"),
        ):
            for method, group in rows.groupby("method", sort=True):
                selected = "selected_by_" + metric
                if selected in group:
                    chosen = group[selected].astype(str).str.lower().isin(("true", "1"))
                    order = [metric + "_order", "point_id"]
                else:
                    chosen = pareto_mask(group[cost], group.mean_success_rate)
                    order = [cost, "point_id"]
                frontier = group[chosen].sort_values(order)
                axis.scatter(group[cost], group.mean_success_rate, s=8, alpha=.12,
                             color=COLORS[method], edgecolors="none")
                if frontier.empty:
                    continue
                axis.errorbar(frontier[cost], frontier.mean_success_rate,
                              xerr=_errors(frontier, sem), yerr=_errors(frontier, "sem_success_rate"),
                              label=LABELS[method], color=COLORS[method], fmt="o-", markersize=3,
                              linewidth=1.3, elinewidth=.7, capsize=2)
            axis.set_xlabel(label)
            axis.set_ylim(-.03, 1.03)
            axis.grid(alpha=.2)
        axes[0].set_ylabel("No-retraining rate")
        axes[1].legend(frameon=False, fontsize=8)
        figure.suptitle(f"{DATASETS.get(dataset, dataset)} - Pareto Curve")
        figure.text(.5, .005, f"Samples removed: {int(k)}", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, .04, 1, .94))
        paths += _save(figure, output_dir / f"{dataset}_k{int(k)}_pareto",
                       dict(dataset=dataset, k=int(k), frozen_selection="selected_by_ce" in rows))
        plt.close(figure)
    return paths


def summary_points(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the two requested operating points, retaining gaps when infeasible."""
    records = []
    for (method, k), group in summary.groupby(["method", "k"], sort=True):
        row = dict(method=method, k=int(k), accuracy_drop=np.nan, sem_accuracy_drop=np.nan,
                   retrain_rate=np.nan, sem_retrain_rate=np.nan, accuracy_point_id="", retrain_point_id="")
        feasible = group[group.mean_success_rate >= .9]
        if not feasible.empty:
            best = feasible.sort_values(["mean_accuracy_drop", "point_id"]).iloc[0]
            row.update(accuracy_drop=best.mean_accuracy_drop,
                       sem_accuracy_drop=best.get("sem_accuracy_drop", np.nan), accuracy_point_id=best.point_id)
        feasible = group[group.mean_ce_cost <= .01]
        if not feasible.empty:
            best = feasible.sort_values(["mean_success_rate", "point_id"], ascending=[False, True]).iloc[0]
            row.update(retrain_rate=1. - best.mean_success_rate,
                       sem_retrain_rate=best.get("sem_success_rate", np.nan), retrain_point_id=best.point_id)
        records.append(row)
    return pd.DataFrame(records)


def plot_summary(summary: pd.DataFrame, output_dir: str | Path, *, dataset: str) -> list[Path]:
    """Write the accuracy-drop and retraining-rate panels against samples removed."""
    plt, output_dir = _pyplot(), Path(output_dir)
    selected = summary_points(summary)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for method, group in selected.groupby("method", sort=True):
        group = group.sort_values("k")
        for axis, metric in zip(axes, ("accuracy_drop", "retrain_rate")):
            axis.errorbar(group.k, group[metric], yerr=_errors(group, "sem_" + metric),
                          fmt="o-", color=COLORS[method], label=LABELS[method], markersize=3,
                          linewidth=1.3, elinewidth=.7, capsize=2)
    axes[0].set_ylabel("Accuracy drop")
    axes[0].set_title("No-retraining rate ≥ 90%", fontsize=10)
    axes[1].set_ylabel("Retraining rate")
    axes[1].set_title("Cross-entropy increase ≤ 0.01", fontsize=10)
    for axis in axes:
        axis.set_xlabel("Samples removed")
        axis.grid(alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(DATASETS.get(dataset, dataset))
    figure.tight_layout(rect=(0, 0, 1, .94))
    paths = _save(figure, output_dir / f"{dataset}_summary",
                  dict(dataset=dataset, success_threshold=.9, ce_cost_threshold=.01))
    selected.to_csv(output_dir / f"{dataset}_summary_points.csv", index=False)
    plt.close(figure)
    return paths
