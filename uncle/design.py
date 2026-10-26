"""Frozen frontier selection and the active Guo/Sekhari refinement designs.

Guo uses the existing seeded arc-length/Gaussian log tube, with rejection at
frozen bounds. Sekhari uses the existing 32 coarse / 64 new lambda rule.
Neither algorithm fits a model or changes an engine's numerical conditions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import heapq
import json
import math

import numpy as np
import pandas as pd


POINT_FIELDS = {
    "uncle": ("geometry_multiplier", "radius_factor"),
    "guo": ("sigma",),
    "sekhari": (),
}


def point_id(row: Mapping) -> str:
    """Identify one operating point; derived tau and irrelevant NaNs are omitted."""
    method = str(row["method"])
    if method not in POINT_FIELDS:
        raise ValueError(f"unknown method: {method}")
    k = float(row["k"])
    if not math.isfinite(k) or k <= 0 or int(k) != k:
        raise ValueError("k must be a positive integer")
    values = [method, int(k)]
    for name in ("lam", *POINT_FIELDS[method]):
        values.append(_positive(row[name], name).hex())
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return "point-" + hashlib.sha256(encoded).hexdigest()[:24]


def _identified(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("operating-point table is empty")
    identities = result.apply(point_id, axis=1)
    if "point_id" in result and not result.point_id.eq(identities).all():
        raise ValueError("point_id disagrees with actual parameters")
    result["point_id"] = identities
    return result


def pareto_mask(cost: Sequence[float], success: Sequence[float]) -> np.ndarray:
    """Minimize cost and maximize success; retain all exact performance ties."""
    cost, success = np.asarray(cost, dtype=float), np.asarray(success, dtype=float)
    if cost.ndim != 1 or success.shape != cost.shape or not np.isfinite([cost, success]).all():
        raise ValueError("cost and success must be finite matching one-dimensional arrays")
    return np.array([not np.any((cost <= x) & (success >= y) & ((cost < x) | (success > y)))
                     for x, y in zip(cost, success)], dtype=bool)


def summarize_seeds(
    rows: pd.DataFrame, expected_seeds: int | Sequence[int] | None = None,
    requests_per_seed: int = 10,
) -> pd.DataFrame:
    """Aggregate one dataset, with each algorithm seed as one independent unit.

    Each row already contains success over its request cluster and utility on
    its predetermined selected request. A seed count means range(count); an
    explicit sequence means those exact indices. None requires the same observed
    seed set at every point. SEM uses sample SD/sqrt(seeds), never sqrt(requests).
    A single seed has unknown SEM. Sekhari's rate of one is by construction.
    """
    frame = _identified(rows)
    if "dataset" in frame and frame.dataset.nunique(dropna=False) != 1:
        raise ValueError("summarize one dataset at a time")
    required = ["test_ce", "test_accuracy", "success_rate", "ce_reference", "accuracy_reference", "seed_index"]
    if not set(required).issubset(frame) or not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("finite seed indices, utilities, rates, and references are required")
    if (not frame.test_ce.ge(0).all() or not frame.test_accuracy.between(0, 1).all()
            or not frame.success_rate.between(0, 1).all() or not frame.ce_reference.ge(0).all()
            or not frame.accuracy_reference.between(0, 1).all()):
        raise ValueError("invalid utility, no-retraining rate, or reference")
    if any(frame[name].nunique() != 1 for name in ("ce_reference", "accuracy_reference")):
        raise ValueError("metric-specific references must be frozen across all points")
    if not (frame.seed_index.ge(0) & frame.seed_index.eq(frame.seed_index.astype(int))).all():
        raise ValueError("seed_index must contain nonnegative integers")
    requests_per_seed = _count(requests_per_seed, 1)
    if "n_removal_requests" in frame and not frame.n_removal_requests.eq(requests_per_seed).all():
        raise ValueError("request cluster size differs")
    if frame.duplicated(["point_id", "seed_index"]).any():
        raise ValueError("duplicate operating-point/seed rows")
    if isinstance(expected_seeds, int):
        expected = set(range(_count(expected_seeds, 1)))
    else:
        expected = set(frame.seed_index) if expected_seeds is None else set(expected_seeds)
        if not expected or any(not math.isfinite(float(i)) or i < 0 or int(i) != i for i in expected):
            raise ValueError("expected seeds must be nonnegative integer indices")
    records = []
    parameters = ("dataset", "method", "lam", "k", "sigma", "geometry_multiplier", "radius_factor", "tau", "R2")
    for identity, group in frame.groupby("point_id", sort=True):
        if set(group.seed_index) != expected:
            raise ValueError("incomplete independent seed set")
        group = group.sort_values("seed_index")
        if group["method"].iloc[0] == "uncle":
            for name in ("tau", "R2"):
                if name in group and group[name].nunique(dropna=False) != 1:
                    raise ValueError("derived geometry changed within one point")
        first = group.iloc[0]
        result = {name: first[name] for name in parameters if name in group}
        result.update(point_id=identity, n_seeds=len(group), n_seed_trials=len(group),
                      n_request_trials=0 if first["method"] == "sekhari" else len(group) * requests_per_seed,
                      n_utility_trials=len(group),
                      ce_reference=float(first.ce_reference), accuracy_reference=float(first.accuracy_reference))
        for metric in ("test_ce", "test_accuracy", "success_rate"):
            result["mean_" + metric] = float(group[metric].mean())
            result["sem_" + metric] = float(group[metric].std(ddof=1) / np.sqrt(len(group)))
        result["mean_ce_cost"] = result["mean_test_ce"] - result["ce_reference"]
        result["mean_accuracy_drop"] = result["accuracy_reference"] - result["mean_test_accuracy"]
        result["sem_ce_cost"], result["sem_accuracy_drop"] = result["sem_test_ce"], result["sem_test_accuracy"]
        result["mean_retrain_rate"] = 1. - result["mean_success_rate"]
        result["sem_retrain_rate"] = result["sem_success_rate"]
        records.append(result)
    return pd.DataFrame(records).sort_values(["method", "k", "lam", "point_id"]).reset_index(drop=True)


def _frontier_flags(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = _identified(summary)
    if result.point_id.duplicated().any():
        raise ValueError("duplicate summary points")
    for metric, cost in (("ce", "mean_ce_cost"), ("accuracy", "mean_accuracy_drop")):
        result[prefix + metric] = False
        for _, group in result.groupby(["method", "k"], sort=True):
            result.loc[group.index, prefix + metric] = pareto_mask(group[cost], group.mean_success_rate)
    return result


def freeze_frontiers(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed union of CE and accuracy frontiers per (method, k)."""
    result = _frontier_flags(summary, "selected_by_")
    result = result[result.selected_by_ce | result.selected_by_accuracy].copy()
    for metric, cost in (("ce", "mean_ce_cost"), ("accuracy", "mean_accuracy_drop")):
        result[metric + "_order"] = -1
        for _, group in result[result["selected_by_" + metric]].groupby(["method", "k"]):
            order = group.sort_values([cost, "point_id"]).index
            result.loc[order, metric + "_order"] = np.arange(len(order))
    return result.sort_values(["method", "k", "point_id"]).reset_index(drop=True)


def mark_fresh_frontiers(summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive reselection among evaluated frozen points; retain every row."""
    return _frontier_flags(summary, "fresh_pareto_")


def guo_refinement_design(
    summary: pd.DataFrame, *, dataset: str, count: int, master_seed: int,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, np.ndarray]:
    """Adapt seed summaries to the unchanged k=1 Gaussian-tube sampler below."""
    count = _count(count, 1)
    frame = summary[summary.method.eq("guo")].copy() if "method" in summary else summary.copy()
    frame["mean_excess_test_loss"] = frame["mean_ce_cost"]
    frame["success_rate"] = frame["mean_success_rate"]
    frame["on_frontier"] = False
    for _, group in frame.groupby("k"):
        frame.loc[group.index, "on_frontier"] = pareto_mask(group.mean_ce_cost, group.mean_success_rate)
    points = len(frame[["lam", "sigma"]].drop_duplicates())
    if not points:
        raise ValueError("Guo refinement needs completed coarse points")
    return build_refinement_design(frame, dataset=dataset, budget_multiplier=count / points,
                                   master_seed=master_seed, allow_singleton_frontier=True)


DATASET_ALIASES = {
    "mnist": "mnist",
    "dogfish": "dogfish",
    "noisy_dogfish": "noisy_dogfish",
    "catdog": "cifar10_catdog_resnet18",
    "imdb": "imdb",
    "auto_truck": "cifar10_auto_truck_resnet18",
    "esc50": "esc50",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def smooth_polyline(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Smooth a polyline with endpoint-preserving Chaikin corner cutting."""
    current = np.asarray(points, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 2 or current.shape[0] < 2:
        raise ValueError("points must have shape (n, 2) with n >= 2")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    for _ in range(iterations):
        refined = [current[0]]
        for left, right in zip(current[:-1], current[1:]):
            refined.extend((0.75 * left + 0.25 * right,
                            0.25 * left + 0.75 * right))
        refined.append(current[-1])
        current = np.asarray(refined, dtype=np.float64)
    return current


def _sample_polyline(
    points: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Interpolate points at coordinates in [0, 1], uniformly by arc length."""
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-14))
    points = points[keep]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0:
        raise ValueError("frontier path has zero length in log space")
    targets = np.asarray(coordinates, dtype=np.float64) * cumulative[-1]
    segment = np.searchsorted(cumulative, targets, side="right") - 1
    segment = np.clip(segment, 0, len(segment_lengths) - 1)
    local = (targets - cumulative[segment]) / segment_lengths[segment]
    return points[segment] + local[:, None] * (points[segment + 1] - points[segment])


def _median_grid_spacing(values: pd.Series) -> float:
    unique = np.unique(np.log10(values.to_numpy(dtype=np.float64)))
    differences = np.diff(unique)
    differences = differences[differences > 1e-12]
    if differences.size == 0:
        raise ValueError("the first-stage design needs at least two distinct values")
    return float(np.median(differences))


def _dataset_seed(master_seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{dataset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def build_refinement_design(
    summary: pd.DataFrame,
    *,
    dataset: str,
    budget_multiplier: float = 3.0,
    master_seed: int = 20260827,
    noise_scale: float = 0.5,
    smoothing_iterations: int = 3,
    allow_singleton_frontier: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, np.ndarray]:
    """Return the explicit parameter design, metadata, raw path, and smooth path."""
    required = {
        "k", "lam", "sigma", "mean_excess_test_loss", "success_rate",
        "on_frontier",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    if dataset not in DATASET_ALIASES:
        raise ValueError(f"unknown dataset: {dataset}")
    if budget_multiplier <= 0 or noise_scale < 0:
        raise ValueError("budget_multiplier must be positive and noise_scale nonnegative")

    positive = summary[(summary["lam"] > 0) & (summary["sigma"] > 0)].copy()
    wide = positive[["lam", "sigma"]].drop_duplicates()
    first_stage_points = int(len(wide))
    n_samples = int(round(first_stage_points * budget_multiplier))
    if n_samples < 1:
        raise ValueError("refinement budget produced no samples")

    k1 = positive[
        (positive["k"].astype(int) == 1) & _as_bool(positive["on_frontier"])
    ].copy()
    k1 = (
        k1.sort_values(
            ["mean_excess_test_loss", "success_rate", "lam", "sigma"],
            ascending=[True, True, True, True],
        )
        .drop_duplicates(["lam", "sigma"], keep="first")
        .reset_index(drop=True)
    )
    if len(k1) < 2 and not (
        len(k1) == 1 and allow_singleton_frontier
    ):
        raise ValueError("the k=1 Pareto frontier needs at least two parameter points")

    raw_path = np.column_stack((
        np.log10(k1["lam"].to_numpy(dtype=np.float64)),
        np.log10(k1["sigma"].to_numpy(dtype=np.float64)),
    ))
    singleton_frontier = len(k1) == 1
    smooth_path = (
        raw_path.copy()
        if singleton_frontier
        else smooth_polyline(raw_path, iterations=smoothing_iterations)
    )

    lambda_spacing = _median_grid_spacing(wide["lam"])
    sigma_spacing = _median_grid_spacing(wide["sigma"])
    noise_std = np.asarray(
        [noise_scale * lambda_spacing, noise_scale * sigma_spacing],
        dtype=np.float64,
    )
    lower = np.log10(wide[["lam", "sigma"]].min().to_numpy(dtype=np.float64))
    upper = np.log10(wide[["lam", "sigma"]].max().to_numpy(dtype=np.float64))

    seed = _dataset_seed(master_seed, dataset)
    rng = np.random.default_rng(seed)
    curve_coordinates = rng.random(n_samples)
    base = (
        np.repeat(raw_path, n_samples, axis=0)
        if singleton_frontier
        else _sample_polyline(smooth_path, curve_coordinates)
    )
    noise = rng.normal(size=(n_samples, 2)) * noise_std
    proposed = base + noise
    outside = np.any((proposed < lower) | (proposed > upper), axis=1)
    rejection_rounds = 0
    while np.any(outside):
        rejection_rounds += 1
        if rejection_rounds > 10_000:
            raise RuntimeError("could not sample the Gaussian tube inside wide-grid bounds")
        noise[outside] = rng.normal(size=(int(outside.sum()), 2)) * noise_std
        proposed[outside] = base[outside] + noise[outside]
        outside = np.any((proposed < lower) | (proposed > upper), axis=1)

    design = pd.DataFrame({
        "lambda": np.power(10.0, proposed[:, 0]),
        "sigma": np.power(10.0, proposed[:, 1]),
        "dataset_name": DATASET_ALIASES[dataset],
        "design_id": [f"k1-tube-{index:04d}" for index in range(n_samples)],
        "role": "k1_pareto_gaussian_tube",
        "source_k": 1,
        "curve_coordinate": curve_coordinates,
        "base_log10_lambda": base[:, 0],
        "base_log10_sigma": base[:, 1],
        "noise_log10_lambda": noise[:, 0],
        "noise_log10_sigma": noise[:, 1],
    })
    metadata: dict[str, object] = {
        "format_version": 1,
        "dataset": dataset,
        "dataset_name": DATASET_ALIASES[dataset],
        "source_k": 1,
        "first_stage_parameter_points": first_stage_points,
        "budget_multiplier": budget_multiplier,
        "refinement_parameter_points": n_samples,
        "master_seed": master_seed,
        "dataset_seed": seed,
        "smoothing": "endpoint-preserving Chaikin corner cutting",
        "smoothing_iterations": smoothing_iterations,
        "sampling": "uniform by smoothed-curve arc length plus independent Gaussian log noise",
        "bounds_policy": "reject outside the original first-stage grid bounds",
        "noise_scale_in_grid_steps": noise_scale,
        "lambda_grid_spacing_log10": lambda_spacing,
        "sigma_grid_spacing_log10": sigma_spacing,
        "lambda_noise_std_log10": float(noise_std[0]),
        "sigma_noise_std_log10": float(noise_std[1]),
        "lambda_bounds": [float(wide["lam"].min()), float(wide["lam"].max())],
        "sigma_bounds": [float(wide["sigma"].min()), float(wide["sigma"].max())],
        "k1_frontier_points": int(len(k1)),
        "rejection_rounds": rejection_rounds,
    }
    # Preserve the generic sampler's metadata contract when the new option is
    # omitted.  Synthetic campaigns opt in explicitly, so their artifacts say
    # whether the otherwise-identical tube used a polyline or a degenerate
    # one-point curve.
    if allow_singleton_frontier:
        metadata.update({
            "allow_singleton_frontier": True,
            "singleton_frontier_used": singleton_frontier,
            "frontier_geometry": (
                "singleton_degenerate_curve" if singleton_frontier else "polyline"
            ),
        })
        if singleton_frontier:
            metadata.update({
                "smoothing": "not applied to singleton degenerate curve",
                "sampling": (
                    "fixed singleton base point plus independent Gaussian log noise"
                ),
                "singleton_log10_point": raw_path[0].tolist(),
            })
    return design, metadata, raw_path, smooth_path


OBJECTIVES = (("mean_test_ce", "min"), ("mean_test_accuracy", "max"))


def _count(value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"count must be an integer >= {minimum}")
    return value


def _positive(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _grid(values: Sequence[float], name: str) -> list[float]:
    result = sorted({_positive(value, name) for value in values})
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two distinct lambdas")
    return result


def coarse_lambdas(config: Mapping, count: int = 32) -> list[float]:
    """Use exactly ``count`` log-spaced candidates, retaining reference anchors.

    Start with an endpoint-inclusive log-uniform grid over min/max guo_lambdas.
    For each distinct in-range reference (ascending lambda), replace the nearest
    unprotected grid point in log space; lower lambda breaks distance ties.
    Endpoints and both references are protected, so inserting an anchor never
    increases the budget or changes the frozen search interval.
    """
    count = _count(count, 2)
    source = _grid(config["guo_lambdas"], "guo_lambdas")
    lower, upper = source[0], source[-1]
    references = config["reference_lambdas"]
    anchors = {_positive(references[metric], f"{metric} reference lambda")
               for metric in ("test_ce", "test_accuracy")}
    anchors = {value for value in anchors if lower <= value <= upper}
    protected = anchors | {lower, upper}
    if len(protected) > count:
        raise ValueError("count cannot retain both endpoints and in-range references")
    lo_log, hi_log = math.log(lower), math.log(upper)
    grid = [lower] + [math.exp(lo_log + (hi_log - lo_log) * i / (count - 1))
                      for i in range(1, count - 1)] + [upper]
    if len(set(grid)) != count or any(not lower <= value <= upper for value in grid):
        raise ValueError("frozen interval is too narrow for the requested coarse count")
    for anchor in sorted(anchors):
        if anchor in grid:
            continue
        candidates = [value for value in grid if value not in protected]
        nearest = min(candidates, key=lambda value: (abs(math.log(value) - math.log(anchor)), value))
        grid.remove(nearest)
        grid.append(anchor)
    return sorted(grid)


def _summary_rows(summary: pd.DataFrame, coarse: list[float]) -> list[dict]:
    frame = pd.DataFrame(summary).copy()
    required = {"dataset", "lam", "k", *(name for name, _ in OBJECTIVES)}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("coarse summary requires dataset, lam, k, mean_test_ce, mean_test_accuracy")
    if frame["dataset"].isna().any() or frame["dataset"].nunique() != 1:
        raise ValueError("refine one dataset at a time")
    rows = []
    identities = set()
    for raw in frame.to_dict("records"):
        lam = _positive(raw["lam"], "summary lambda")
        if lam not in coarse:
            raise ValueError("coarse summary contains an unplanned lambda")
        k = _positive(raw["k"], "summary k")
        if int(k) != k:
            raise ValueError("summary k must be an integer")
        k = int(k)
        identity = (k, lam)
        if identity in identities:
            raise ValueError("duplicate (k, lambda) in coarse summary")
        identities.add(identity)
        ce, accuracy = float(raw["mean_test_ce"]), float(raw["mean_test_accuracy"])
        if not math.isfinite(ce) or ce < 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("coarse summary has invalid utility; failed outputs cannot select refinement")
        rows.append(dict(dataset=str(raw["dataset"]), lam=lam, k=k,
                         mean_test_ce=ce, mean_test_accuracy=accuracy))
    ks = sorted({row["k"] for row in rows})
    if identities != {(k, lam) for k in ks for lam in coarse}:
        raise ValueError("coarse summary must include every coarse lambda for every k")
    return sorted(rows, key=lambda row: (row["k"], row["lam"]))


def refinement_lambdas(summary: pd.DataFrame, coarse: Sequence[float], count: int = 64) -> tuple[list[float], dict]:
    """Add ``count`` unique lambdas within all per-k utility-winner brackets.

    ``summary`` contains one complete row per (dataset, lambda, k), with columns
    dataset, lam, k, mean_test_ce, mean_test_accuracy. Min CE and max accuracy
    independently select a winner for every k; exact ties prefer low lambda.
    Each bracket spans the adjacent coarse points around its winner. Endpoint
    winners use their one adjacent interval and are flagged in metadata.

    Repeatedly bisect the widest remaining log gap in the union of brackets,
    breaking equal-width ties by low lambda. Coarse points split initial gaps,
    so all returned points are new. Overlapping brackets count only once.
    """
    count = _count(count, 1)
    supplied = list(coarse)
    grid = _grid(supplied, "coarse lambdas")
    if len(grid) != len(supplied):
        raise ValueError("coarse lambdas must be distinct")
    rows = _summary_rows(summary, grid)
    index = {lam: i for i, lam in enumerate(grid)}
    winners, edges = [], set()
    for k in sorted({row["k"] for row in rows}):
        candidates = [row for row in rows if row["k"] == k]
        for objective, direction in OBJECTIVES:
            winner = min(candidates, key=lambda row: (
                row[objective] if direction == "min" else -row[objective], row["lam"]))
            i = index[winner["lam"]]
            left, right = max(0, i - 1), min(len(grid) - 1, i + 1)
            edges.update(range(left, right))
            boundary = "lower" if i == 0 else "upper" if i == len(grid) - 1 else None
            winners.append(dict(dataset=winner["dataset"], k=k, objective=objective,
                                direction=direction, lam=winner["lam"], value=winner[objective],
                                bracket=[grid[left], grid[right]], boundary=boundary))

    union = []
    for edge in sorted(edges):
        left, right = grid[edge], grid[edge + 1]
        if union and union[-1][1] == left:
            union[-1][1] = right
        else:
            union.append([left, right])
    heap = [(-(math.log(grid[i + 1]) - math.log(grid[i])), grid[i], grid[i + 1])
            for i in edges]
    heapq.heapify(heap)
    added = []
    while len(added) < count:
        if not heap:
            raise ValueError("refinement brackets have too few distinct representable lambdas")
        _, left, right = heapq.heappop(heap)
        middle = math.exp(math.log(left) + (math.log(right) - math.log(left)) / 2)
        if not left < middle < right:
            middle = math.nextafter(left, right)
        if not left < middle < right:
            continue
        added.append(middle)
        for a, b in ((left, middle), (middle, right)):
            heapq.heappush(heap, (-(math.log(b) - math.log(a)), a, b))

    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    metadata = dict(
        schema="sekhari_lambda_refinement_v1", algorithm="widest_log_gap_bisection",
        dataset=rows[0]["dataset"], count=count, coarse_count=len(grid),
        frozen_lambda_bounds=[grid[0], grid[-1]], coarse_lambdas=grid,
        objectives=[name for name, _ in OBJECTIVES], selection="utility_only",
        objective_tie_break="lowest_lambda", gap_tie_break="lowest_lambda",
        objective_winners=winners,
        boundary_winners=[winner.copy() for winner in winners if winner["boundary"] is not None],
        interval_union=union, coarse_summary_sha256=hashlib.sha256(encoded).hexdigest(),
        range_expanded=False,
    )
    result = sorted(added)
    metadata["interval_point_counts"] = [sum(left < lam < right for lam in result)
                                         for left, right in union]
    return result, metadata
