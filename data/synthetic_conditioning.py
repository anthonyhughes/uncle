"""Deterministic synthetic logistic data for conditioning experiments.

This module is experiment infrastructure only.  It deliberately does not
depend on either certification implementation, so Guo and UnERRING consume
exactly the same generated cohorts.

The construction couples cells as follows.  For a fixed ``cohort`` and
``data_seed``, the latent Gaussian rows, ground-truth direction, and label
uniforms do not depend on ``n`` or ``kappa``.  Smaller sample sizes are prefixes
of larger ones, while changing ``kappa`` only rescales feature coordinates.
Consequently the latent logits and binary labels are identical across all
condition numbers in a paired cell when the same signal scale is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


FORMAT_VERSION = 1
DEFAULT_CAMPAIGN_SEED = 20260902
DEFAULT_DIMENSION = 1000
DEFAULT_N_VALUES = (1000, 2000, 5000)
DEFAULT_KAPPAS = (1.0, 10.0, 100.0, 1000.0)
DEFAULT_CALIBRATION_DATA_SEEDS = tuple(range(10))
DEFAULT_EVALUATION_DATA_SEEDS = tuple(range(10))
DEFAULT_LAMBDA_GRID = np.geomspace(1e-6, 1.0, 13)

_COHORT_CODES = {
    "calibration": 1103,
    "evaluation": 2207,
}
_STREAM_CODES = {
    "direction": 11,
    "train_features": 23,
    "test_features": 37,
    "train_labels": 41,
    "test_labels": 53,
}


@dataclass(frozen=True)
class SyntheticConditioningSpec:
    """Identity of one deterministic train/test synthetic cell."""

    n: int
    d: int = DEFAULT_DIMENSION
    kappa: float = 1.0
    data_seed: int = 0
    cohort: str = "evaluation"
    campaign_seed: int = DEFAULT_CAMPAIGN_SEED
    signal_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive; got {self.n}")
        if self.d <= 0:
            raise ValueError(f"d must be positive; got {self.d}")
        if not np.isfinite(self.kappa) or self.kappa < 1.0:
            raise ValueError(f"kappa must be finite and >= 1; got {self.kappa}")
        if self.data_seed < 0:
            raise ValueError(f"data_seed must be nonnegative; got {self.data_seed}")
        if self.campaign_seed < 0:
            raise ValueError(
                f"campaign_seed must be nonnegative; got {self.campaign_seed}"
            )
        if not np.isfinite(self.signal_scale) or self.signal_scale <= 0.0:
            raise ValueError(
                "signal_scale must be finite and positive; "
                f"got {self.signal_scale}"
            )
        if self.cohort not in _COHORT_CODES:
            raise ValueError(
                f"cohort must be one of {sorted(_COHORT_CODES)}; got {self.cohort!r}"
            )

    @property
    def cell_id(self) -> str:
        return conditioning_cell_id(
            self.n, self.d, self.kappa, self.data_seed, self.cohort
        )


@dataclass(frozen=True)
class SyntheticConditioningCell:
    """Arrays and frozen provenance for one synthetic cell."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    theta_star: np.ndarray
    covariance_diagonal: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PooledLambdaSelection:
    """Pooled held-out loss scan for one ``(n, d, kappa)`` cell."""

    lambda_grid: np.ndarray
    data_seeds: tuple[int, ...]
    losses: np.ndarray
    mean_losses: np.ndarray
    selected_index: int

    @property
    def lambda_star(self) -> float:
        return float(self.lambda_grid[self.selected_index])

    @property
    def endpoint_status(self) -> str:
        if self.selected_index == 0:
            return "lower_endpoint_selected"
        if self.selected_index == len(self.lambda_grid) - 1:
            return "upper_endpoint_selected"
        return "interior"


@dataclass(frozen=True)
class SyntheticConditioningCampaign:
    """Validated frozen campaign manifest."""

    manifest_path: Path
    campaign_seed: int
    n_values: tuple[int, ...]
    d: int
    kappas: tuple[float, ...]
    calibration_data_seeds: tuple[int, ...]
    evaluation_data_seeds: tuple[int, ...]
    lambda_star_path: Path
    raw: Mapping[str, object]


def _format_number(value: float) -> str:
    return np.format_float_positional(float(value), trim="-").replace(".", "p")


def conditioning_cell_id(
    n: int,
    d: int,
    kappa: float,
    data_seed: int,
    cohort: str = "evaluation",
) -> str:
    """Return a stable human-readable cell identifier."""

    return (
        f"{cohort}-n{int(n)}-d{int(d)}-kappa{_format_number(kappa)}"
        f"-data{int(data_seed):03d}"
    )


def covariance_diagonal(d: int, kappa: float) -> np.ndarray:
    r"""Trace-normalized exponentially decaying diagonal covariance.

    Before trace normalization, coordinate ``j`` has variance

    ``exp(-log(kappa) * j / (d - 1))``.

    Thus the first-to-last ratio is exactly ``kappa`` (for ``d > 1``), and
    the diagonal is rescaled to sum to ``d``.
    """

    if d <= 0:
        raise ValueError(f"d must be positive; got {d}")
    if not np.isfinite(kappa) or kappa < 1.0:
        raise ValueError(f"kappa must be finite and >= 1; got {kappa}")
    if d == 1:
        return np.ones(1, dtype=np.float64)
    coordinate = np.arange(d, dtype=np.float64) / float(d - 1)
    raw = np.exp(-np.log(float(kappa)) * coordinate)
    return np.asarray(raw * (float(d) / float(raw.sum())), dtype=np.float64)


def _rng(spec: SyntheticConditioningSpec, stream: str) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [
            int(spec.campaign_seed),
            _COHORT_CODES[spec.cohort],
            int(spec.data_seed),
            int(spec.d),
            _STREAM_CODES[stream],
        ]
    )
    return np.random.default_rng(seed_sequence)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    # Stable logistic expression for both the historical and rescaled logits.
    return 0.5 * (1.0 + np.tanh(0.5 * logits))


def _make_split(
    spec: SyntheticConditioningSpec,
    stream_prefix: str,
    u: np.ndarray,
    sqrt_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = _rng(spec, f"{stream_prefix}_features").standard_normal(
        (spec.n, spec.d)
    )
    # ``optimize=False`` fixes a row-wise contraction whose prefix is stable
    # when n changes.  The labels therefore remain exactly paired across n.
    logits = np.einsum("ij,j->i", z, u, optimize=False)
    # Keep the historical scale-one path bit-for-bit unchanged.
    if float(spec.signal_scale) != 1.0:
        logits = float(spec.signal_scale) * logits
    uniforms = _rng(spec, f"{stream_prefix}_labels").random(spec.n)
    labels = (uniforms < _probabilities(logits)).astype(np.int8)
    features = np.multiply(z, sqrt_covariance[None, :])
    return np.asarray(features, dtype=np.float64), labels


def generate_conditioning_cell(
    spec: SyntheticConditioningSpec,
) -> SyntheticConditioningCell:
    r"""Generate one paired Gaussian-logistic train/test cell.

    If ``Sigma`` is the returned diagonal covariance, a shared unit vector
    ``u`` defines ``theta_star = Sigma^{-1/2} u``.  Hence

    ``theta_star.T @ Sigma @ theta_star == signal_scale**2``

    up to floating-point roundoff, and the latent label logit is
    ``signal_scale * z.T @ u``.  Train and test sets both contain exactly
    ``spec.n`` rows.
    """

    covariance = covariance_diagonal(spec.d, spec.kappa)
    sqrt_covariance = np.sqrt(covariance)
    u = _rng(spec, "direction").standard_normal(spec.d)
    u = np.asarray(u / np.linalg.norm(u), dtype=np.float64)
    theta_star = np.asarray(u / sqrt_covariance, dtype=np.float64)
    # Keep the historical scale-one path bit-for-bit unchanged.
    if float(spec.signal_scale) != 1.0:
        theta_star = np.asarray(
            float(spec.signal_scale) * theta_star, dtype=np.float64
        )

    X_train, y_train = _make_split(spec, "train", u, sqrt_covariance)
    X_test, y_test = _make_split(spec, "test", u, sqrt_covariance)

    metadata: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "generator": "trace_normalized_diagonal_exponential_gaussian_logistic",
        "cell_id": spec.cell_id,
        "n": int(spec.n),
        "n_train": int(spec.n),
        "n_test": int(spec.n),
        "train_split_role": (
            "lambda_selection_fit" if spec.cohort == "calibration" else "evaluation_fit"
        ),
        "test_split_role": (
            "lambda_selection_holdout"
            if spec.cohort == "calibration"
            else "final_evaluation_test"
        ),
        "d": int(spec.d),
        "kappa": float(spec.kappa),
        "data_seed": int(spec.data_seed),
        "cohort": spec.cohort,
        "campaign_seed": int(spec.campaign_seed),
        "signal_scale": float(spec.signal_scale),
        "fit_intercept": False,
        "covariance_trace": float(covariance.sum()),
        "covariance_condition_number": float(covariance[0] / covariance[-1]),
        "ground_truth_logit_variance": float(
            np.dot(theta_star * covariance, theta_star)
        ),
        "pairing_contract": (
            "direction, latent Gaussian rows, and label uniforms are shared "
            "across kappa; smaller n is a row prefix"
        ),
    }
    return SyntheticConditioningCell(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        theta_star=theta_star,
        covariance_diagonal=covariance,
        metadata=metadata,
    )


def _hash_array(hasher: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(json.dumps(contiguous.shape).encode("ascii"))
    hasher.update(memoryview(contiguous).cast("B"))


def cell_array_sha256(cell: SyntheticConditioningCell) -> str:
    """Hash all scientific arrays in a generated cell."""

    hasher = hashlib.sha256()
    for name in (
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "theta_star",
        "covariance_diagonal",
    ):
        _hash_array(hasher, name, getattr(cell, name))
    return hasher.hexdigest()


def train_test_array_sha256(cell: SyntheticConditioningCell) -> str:
    """Hash only arrays consumed as train/test data by experiment drivers."""

    hasher = hashlib.sha256()
    for name in ("X_train", "y_train", "X_test", "y_test"):
        _hash_array(hasher, name, getattr(cell, name))
    return hasher.hexdigest()


def select_pooled_lambda(
    cells: Mapping[int, SyntheticConditioningCell],
    lambda_grid: Sequence[float],
    score: Callable[[SyntheticConditioningCell, float], float],
) -> PooledLambdaSelection:
    """Select one lambda by mean held-out score across data initializations.

    ``score(cell, lambda)`` must return unregularized held-out cross-entropy
    (lower is better).  Exact ties are resolved in favor of the larger lambda.
    This function is deliberately callback-based so selection logic can be
    tested without invoking the project's heavy optimizer.
    """

    seeds = tuple(sorted(int(seed) for seed in cells))
    if not seeds:
        raise ValueError("at least one calibration cell is required")
    grid = np.asarray(lambda_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("lambda_grid must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("lambda_grid must contain finite, positive, increasing values")

    losses = np.empty((len(seeds), len(grid)), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for lambda_index, lam in enumerate(grid):
            losses[seed_index, lambda_index] = float(score(cells[seed], float(lam)))
    if not np.all(np.isfinite(losses)):
        raise ValueError("all held-out scores must be finite")
    mean_losses = losses.mean(axis=0)
    best = float(mean_losses.min())
    # Grid is increasing, so the final exact minimizer is the stronger ridge.
    selected_index = int(np.flatnonzero(mean_losses == best)[-1])
    return PooledLambdaSelection(
        lambda_grid=grid,
        data_seeds=seeds,
        losses=losses,
        mean_losses=mean_losses,
        selected_index=selected_index,
    )


def load_lambda_star(
    csv_path: str | Path,
    n: int,
    d: int,
    kappa: float,
) -> float:
    """Load the unique pooled lambda-star for a structural cell."""

    matches: list[float] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["n"]) == int(n)
                and int(row["d"]) == int(d)
                and float(row["kappa"]) == float(kappa)
            ):
                matches.append(float(row["lambda_star"]))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one lambda_star for n={n}, d={d}, kappa={kappa}; "
            f"found {len(matches)} in {csv_path}"
        )
    return matches[0]


def load_signal_scale(
    csv_path: str | Path,
    n: int,
    d: int,
    kappa: float,
) -> float:
    """Load a per-cell signal scale, defaulting old campaign tables to one."""

    matches: list[float] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["n"]) == int(n)
                and int(row["d"]) == int(d)
                and float(row["kappa"]) == float(kappa)
            ):
                raw = row.get("signal_scale")
                matches.append(1.0 if raw in (None, "") else float(raw))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one signal_scale for n={n}, d={d}, "
            f"kappa={kappa}; found {len(matches)} in {csv_path}"
        )
    value = matches[0]
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid signal_scale={value!r} in {csv_path}")
    return value


def load_campaign_manifest(path: str | Path) -> SyntheticConditioningCampaign:
    """Load and validate generator-critical fields of a campaign manifest."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "campaign_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"unsupported synthetic campaign format_version "
            f"{payload.get('format_version')!r}; expected {FORMAT_VERSION}"
        )
    generator = payload.get("generator_module")
    if generator not in {
        "data.synthetic_conditioning",
        "unerring.data.synthetic_conditioning",
    }:
        raise ValueError(f"unexpected generator_module in {manifest_path}")
    files = payload.get("files")
    if not isinstance(files, dict) or "lambda_star" not in files:
        raise ValueError(f"manifest {manifest_path} has no lambda_star file")
    return SyntheticConditioningCampaign(
        manifest_path=manifest_path,
        campaign_seed=int(payload["campaign_seed"]),
        n_values=tuple(int(value) for value in payload["n_values"]),
        d=int(payload["d"]),
        kappas=tuple(float(value) for value in payload["kappas"]),
        calibration_data_seeds=tuple(
            int(value) for value in payload["calibration_data_seeds"]
        ),
        evaluation_data_seeds=tuple(
            int(value) for value in payload["evaluation_data_seeds"]
        ),
        lambda_star_path=manifest_path.parent / str(files["lambda_star"]),
        raw=payload,
    )


def load_conditioning_cell(
    manifest_path: str | Path,
    n: int,
    d: int,
    kappa: float,
    data_seed: int,
    cohort: str = "evaluation",
) -> SyntheticConditioningCell:
    """Generate a cell using only parameters authorized by a frozen manifest."""

    campaign = load_campaign_manifest(manifest_path)
    if int(n) not in campaign.n_values:
        raise ValueError(f"n={n} is not present in {campaign.manifest_path}")
    if int(d) != campaign.d:
        raise ValueError(f"d={d} does not match manifest d={campaign.d}")
    if float(kappa) not in campaign.kappas:
        raise ValueError(f"kappa={kappa} is not present in {campaign.manifest_path}")
    if cohort not in _COHORT_CODES:
        raise ValueError(
            f"cohort must be one of {sorted(_COHORT_CODES)}; got {cohort!r}"
        )
    allowed_seeds = (
        campaign.calibration_data_seeds
        if cohort == "calibration"
        else campaign.evaluation_data_seeds
    )
    if int(data_seed) not in allowed_seeds:
        raise ValueError(
            f"data_seed={data_seed} is not authorized for cohort={cohort!r}"
        )
    signal_scale = load_signal_scale(
        campaign.lambda_star_path, int(n), int(d), float(kappa)
    )
    return generate_conditioning_cell(
        SyntheticConditioningSpec(
            n=int(n),
            d=int(d),
            kappa=float(kappa),
            data_seed=int(data_seed),
            cohort=cohort,
            campaign_seed=campaign.campaign_seed,
            signal_scale=signal_scale,
        )
    )


def load_campaign_lambda_star(
    manifest_path: str | Path, n: int, d: int, kappa: float
) -> float:
    """Load lambda-star through the manifest after validating the cell."""

    campaign = load_campaign_manifest(manifest_path)
    if (
        int(n) not in campaign.n_values
        or int(d) != campaign.d
        or float(kappa) not in campaign.kappas
    ):
        raise ValueError(
            f"structural cell n={n}, d={d}, kappa={kappa} is outside campaign"
        )
    return load_lambda_star(campaign.lambda_star_path, n, d, kappa)


def structural_cells(
    n_values: Iterable[int], d: int, kappas: Iterable[float]
) -> tuple[tuple[int, int, float], ...]:
    """Canonical ordering used by builders and Batch task mapping."""

    return tuple(
        (int(n), int(d), float(kappa))
        for n in n_values
        for kappa in kappas
    )
