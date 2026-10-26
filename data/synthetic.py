from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from data.common import Dataset, SplitDataset


CovarianceType = Literal["isotropic", "ill_conditioned", "ill_normalized"]

COV_SHORT_NAMES = {
    "isotropic": "iso",
    "ill_conditioned": "ill",
    "ill_normalized": "illn",
}


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for synthetic Gaussian logistic regression data."""

    n: int
    d: int
    cov_type: CovarianceType = "isotropic"
    decay: float = 0.95
    kappa: float = 165.0
    seed: int = 42
    num_test: Optional[int] = None
    fit_intercept: bool = True


def synthetic_config_name(config: SyntheticConfig) -> str:
    cov = COV_SHORT_NAMES[config.cov_type]
    affine = "_affine" if config.fit_intercept else ""
    return f"synthetic_{cov}_n{config.n}_d{config.d}{affine}_seed{config.seed}"


def default_removal_sizes(n: int) -> list[int]:
    """Removal sizes: k=1 plus 0.5%–3% of n (deduplicated, sorted)."""
    sizes = {1}
    for pct in (0.5, 1.0, 2.0, 3.0):
        sizes.add(max(1, int(round(pct * n / 100.0))))
    return sorted(sizes)


def _covariance_factor(
    rng: np.random.Generator,
    d: int,
    cov_type: CovarianceType,
    decay: float,
    kappa: float = 165.0,
) -> Optional[np.ndarray]:
    """Factor A with feature covariance A A^T; None means identity (isotropic).

    Drawn ONCE per dataset and shared by every split: a fresh random eigenbasis
    per call would give train and test covariances with the same eigenvalues
    but independently rotated eigenvectors — a train/test distribution shift.

    Every anisotropic branch must consume exactly one standard_normal((d, d))
    draw (the eigenbasis) and nothing else: downstream tooling replays this rng
    sequence from the persisted generator seed to recover theta*/Sigma.
    """
    if cov_type == "isotropic":
        return None
    if cov_type == "ill_conditioned":
        eigvals = decay ** np.arange(d)
    elif cov_type == "ill_normalized":
        # Fixed condition number kappa at every d, trace normalized to d
        # (mean eigenvalue 1): effective rank grows with d and the label SNR
        # under a unit-norm theta* is d-independent, unlike ill_conditioned
        # whose trace saturates at 1/(1-decay).
        eigvals = kappa ** (-np.arange(d) / max(d - 1, 1))
        eigvals = eigvals * (d / eigvals.sum())
    else:
        raise ValueError(f"unknown cov_type: {cov_type!r}")
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q @ np.diag(np.sqrt(eigvals))


def _sample_features(
    rng: np.random.Generator,
    count: int,
    d: int,
    chol: Optional[np.ndarray],
) -> np.ndarray:
    z = rng.standard_normal((count, d))
    return z if chol is None else z @ chol.T


def _labels_from_theta(
    rng: np.random.Generator,
    features: np.ndarray,
    theta_star: np.ndarray,
    bias_star: float = 0.0,
) -> np.ndarray:
    logits = features @ theta_star + bias_star
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (rng.random(features.shape[0]) < probs).astype(np.int64)


def generate_synthetic_logistic(
    config: SyntheticConfig,
) -> Tuple[SplitDataset, np.ndarray, dict]:
    """Generate synthetic logistic regression data.

    ``config.n`` is the number of **training** samples; test size defaults to
  ``int(0.2 * n)``.

    Returns:
        SplitDataset with train/test splits.
        theta_star: unit-norm ground-truth weight vector on raw features.
        metadata: generation parameters and diagnostics.
    """
    if config.n < 1 or config.d < 1:
        raise ValueError(f"n and d must be positive; got n={config.n}, d={config.d}")

    rng = np.random.default_rng(config.seed)
    n_train = config.n
    n_test = config.num_test if config.num_test is not None else int(0.2 * config.n)

    theta_star = rng.standard_normal(config.d)
    theta_star = theta_star / np.linalg.norm(theta_star)
    bias_star = float(rng.standard_normal()) if config.fit_intercept else 0.0

    chol = _covariance_factor(
        rng, config.d, config.cov_type, config.decay, config.kappa
    )
    train_features = _sample_features(rng, n_train, config.d, chol)
    test_features = _sample_features(rng, n_test, config.d, chol)
    train_labels = _labels_from_theta(
        rng, train_features, theta_star, bias_star=bias_star
    )
    test_labels = _labels_from_theta(
        rng, test_features, theta_star, bias_star=bias_star
    )

    dataset = SplitDataset(
        train=Dataset(features=train_features, labels=train_labels),
        test=Dataset(features=test_features, labels=test_labels),
    )

    metadata = {
        "theta_star": theta_star,
        "bias_star": bias_star,
        "fit_intercept": config.fit_intercept,
        "cov_type": config.cov_type,
        "decay": config.decay,
        "kappa": config.kappa,
        "n": config.n,
        "d": config.d,
        "n_train": n_train,
        "n_test": n_test,
        "seed": config.seed,
        "config_name": synthetic_config_name(config),
        "synthetic_random_only": True,
    }
    return dataset, theta_star, metadata


def create_synthetic_logistic_regression_data(
    n: int,
    d: int,
    num_test: Optional[int] = None,
    ground_truth_model: Optional[np.ndarray] = None,
) -> SplitDataset:
    """Generate isotropic synthetic data (legacy API)."""
    rng = np.random.default_rng()
    if num_test is None:
        num_test = int(0.2 * n)

    if ground_truth_model is None:
        theta_star = rng.standard_normal(d)
        theta_star = theta_star / np.linalg.norm(theta_star)
    else:
        theta_star = np.asarray(ground_truth_model, dtype=np.float64)

    train_features = rng.standard_normal((n, d))
    test_features = rng.standard_normal((num_test, d))
    train_labels = _labels_from_theta(rng, train_features, theta_star)
    test_labels = _labels_from_theta(rng, test_features, theta_star)

    return SplitDataset(
        train=Dataset(features=train_features, labels=train_labels),
        test=Dataset(features=test_features, labels=test_labels),
    )
