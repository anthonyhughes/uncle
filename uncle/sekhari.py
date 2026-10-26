"""Sekhari et al., Algorithm 1: a Newton update plus isotropic Gaussian noise.

Unlike UNCLE/Guo, its retained objective is mean-retained CE + ridge/2 ||theta||².
There is no acceptance/retrain test, no tunable noise multiplier, and no output
geometry scan. The only experiment hyperparameter is the training ridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from scipy.special import expit

Array = NDArray[np.float64]
Indices = NDArray[np.int64]
THIRD_DERIVATIVE_CAP = float(1 / (6 * np.sqrt(3)))


@dataclass(frozen=True)
class SekhariGeometry:
    """Full-data scalar bounds, shared by every lambda and deletion request."""

    n: int
    d: int
    gram_op: float
    squared_gram_op: float
    feature_sha256: str

    @property
    def M_sum(self) -> float:
        # No /n: preserve the approved conservative full-SUM-loss bound.
        return float(THIRD_DERIVATIVE_CAP * np.sqrt(self.gram_op) * np.sqrt(self.squared_gram_op))

    @property
    def loss_lipschitz(self) -> float:
        return float(np.sqrt(self.gram_op))

    @classmethod
    def build(cls, X: Array) -> SekhariGeometry:
        """Compute actual dense spectral targets, not trace/power substitutes.

        M = cap * sqrt(||X.T X||op * ||(X X.T) elementwise-squared||op).
        This is heavy setup: O(n²d+n³+min(n,d)³) time, O(n²) Gram storage.
        The float64 eigensolver is the existing policy, not interval arithmetic.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix")
        n, d = X.shape
        gram = X @ X.T
        small = X.T @ X if d < n else gram
        last = len(small) - 1
        gram_op = max(0.0, float(eigvalsh(small, subset_by_index=(last, last), check_finite=True)[0]))
        np.square(gram, out=gram)
        squared_op = max(0.0, float(eigvalsh(gram, subset_by_index=(n - 1, n - 1), check_finite=True)[0]))
        fingerprint = hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest()
        result = cls(n, d, gram_op, squared_op, fingerprint)
        if not np.isfinite(result.M_sum):
            raise ValueError("Hessian Lipschitz bound overflowed float64")
        return result


def noise_scale(
    n: int, k: int, training_ridge: float, M_sum: float, loss_lipschitz: float,
    epsilon: float, delta: float,
) -> tuple[float, float]:
    """Algorithm 1: return its prescribed (sensitivity, isotropic noise std)."""
    if (isinstance(n, (bool, np.bool_)) or isinstance(k, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or not isinstance(k, (int, np.integer))
            or not 1 <= k < n):
        raise ValueError("require integer counts 1 <= k < n")
    if (not np.isfinite([training_ridge, M_sum, loss_lipschitz, epsilon, delta]).all()
            or training_ridge <= 0 or M_sum < 0 or loss_lipschitz < 0
            or not 0 < epsilon <= 1 or not 0 < delta < 1):
        raise ValueError("invalid ridge, bounds, or privacy budget")
    sensitivity = 2 * M_sum * (k / n) ** 2 * loss_lipschitz ** 2 / training_ridge ** 3
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    if not np.isfinite([sensitivity, sigma]).all():
        raise ValueError("noise scale overflowed float64")
    return float(sensitivity), float(sigma)


@dataclass(frozen=True)
class SekhariResult:
    theta: Array
    theta_newton: Array
    noise: Array
    sensitivity: float
    sigma: float


@dataclass
class SekhariModel:
    X: Array
    y: Array
    theta: Array
    training_ridge: float
    geometry: SekhariGeometry
    data_hessian_sum: Array
    full_gradient_norm: float
    factors: dict[int, tuple[Array, bool]] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls, X: Array, y: Array, theta: Array, training_ridge: float,
        geometry: SekhariGeometry | None = None,
    ) -> SekhariModel:
        """Use a fitted full model; geometry must describe this exact dataset.

        Record stationarity error rather than changing the prescribed Newton
        update to correct it. A numerical optimizer is not a proof of exact ERM.
        """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        theta = np.array(theta, dtype=np.float64, copy=True)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix")
        if y.shape != (len(X),) or not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must contain binary labels")
        if theta.shape != (X.shape[1],) or not np.isfinite(theta).all():
            raise ValueError("theta must be a finite feature-dimensional vector")
        if not np.isfinite(training_ridge) or training_ridge <= 0:
            raise ValueError("training_ridge must be finite and positive")
        geometry = SekhariGeometry.build(X) if geometry is None else geometry
        if (geometry.n, geometry.d) != X.shape:
            raise ValueError("geometry has different data dimensions")
        fingerprint = hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest()
        if geometry.feature_sha256 != fingerprint:
            raise ValueError("geometry was computed from a different feature matrix")
        theta.flags.writeable = False
        p = expit(X @ theta)
        hessian = X.T @ ((p * (1 - p))[:, None] * X)
        gradient_norm = float(np.linalg.norm(X.T @ (p - y) / len(X) + training_ridge * theta))
        return cls(X, y, theta, training_ridge, geometry, hessian, gradient_norm)

    def newton_step(self, indices: Indices) -> Array:
        """Exact Woodbury update; factor one full d-by-d base per deletion k."""
        indices = np.asarray(indices)
        n, d = self.X.shape
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not 1 <= len(indices) < n
                or np.any(indices < 0) or np.any(indices >= n)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("deletion indices must be unique valid integers, 1 <= k < n")
        k = len(indices)
        if k not in self.factors:
            base = self.data_hessian_sum.copy()
            base.flat[::d + 1] += (n - k) * self.training_ridge
            self.factors[k] = cho_factor(base, lower=True, check_finite=False)
        deleted = self.X[indices]
        p = expit(deleted @ self.theta)
        U = deleted.T * np.sqrt(p * (1 - p))[None, :]
        # The retained MEAN objective gives +k*lambda*theta in the sum RHS.
        rhs = deleted.T @ (p - self.y[indices]) + k * self.training_ridge * self.theta
        solved = cho_solve(self.factors[k], np.column_stack((rhs, U)), check_finite=False)
        q, inverse_U = solved[:, 0], solved[:, 1:]
        middle = np.eye(k) - U.T @ inverse_U
        correction = cho_solve(cho_factor(middle, lower=True, check_finite=False),
                               U.T @ q, check_finite=False)
        return self.theta + q + inverse_U @ correction

    def remove(
        self, indices: Indices, epsilon: float, delta: float, rng: np.random.Generator,
    ) -> SekhariResult:
        theta_newton = self.newton_step(indices)
        sensitivity, sigma = noise_scale(
            len(self.X), len(indices), self.training_ridge, self.geometry.M_sum,
            self.geometry.loss_lipschitz, epsilon, delta,
        )
        noise = sigma * rng.standard_normal(self.X.shape[1])
        return SekhariResult(theta_newton + noise, theta_newton, noise, sensitivity, sigma)
