"""Guo et al.: perturbed training, a Woodbury Newton step, and retraining.

The v4/v5 distinction is a choice of (training_ridge, noise_std) search points,
not a different removal algorithm. All coordinates, including a supplied bias
column, are regularized. Independent requests start from the same fitted model.

The full and retained objectives both use the ORIGINAL sample count n:
    sum_ret CE / n + training_ridge * ||theta||^2 / 2 + b.T @ theta / n.
Noise is objective perturbation b, sampled at training time. Guo does not add
UNCLE-style output noise after the Newton step or fallback retrain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit

Array = NDArray[np.float64]
Indices = NDArray[np.int64]


def _indices(values: NDArray, n: int) -> Indices:
    """Reject ambiguous requests rather than silently rounding/deduplicating."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("deletion indices must be a one-dimensional integer array")
    result = raw.astype(np.int64, copy=False)
    if (len(result) >= n or np.any(result < 0) or np.any(result >= n)
            or len(np.unique(result)) != len(result)):
        raise ValueError("indices must be unique, in range, and leave a retained row")
    return result


def _fit_perturbed(
    X: Array, y: Array, ridge: float, b: Array, n: int, start: Array,
) -> Array:
    """Keep the production Guo optimizer and tolerances, including default ftol."""
    def loss(theta: Array) -> float:
        logits = X @ theta
        return float(np.sum(np.logaddexp(0, logits) - y * logits) / n
                     + 0.5 * ridge * np.dot(theta, theta) + np.dot(b, theta) / n)

    def gradient(theta: Array) -> Array:
        return (X.T @ (expit(X @ theta) - y) + b) / n + ridge * theta

    result = minimize(loss, start, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": 5000, "gtol": 1e-10})
    return np.asarray(result.x, dtype=np.float64)


@dataclass
class GuoGeometry:
    """Data-only request spectra reusable across models and hyperparameters.

    Creating this cache costs O(nd^2). A new request costs O(kd^2 + 20d^2).
    The 20-step deterministic power estimate is the existing numerical policy;
    it is not an interval-certified upper eigenvalue bound. Reusing a request
    spectrum in experiments does not make it available before an unseen query.
    """

    X: Array
    full_gram: Array

    @classmethod
    def build(cls, X: Array) -> GuoGeometry:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix")
        return cls(X, X.T @ X)

    def spectral_norm(self, indices: Indices) -> float:
        """Return the historical estimate of sqrt(||X_ret.T @ X_ret||_op)."""
        indices = _indices(indices, len(self.X))
        deleted = self.X[indices]
        gram = self.full_gram - deleted.T @ deleted if len(indices) else self.full_gram
        vector = np.random.default_rng(0).standard_normal(self.X.shape[1])
        vector /= np.linalg.norm(vector)
        eigenvalue = 0.0
        for _ in range(20):
            product = gram @ vector
            eigenvalue = float(np.linalg.norm(product))
            if eigenvalue < 1e-15:
                break
            vector = product / eigenvalue
        return float(np.sqrt(max(0.0, eigenvalue)))


@dataclass(frozen=True)
class GuoResult:
    theta: Array
    theta_newton: Array
    accepted: bool
    beta: float
    threshold: float


@dataclass(frozen=True)
class GuoRequest:
    """A prepared independent request; release retrains only on rejection."""

    model: GuoModel
    indices: Indices
    theta_newton: Array
    beta: float
    prediction_norm: float
    spectral_norm: float

    def certify(self, epsilon: float, delta: float) -> tuple[bool, float, float]:
        if not np.isfinite(epsilon) or epsilon <= 0 or not 0 < delta < 1:
            raise ValueError("require finite epsilon > 0 and 0 < delta < 1")
        threshold = self.model.noise_std * epsilon / math.sqrt(2 * math.log(1.5 / delta))
        return bool(self.beta <= threshold), self.beta, threshold

    def release(self, epsilon: float, delta: float) -> GuoResult:
        accepted, beta, threshold = self.certify(epsilon, delta)
        theta = self.theta_newton
        if not accepted:
            retained = np.ones(len(self.model.X), dtype=bool)
            retained[self.indices] = False
            # Reuse the SAME objective perturbation. The Newton point is the
            # production warm start, not a second independently trained model.
            theta = _fit_perturbed(
                self.model.X[retained], self.model.y[retained],
                self.model.training_ridge, self.model.perturbation,
                len(self.model.X), self.theta_newton,
            )
        return GuoResult(theta, self.theta_newton, accepted, beta, threshold)


@dataclass
class GuoModel:
    X: Array
    y: Array
    theta: Array
    perturbation: Array
    training_ridge: float
    noise_std: float
    probabilities: Array
    logistic_weights: Array
    hessian_factor: tuple[Array, bool]
    geometry: GuoGeometry | None = None

    @classmethod
    def build(
        cls, X: Array, y: Array, training_ridge: float, noise_std: float,
        seed: int, warm_start: Array | None = None,
        geometry: GuoGeometry | None = None,
    ) -> GuoModel:
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix")
        if y.shape != (len(X),) or not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must be a binary vector with one label per row")
        if (not np.isfinite(training_ridge) or training_ridge <= 0
                or not np.isfinite(noise_std) or noise_std < 0):
            raise ValueError("training_ridge must be positive and noise_std nonnegative")
        if geometry is not None and not np.array_equal(geometry.X, X):
            raise ValueError("geometry was computed from a different feature matrix")
        start = np.zeros(X.shape[1]) if warm_start is None else np.asarray(warm_start)
        if start.shape != (X.shape[1],) or not np.isfinite(start).all():
            raise ValueError("warm_start must be a finite vector of feature dimension")
        b = np.random.default_rng(seed).normal(0, noise_std, size=X.shape[1])
        theta = _fit_perturbed(X, y, training_ridge, b, len(X), start)
        p = expit(X @ theta)
        weights = p * (1 - p)
        hessian = (X.T * weights) @ X / len(X)
        hessian.flat[::X.shape[1] + 1] += training_ridge
        factor = cho_factor(hessian, lower=True, check_finite=False)
        return cls(X, y, theta, b, training_ridge, noise_std, p, weights, factor, geometry)

    def prepare_request(
        self, indices: Indices, spectral_norm: float | None = None,
    ) -> GuoRequest:
        """One unseen query; optional spectrum must belong to this exact request."""
        indices = _indices(indices, len(self.X))
        solved_rows = cho_solve(self.hessian_factor, self.X[indices].T, check_finite=False)
        return self._prepare(indices, solved_rows, spectral_norm)

    def prepare_requests(
        self, prefixes: Indices, removal_sizes: list[int],
        spectral_norms: Array | None = None,
    ) -> list[list[GuoRequest]]:
        """Experiment reuse: one wide solve for all distinct deleted rows.

        Return [k index][request index]. Prefix rows describe independent
        requests, not sequential removals. Optional spectra have the same shape.
        """
        prefixes = np.asarray(prefixes)
        if prefixes.ndim != 2 or len(prefixes) == 0:
            raise ValueError("prefixes must contain at least one request row")
        for prefix in prefixes:
            _indices(prefix, len(self.X))
        if (not removal_sizes or any(k < 0 or k > prefixes.shape[1] for k in removal_sizes)
                or len(set(removal_sizes)) != len(removal_sizes)):
            raise ValueError("removal sizes must be unique valid prefix lengths")
        if spectral_norms is not None and np.shape(spectral_norms) != (len(removal_sizes), len(prefixes)):
            raise ValueError("spectral_norms must have shape (number of k, requests)")
        unique = np.unique(prefixes[:, :max(removal_sizes)])
        solved = cho_solve(self.hessian_factor, self.X[unique].T, check_finite=False)
        result = []
        for ki, k in enumerate(removal_sizes):
            requests = []
            for ri, prefix in enumerate(prefixes):
                indices = prefix[:k]
                norm = None if spectral_norms is None else float(spectral_norms[ki, ri])
                requests.append(self._prepare(indices, solved[:, np.searchsorted(unique, indices)], norm))
            result.append(requests)
        return result

    def _prepare(
        self, indices: Indices, inverse_rows: Array, spectral_norm: float | None,
    ) -> GuoRequest:
        n = len(self.X)
        coefficients = (self.probabilities[indices] - self.y[indices]) / n
        update = inverse_rows @ coefficients
        if len(indices):
            sqrt_weights = np.sqrt(self.logistic_weights[indices] / n)
            U = self.X[indices].T * sqrt_weights
            inverse_U = inverse_rows * sqrt_weights
            middle = np.eye(len(indices)) - U.T @ inverse_U
            rhs = U.T @ update
            try:
                correction = cho_solve(cho_factor(middle, lower=True, check_finite=False),
                                       rhs, check_finite=False)
            except np.linalg.LinAlgError:
                correction = np.linalg.solve(middle, rhs)
            update = update + inverse_U @ correction
        candidate = self.theta + update
        displacement = candidate - self.theta
        prediction = self.X @ displacement
        prediction_sq = float(prediction @ prediction - prediction[indices] @ prediction[indices])
        prediction_norm = float(np.sqrt(max(0.0, prediction_sq)))
        if spectral_norm is None:
            if self.geometry is None:
                self.geometry = GuoGeometry.build(self.X)
            spectral_norm = self.geometry.spectral_norm(indices)
        if not np.isfinite(spectral_norm) or spectral_norm < 0:
            raise ValueError("spectral_norm must be finite and nonnegative")
        beta = 0.25 * spectral_norm * float(np.linalg.norm(displacement)) * prediction_norm
        return GuoRequest(self, indices.copy(), candidate, beta, prediction_norm, spectral_norm)
