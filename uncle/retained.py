"""UNCLE: augmented, non-A, retained-spectrum certification only.

Read top to bottom: full-model setup -> deletion preparation -> four scalar
checks. Training uses sum(retained CE)/n_original + training_ridge*||theta||²/2.
The certificate uses Q=X_retained.T@X_retained+geometry.tau*I; the exact same
RetainedGeometry object must be used for masking. Neither training ridge nor
the removal count silently changes that precision.

This implements the continuing algorithm, not the old theta1-Hessian Local,
anchored, fixed-S, B1/B2, or raw-prediction certificates. Its inherited numerical
eigenvalue policy is not a rigorous floating-point upper-bound certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve, solve_triangular
from scipy.special import expit

from .curvature import dot_error_upper, hull_coefficients, retained_spectra, upper_nonnegative
from .masking import RetainedGeometry

Array = NDArray[np.float64]
Indices = NDArray[np.intp]
DEFAULT_R_INF = (1.0, 0.1, 0.01, 0.001, 0.0001)


@dataclass(frozen=True)
class Check:
    """A strict scalar condition, including the values needed to audit it."""
    name: str
    lhs: float
    rhs: float

    @property
    def passed(self) -> bool:
        return math.isfinite(self.lhs) and math.isfinite(self.rhs) and self.lhs < self.rhs


@dataclass(frozen=True)
class BranchChecks:
    r_inf: float
    c1: Check
    c2: Check
    c3: Check
    c4: Check
    b2_aug: float
    g_mix_aug: float
    lambda_off: float

    @property
    def accepted(self) -> bool:
        return all(check.passed for check in (self.c1, self.c2, self.c3, self.c4))


@dataclass(frozen=True)
class Certificate:
    accepted: bool
    witness_r_inf: float | None
    radius: float
    geometry: RetainedGeometry
    branches: tuple[BranchChecks, ...]


@dataclass(frozen=True)
class RadiusState:
    """Prepared quantities for one fixed Rinf; independent of R2 and tau."""
    r_inf: float
    s_inf: float
    m: Array
    delta: Array
    g_op: float
    g_mix: float
    g_diag: float
    b2_local: float
    off_l2: float


@dataclass(frozen=True)
class UncleModel:
    """Immutable full-model factors. Input arrays must not be mutated after build."""
    X: Array
    y: Array
    theta: Array
    training_ridge: float
    hessian_chol: Array
    weights: Array
    gram: Array
    prediction_factor: Array
    prediction_diagonal: Array
    logits: Array
    logit_error: Array

    @classmethod
    def build(cls, X: Array, y: Array, theta: Array, training_ridge: float) -> UncleModel:
        """Factor H_full(theta0); cache its full Gram and whitened prediction factor.

        This retains O(n²d) Gram setup/O(n²) storage to make deletion updates
        use cached columns. Unused historical full-spectrum summaries are gone.
        The supplied theta is the numerically fitted full-data optimum; the
        historical zero-stationarity-residual assumption is unchanged.
        """
        X, y, theta = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(theta, dtype=float)
        if X.ndim != 2 or min(X.shape) == 0 or y.shape != (len(X),) or theta.shape != (X.shape[1],):
            raise ValueError("model array shapes do not agree")
        if not all(np.all(np.isfinite(a)) for a in (X, y, theta)) or np.any((y != 0) & (y != 1)):
            raise ValueError("finite model arrays and binary labels required")
        ridge = float(training_ridge)
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("training ridge must be finite and positive")
        n, d = X.shape
        logits = X @ theta
        probabilities = expit(logits)
        weights = probabilities * (1.0 - probabilities)
        hessian = (X.T * weights) @ X / n + ridge * np.eye(d)
        chol, _ = cho_factor(hessian, lower=True, check_finite=False)
        gram = X @ solve(hessian, X.T, assume_a="pos", check_finite=False)
        gram = 0.5 * (gram + gram.T)
        factor = solve_triangular(chol, X.T, lower=True, check_finite=False).T / math.sqrt(n)
        diagonal = np.einsum("ij,ij->i", factor, factor)
        error = dot_error_upper(X, theta)
        if not all(np.all(np.isfinite(a)) for a in (gram, factor, diagonal)):
            raise ValueError("nonfinite full curvature geometry")
        # Read-only buffers prevent accidental mutation of cached mathematics.
        for array in (X, y, theta, chol, weights, gram, factor, diagonal, logits, error):
            array.setflags(write=False)
        return cls(X, y, theta, ridge, chol, weights, gram, factor, diagonal, logits, error)

    def prepare_request(self, deleted_indices: Indices, r_inf_values: Sequence[float] = DEFAULT_R_INF) -> UncleRequest:
        """Compute one Newton candidate and all fixed-Rinf spectral states.

        Retained H(theta0)=H_full-U U.T, U=X_deleted.T sqrt(W_deleted/n).
        Its inverse is H_full^-1+V K^-1 V.T, V=H_full^-1 U,
        K=I-U.T H_full^-1 U. All solves below reuse these factors.
        """
        indices = np.asarray(deleted_indices)
        n, d = self.X.shape
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not 0 < len(indices) < n or len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= n)):
            raise ValueError("deleted indices must be unique integers and leave retained data")
        radii = tuple(float(r) for r in r_inf_values)
        if not radii or len(set(radii)) != len(radii) or any(not math.isfinite(r) or r <= 0 for r in radii):
            raise ValueError("Rinf grid must contain distinct finite positive values")
        indices = indices.astype(np.intp, copy=True)
        retained = np.flatnonzero(~np.isin(np.arange(n), indices))
        removed, Z = self.X[indices], self.X[retained]
        root_weights = np.sqrt(self.weights[indices])
        K = np.eye(len(indices)) - (1.0 / n) * (root_weights[:, None] * self.gram[np.ix_(indices, indices)] * root_weights[None, :])
        K_chol = cho_factor(K, lower=True, check_finite=False)
        U = removed.T * (root_weights[None, :] / np.sqrt(n))
        V = cho_solve((self.hessian_chol, True), U, check_finite=False)

        # Deleted-only gradient: assumes theta0 is the fitted full optimum.
        deleted_gradient = removed.T @ (expit(removed @ self.theta) - self.y[indices]) / n
        step = _retained_solve(self.hessian_chol, V, K_chol, deleted_gradient)
        candidate = self.theta + step
        logits1 = self.X @ candidate
        error1 = dot_error_upper(self.X, candidate)
        gradient = Z.T @ (expit(logits1[retained]) - self.y[retained]) / n + self.training_ridge * candidate
        inverse_gradient = _retained_solve(self.hessian_chol, V, K_chol, gradient)
        b = Z @ inverse_gradient
        displacement = Z @ (candidate - self.theta)
        a2, a_inf = float(np.linalg.norm(displacement)), float(np.linalg.norm(displacement, np.inf))
        b2, b_inf = float(np.linalg.norm(b)), float(np.linalg.norm(b, np.inf))
        v_norm = upper_nonnegative(float(np.linalg.norm(inverse_gradient)))

        # Cached Gram columns avoid a fresh O(ndk) feature-product. B=P P.T+C C.T.
        E = self.gram[:, indices] * (root_weights[None, :] / n)
        # Production uses a second plain Cholesky for this prediction factor.
        prediction_K = np.eye(len(indices)) - (root_weights[:, None] * self.gram[np.ix_(indices, indices)] * root_weights[None, :]) / n
        correction_chol = np.linalg.cholesky(prediction_K)
        C = solve_triangular(correction_chol, E.T, lower=True, check_finite=False).T[retained]
        P = self.prediction_factor[retained]
        diagonal = self.prediction_diagonal[retained] + np.einsum("ij,ij->i", C, C)
        dmax = upper_nonnegative(float(np.max(diagonal)))
        states = []
        for r_inf in radii:
            s_inf = a_inf + r_inf
            m = hull_coefficients(self.logits[retained], self.logit_error[retained], logits1[retained], error1[retained], r_inf)
            delta = np.nextafter(s_inf * m, np.inf)
            if not np.all(np.isfinite(delta)) or np.any(delta <= 0):
                raise ValueError("invalid local curvature weights")
            g_op, g_mix = retained_spectra(P, C, delta)
            g_diag = upper_nonnegative(float(np.max(delta * diagonal)))
            b2_local = upper_nonnegative(float(np.linalg.norm(np.sqrt(delta) * b)))
            off_l2 = upper_nonnegative(math.sqrt(dmax * g_diag))
            m.setflags(write=False)
            delta.setflags(write=False)
            states.append(RadiusState(r_inf, s_inf, m, delta, g_op, g_mix, g_diag, b2_local, off_l2))
        if not all(math.isfinite(v) and v >= 0 for v in (a2, a_inf, b2, b_inf)):
            raise ValueError("nonfinite request quantities")
        for array in (indices, retained, candidate, inverse_gradient, b):
            array.setflags(write=False)
        return UncleRequest(self, indices, retained, candidate, inverse_gradient, b, a2, a_inf, b2, b_inf, v_norm, tuple(states))


def _retained_solve(hessian_chol: Array, V: Array, K_chol: tuple[Array, bool], rhs: Array) -> Array:
    """Apply H_retained(theta0)^-1 to one vector, using the prepared downdate."""
    full = cho_solve((hessian_chol, True), rhs, check_finite=False)
    small = cho_solve(K_chol, V.T @ rhs, check_finite=False)
    return full + V @ small


@dataclass(frozen=True)
class UncleRequest:
    model: UncleModel
    indices: Indices
    retained: Indices
    candidate: Array
    inverse_gradient: Array
    b: Array
    a2: float
    a_inf: float
    b2: float
    b_inf: float
    inverse_gradient_norm: float
    states: tuple[RadiusState, ...]

    def certify(self, radius: float, geometry: RetainedGeometry) -> Certificate:
        """Check every Rinf at one fixed (R2,tau); never OR different tau values.

        C3 bounds the augmented norm. C1/C2/C4 and LambdaOff stay in prediction
        space. Checking matching data/deletions binds this certificate to the
        masking geometry; release must use this same geometry object.
        """
        if geometry.X is not self.model.X or not np.array_equal(geometry.indices, self.indices):
            raise ValueError("certificate and masking must share the same data/deletion geometry")
        radius, tau = float(radius), float(geometry.tau)
        if not math.isfinite(radius) or radius <= 0 or not math.isfinite(tau) or tau <= 0:
            raise ValueError("radius and geometry ridge must be finite and positive")
        n = len(self.model.X)
        b2_aug = upper_nonnegative(math.hypot(self.b2, math.sqrt(tau) * self.inverse_gradient_norm))
        branches = []
        for state in self.states:
            extra = (math.sqrt(tau) / math.sqrt(n) / math.sqrt(self.model.training_ridge)) * math.sqrt(state.g_op)
            g_mix_aug = upper_nonnegative(math.hypot(state.g_mix, extra))
            # Deliberately prediction Gmix here, NOT augmented Gmix.
            off = upper_nonnegative(min(state.off_l2 * ((self.a2 + radius) / state.s_inf), state.g_mix))
            c1 = Check("Gop < 1", state.g_op, 1.0)
            c2 = Check("Gdiag < 1", state.g_diag, 1.0)
            lhs3 = b2_aug + g_mix_aug * state.b2_local / (1.0 - state.g_op) if c1.passed else math.inf
            lhs4 = ((self.b_inf + off * state.b2_local / (1.0 - state.g_op)) / (1.0 - state.g_diag)) if c1.passed and c2.passed else math.inf
            c3 = Check("B2Aug + GmixAug*B2Local/(1-Gop) < R2", lhs3, radius)
            c4 = Check("(Binf+LambdaOff*B2Local/(1-Gop))/(1-Gdiag) < Rinf", lhs4, state.r_inf)
            branches.append(BranchChecks(state.r_inf, c1, c2, c3, c4, b2_aug, g_mix_aug, off))
        witness = next((b.r_inf for b in branches if b.accepted), None)
        return Certificate(witness is not None, witness, radius, geometry, tuple(branches))
