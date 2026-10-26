"""UNCLE A+B2, with the SAME augmented geometry for checks and noise.

Read top to bottom: setup before deletions -> small Woodbury query -> four
checks. A fixes S_inf around theta0. B2 inflates cached full-data quantities
using a k-dimensional eigenvalue. There is no retained-spectrum query path.
The inherited floating-point eigenvalue policy is not interval-certified.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve, solve_triangular
from scipy.special import expit

from .curvature import (dot_error_upper, fixed_coefficients, full_weighted_norm,
                        largest_psd_eigenvalue, upper_nonnegative)
from .masking import RetainedGeometry

Array = NDArray[np.float64]
Indices = NDArray[np.intp]
DEFAULT_S_INF = (10.0, 1.0, 0.1, 0.01, 0.001)
CERTIFICATE_VARIANT = "theta0-A-B2-augmented-v1"


@dataclass(frozen=True)
class Check:
    """One strict scalar condition, with both sides available for auditing."""
    name: str
    lhs: float
    rhs: float

    @property
    def passed(self) -> bool:
        return math.isfinite(self.lhs) and math.isfinite(self.rhs) and self.lhs < self.rhs


@dataclass(frozen=True)
class BranchChecks:
    s_inf: float
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
    witness_s_inf: float | None
    witness_r_inf: float | None
    radius: float
    geometry: RetainedGeometry
    branches: tuple[BranchChecks, ...]


@dataclass(frozen=True)
class FixedSummary:
    """Full-data quantities initialized once for one committed S_inf."""
    s_inf: float
    m: Array
    delta: Array
    a_s: float
    g_s: float


@dataclass(frozen=True)
class RadiusState:
    """Cheap query-specific B2 bounds, independent of R2 and tau."""
    s_inf: float
    r_inf: float
    m: Array
    delta: Array
    g_op: float
    g_mix: float
    g_diag: float
    b2_local: float
    off_l2: float


def _s_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(s) for s in values)
    if not grid or len(set(grid)) != len(grid) or any(not math.isfinite(s) or s <= 0 for s in grid):
        raise ValueError("Sinf grid must contain distinct finite positive values")
    return grid


@dataclass(frozen=True)
class UncleModel:
    """Read-only full model. All large spectral work belongs to build()."""
    X: Array
    y: Array
    theta: Array
    training_ridge: float
    hessian_chol: Array
    weights: Array
    gram: Array
    k0: float
    d0: float
    summaries: tuple[FixedSummary, ...]

    @classmethod
    def build(cls, X: Array, y: Array, theta: Array, training_ridge: float,
              s_inf_values: Sequence[float] = DEFAULT_S_INF) -> UncleModel:
        """Precompute A's local weights and B2's full-data spectral bounds.

        H_full uses original-n mean loss. G=X H_full^-1 X.T is kept to match
        the existing initialization/numerical policy; B_full=G/n. Thus setup
        still includes O(n^2 d) work/O(n^2) storage. No such matrix is rebuilt
        at theta1 or per request. The numerical optimum assumption is unchanged.
        """
        grid = _s_grid(s_inf_values)
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
        P = solve_triangular(chol, X.T, lower=True, check_finite=False).T / math.sqrt(n)
        diagonal = np.einsum("ij,ij->i", P, P)
        if not all(np.all(np.isfinite(a)) for a in (gram, P, diagonal)):
            raise ValueError("nonfinite full curvature geometry")
        k0 = upper_nonnegative(largest_psd_eigenvalue(n, lambda v: gram @ v) / n)
        d0 = upper_nonnegative(float(np.max(diagonal)))
        error = dot_error_upper(X, theta)
        summaries = []
        for s in grid:
            m = fixed_coefficients(logits, error, s)
            delta = np.nextafter(s * m, np.inf)
            if not np.all(np.isfinite(delta)) or np.any(delta <= 0):
                raise ValueError("invalid local curvature weights")
            a_s = full_weighted_norm(P, delta)
            g_s = upper_nonnegative(float(np.max(delta * diagonal)))
            m.setflags(write=False)
            delta.setflags(write=False)
            summaries.append(FixedSummary(s, m, delta, a_s, g_s))
        for array in (X, y, theta, chol, weights, gram):
            array.setflags(write=False)
        return cls(X, y, theta, ridge, chol, weights, gram, k0, d0, tuple(summaries))

    def prepare_request(self, deleted_indices: Indices) -> UncleRequest:
        """One Newton step and residual solve, then O(Qn) scalar/weighted norms.

        H_R=H_full-U U.T; V=H_full^-1 U; K=I-U.T V.
        H_R^-1=H_full^-1+V K^-1 V.T. B2 uses eta=1/(1-lambda_max(I-K)).
        Its only query eigenproblem is k by k. Fixed-S weights and full-data
        spectra already exist; there is no prediction-space low-rank factor.
        """
        indices = np.asarray(deleted_indices)
        n, _ = self.X.shape
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not 0 < len(indices) < n or len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= n)):
            raise ValueError("deleted indices must be unique integers and leave retained data")
        indices = indices.astype(np.intp, copy=True)
        retained = np.flatnonzero(~np.isin(np.arange(n), indices))
        removed, Z = self.X[indices], self.X[retained]
        root_weights = np.sqrt(self.weights[indices])
        small_gram = (root_weights[:, None] * self.gram[np.ix_(indices, indices)] * root_weights[None, :]) / n
        K = np.eye(len(indices)) - small_gram
        K_chol = cho_factor(K, lower=True, check_finite=False)
        q_matrix = np.eye(len(indices)) - K
        q = upper_nonnegative(largest_psd_eigenvalue(len(indices), lambda v: q_matrix @ v))
        if q >= 1:
            raise ValueError("nonpositive B2 inflation denominator")
        eta = upper_nonnegative(1.0 / (1.0 - q))
        U = removed.T * (root_weights[None, :] / math.sqrt(n))
        V = cho_solve((self.hessian_chol, True), U, check_finite=False)

        deleted_gradient = removed.T @ (expit(removed @ self.theta) - self.y[indices]) / n
        candidate = self.theta + _retained_solve(self.hessian_chol, V, K_chol, deleted_gradient)
        gradient = Z.T @ (expit(Z @ candidate) - self.y[retained]) / n + self.training_ridge * candidate
        inverse_gradient = _retained_solve(self.hessian_chol, V, K_chol, gradient)
        b = Z @ inverse_gradient
        displacement = Z @ (candidate - self.theta)
        a2, a_inf = float(np.linalg.norm(displacement)), float(np.linalg.norm(displacement, np.inf))
        b2, b_inf = float(np.linalg.norm(b)), float(np.linalg.norm(b, np.inf))
        v_norm = upper_nonnegative(float(np.linalg.norm(inverse_gradient)))
        if not all(math.isfinite(v) and v >= 0 for v in (a2, a_inf, b2, b_inf)):
            raise ValueError("nonfinite request quantities")
        states = []
        for summary in self.summaries:
            s, r = summary.s_inf, summary.s_inf - a_inf
            if r <= 0:
                # Keep the failed grid slot in the diagnostics and the OR.
                empty = np.empty(0)
                empty.setflags(write=False)
                states.append(RadiusState(s, r, empty, empty, math.inf, math.inf,
                                          math.inf, math.inf, math.inf))
                continue
            m, delta = summary.m[retained], summary.delta[retained]
            g_op = upper_nonnegative(eta * summary.a_s)
            g_diag = upper_nonnegative(eta * summary.g_s)
            dmax = upper_nonnegative(eta * self.d0)
            # Loewner domination does NOT justify eta*||B_full D||.
            g_mix = upper_nonnegative(eta * math.sqrt(self.k0 * summary.a_s))
            b2_local = upper_nonnegative(float(np.linalg.norm(np.sqrt(delta) * b)))
            off_l2 = upper_nonnegative(math.sqrt(dmax * g_diag))
            m.setflags(write=False)
            delta.setflags(write=False)
            states.append(RadiusState(s, r, m, delta, g_op, g_mix, g_diag, b2_local, off_l2))
        for array in (indices, retained, candidate, inverse_gradient, b):
            array.setflags(write=False)
        return UncleRequest(self, indices, retained, candidate, inverse_gradient, b,
                            a2, a_inf, b2, b_inf, v_norm, q, eta, tuple(states))


def _retained_solve(hessian_chol: Array, V: Array, K_chol: tuple[Array, bool], rhs: Array) -> Array:
    """Apply the retained theta0 Hessian inverse, without forming that inverse."""
    full = cho_solve((hessian_chol, True), rhs, check_finite=False)
    return full + V @ cho_solve(K_chol, V.T @ rhs, check_finite=False)


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
    q: float
    eta: float
    states: tuple[RadiusState, ...]

    def certify(self, radius: float, geometry: RetainedGeometry) -> Certificate:
        """Four strict checks per committed S; OR complete branches, never tau.

        C3 uses the augmented norm. C1/C2/C4 and LambdaOff remain prediction-
        space bounds. The same geometry object must subsequently draw noise.
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
            valid = state.r_inf > 0
            if valid:
                extra = (math.sqrt(tau) / math.sqrt(n) / math.sqrt(self.model.training_ridge)) * math.sqrt(state.g_op)
                g_mix_aug = upper_nonnegative(math.hypot(state.g_mix, extra))
                off = upper_nonnegative(min(state.off_l2 * ((self.a2 + radius) / state.s_inf), state.g_mix))
            else:
                g_mix_aug = off = math.inf
            c1 = Check("Gop < 1", state.g_op if valid else math.inf, 1.0)
            c2 = Check("Gdiag < 1", state.g_diag if valid else math.inf, 1.0)
            lhs3 = b2_aug + g_mix_aug * state.b2_local / (1.0 - state.g_op) if c1.passed else math.inf
            lhs4 = ((self.b_inf + off * state.b2_local / (1.0 - state.g_op)) / (1.0 - state.g_diag)) if c1.passed and c2.passed else math.inf
            c3 = Check("B2Aug + GmixAug*B2Local/(1-Gop) < R2", lhs3, radius)
            c4 = Check("(Binf+LambdaOff*B2Local/(1-Gop))/(1-Gdiag) < Sinf-Ainf", lhs4, state.r_inf)
            branches.append(BranchChecks(state.s_inf, state.r_inf, c1, c2, c3, c4, b2_aug, g_mix_aug, off))
        witness = next((branch for branch in branches if branch.accepted), None)
        return Certificate(witness is not None, None if witness is None else witness.s_inf,
                           None if witness is None else witness.r_inf, radius, geometry, tuple(branches))
