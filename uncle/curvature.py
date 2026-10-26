"""Shared numerical primitives for both UNCLE certificates.

The logistic envelope is analytic, not a sampled maximum. The spectral policy
is the production dense/ARPACK/power policy: these floating-point eigenvalue
estimates are NOT interval-certified upper bounds. That limitation is retained,
not concealed by the outward rounding of scalar expressions.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, localcontext
import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh

Array = NDArray[np.float64]

# Same guarded, upward-rounded global maximum as the production implementation.
with localcontext() as context:
    context.prec = 80
    _maximum = Decimal(1) / (Decimal(6) * Decimal(3).sqrt())
    GLOBAL_THIRD_DERIVATIVE = float(np.nextafter(float(_maximum + Decimal("1e-64")), np.inf))
_peak = math.log(2.0 + math.sqrt(3.0))
_PEAK_LOWER = float(np.nextafter(_peak, -np.inf))
_PEAK_UPPER = float(np.nextafter(_peak, np.inf))


def upper_nonnegative(value: float) -> float:
    """Retain production's one-ULP outward scalar rounding (zero stays zero)."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("nonnegative certificate quantity must be finite")
    return float(np.nextafter(value, np.inf)) if value > 0 else 0.0


def dot_error_upper(X: Array, theta: Array) -> Array:
    """Binary64 dot-product error envelope for each computed logit X @ theta."""
    operations = 2 * X.shape[1]
    product = operations * (np.finfo(np.float64).eps / 2.0)
    if product >= 1:
        raise ValueError("dimension exceeds the dot-product error model")
    gamma = product / (1.0 - product)
    smallest = float(np.nextafter(0.0, 1.0))
    underflow = float(np.nextafter(4.0 * operations * smallest, np.inf))
    scale = np.abs(X) @ np.abs(theta)
    scale_upper = np.nextafter((scale + underflow) / (1.0 - gamma), np.inf)
    error = np.nextafter(gamma * scale_upper + underflow, np.inf)
    if not np.all(np.isfinite(error)):
        raise ValueError("logit error envelope overflowed")
    return error


def _third_derivative_upper(magnitude: Array) -> Array:
    """Outward evaluation of |ell'''(t)|, with stable near-zero and tail forms."""
    result = np.empty_like(magnitude)
    near = magnitude <= 1.0
    tangent = np.tanh(0.5 * magnitude[near])
    result[near] = 0.25 * tangent * (1.0 - tangent * tangent)
    exponential = np.exp(-magnitude[~near])
    result[~near] = exponential * (1.0 - exponential) / (1.0 + exponential) ** 3
    smallest = float(np.nextafter(0.0, 1.0))
    guard = 128.0 * np.finfo(np.float64).eps * np.maximum(result, smallest) + 128.0 * smallest
    return np.maximum(np.nextafter(result + guard, np.inf), smallest)


def interval_third_derivative_upper(lower: Array, upper: Array) -> Array:
    """Bound |ell'''| on each [lower,upper] using endpoints and its known peak.

    The absolute derivative increases up to |t|=log(2+sqrt(3)), then decreases.
    Intervals containing that point use the global maximum 1/(6 sqrt(3)).
    Endpoints here already include the uncertainty in computed logits.
    """
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.ndim != 1 or lo.shape != hi.shape or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo > hi):
        raise ValueError("invalid local-curvature interval")
    lo, hi = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
    nearest = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    farthest = np.maximum(np.abs(lo), np.abs(hi))
    increasing, decreasing = farthest < _PEAK_LOWER, nearest > _PEAK_UPPER
    result = np.full(lo.shape, GLOBAL_THIRD_DERIVATIVE)
    result[increasing] = _third_derivative_upper(farthest[increasing])
    result[decreasing] = _third_derivative_upper(nearest[decreasing])
    return np.nextafter(result, np.inf)


def hull_coefficients(a0: Array, error0: Array, a1: Array, error1: Array, r_inf: float) -> Array:
    """Non-A interval: convex hull of a0 and [a1-Rinf,a1+Rinf]."""
    lo0, hi0 = np.nextafter(a0 - error0, -np.inf), np.nextafter(a0 + error0, np.inf)
    lo1 = np.nextafter(np.nextafter(a1 - error1, -np.inf) - r_inf, -np.inf)
    hi1 = np.nextafter(np.nextafter(a1 + error1, np.inf) + r_inf, np.inf)
    return interval_third_derivative_upper(np.minimum(lo0, lo1), np.maximum(hi0, hi1))


def fixed_coefficients(a0: Array, error0: Array, s_inf: float) -> Array:
    """A: bound curvature on [a0-s, a0+s], before any deletion is known.

    This is the production fixed-S endpoint construction. No candidate logits
    or deletion-specific spectral quantities enter the local envelope.
    """
    magnitude = np.abs(a0)
    enlarged = np.nextafter(error0 + s_inf, np.inf)
    lower = magnitude - enlarged
    lower = np.where(lower > 0, np.nextafter(lower, -np.inf), 0.0)
    upper = np.nextafter(magnitude + enlarged, np.inf)
    if not np.all(np.isfinite(upper)):
        raise ValueError("local-curvature interval overflowed")
    increasing, decreasing = upper < _PEAK_LOWER, lower > _PEAK_UPPER
    result = np.full(a0.shape, GLOBAL_THIRD_DERIVATIVE)
    result[increasing] = _third_derivative_upper(upper[increasing])
    result[decreasing] = _third_derivative_upper(lower[decreasing])
    return np.nextafter(result, np.inf)


def largest_psd_eigenvalue(size: int, matvec: Callable[[Array], Array]) -> float:
    """Preserve production policy exactly; matvec is only a linear operator.

    Dense eigvalsh for size<500; otherwise deterministic ARPACK (tol=1e-10).
    On ARPACK failure, retain its 500-step power/Rayleigh fallback. The uniform
    start and fallback can miss a dominant direction: see the module caveat.
    """
    if size == 0:
        return 0.0
    if size < 500:
        eye = np.eye(size, dtype=np.float64)
        matrix = np.column_stack([np.asarray(matvec(eye[:, j]), dtype=np.float64) for j in range(size)])
        return float(eigvalsh(matrix, subset_by_index=[size - 1, size - 1])[0])
    operator = LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=np.dtype(np.float64))
    start = np.full(size, 1.0 / math.sqrt(size), dtype=np.float64)
    try:
        return float(eigsh(operator, k=1, which="LA", v0=start, return_eigenvectors=False, tol=1e-10)[0])
    except (ArpackError, np.linalg.LinAlgError):
        vector, value = start, 0.0
        for _ in range(500):
            product = np.asarray(operator @ vector, dtype=np.float64)
            next_value = float(np.linalg.norm(product))
            if next_value < 1e-30:
                return 0.0
            vector = product / next_value
            if abs(next_value - value) <= 1e-10 * max(next_value, 1.0):
                break
            value = next_value
        return float(vector @ (operator @ vector))


def retained_spectra(P: Array, C: Array, delta: Array) -> tuple[float, float]:
    """Literal Gop=||DBD|| and Gmix=||BD|| for B=P P.T+C C.T.

    The second PSD operator is D B B D, whose largest eigenvalue is Gmix^2.
    Only operator-vector products are needed above the dense-size threshold.
    """
    root = np.sqrt(delta)

    def gram(vector: Array) -> Array:
        return P @ (P.T @ vector) + C @ (C.T @ vector)

    op = largest_psd_eigenvalue(len(delta), lambda v: root * gram(root * v))
    mix_squared = largest_psd_eigenvalue(len(delta), lambda v: root * gram(gram(root * v)))
    return upper_nonnegative(op), upper_nonnegative(math.sqrt(upper_nonnegative(mix_squared)))


def full_weighted_norm(P: Array, delta: Array) -> float:
    """Initialize a_s=||D_s B_full D_s||; never called during a query.

    B_full=P P.T. Keep the inherited sample-space eigensolver policy, rather
    than switching to a feature-space eigenproblem with different numerics.
    """
    root = np.sqrt(delta)
    return upper_nonnegative(largest_psd_eigenvalue(
        len(delta), lambda v: root * (P @ (P.T @ (root * v)))))
