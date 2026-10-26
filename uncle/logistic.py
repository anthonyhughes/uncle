"""Binary logistic regression; every coordinate, including a bias, is penalized.

Samples are rows. ``normalization_n`` is explicit because UNCLE/Guo keep the
original n after deletion, whereas Sekhari uses the retained sample count.
Reported test loss is always cross-entropy alone, never the ridge penalty.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import expit

Array = NDArray[np.float64]


def loss_gradient(theta: Array, X: Array, y: Array, ridge: float,
                  normalization_n: int | None = None,
                  linear: Array | None = None) -> tuple[float, Array]:
    """Return sum(CE)/N + ridge*||theta||²/2 + linear·theta/N and its gradient."""
    n = len(X) if normalization_n is None else normalization_n
    logits = X @ theta
    loss = float(np.sum(np.logaddexp(0, logits) - y * logits) / n
                 + 0.5 * ridge * np.dot(theta, theta))
    gradient = X.T @ (expit(logits) - y) / n + ridge * theta
    if linear is not None:
        loss += float(np.dot(linear, theta) / n)
        gradient = gradient + linear / n
    return loss, gradient


def fit(X: Array, y: Array, ridge: float, normalization_n: int | None = None,
        linear: Array | None = None, start: Array | None = None,
        *, polish: bool = False) -> Array:
    """Existing UNCLE L-BFGS fit; two Newton polish steps only for full models.

    For a retained fit we minimize the positive rescaling N/n_ret of the
    objective, exactly as the old retrain_exact routine does. ``polish=False``
    preserves its fallback optimizer. This is numerical ERM, not an assertion
    that a finite tolerance proves an exact optimum.
    """
    if not np.isfinite(ridge) or ridge <= 0 or len(X) == 0:
        raise ValueError("fit requires positive ridge and nonempty data")
    n = len(X) if normalization_n is None else normalization_n
    if n < len(X):
        raise ValueError("normalization cannot be smaller than the retained count")
    effective_ridge = ridge * n / len(X)
    initial = np.zeros(X.shape[1]) if start is None else np.array(start, copy=True)
    result = minimize(loss_gradient, initial, args=(X, y, effective_ridge, None, linear),
                      jac=True, method="L-BFGS-B", options={"maxiter": 50000, "gtol": 1e-12})
    theta = result.x
    residual = np.linalg.norm(loss_gradient(theta, X, y, effective_ridge, linear=linear)[1])
    if not result.success and residual > 1e-6:
        raise RuntimeError(f"L-BFGS failed: {result.message}; gradient={residual:g}")
    if polish:
        for _ in range(2):
            p = expit(X @ theta)
            hessian = (X.T * (p * (1 - p))) @ X / len(X)
            hessian += effective_ridge * np.eye(X.shape[1])
            gradient = loss_gradient(theta, X, y, effective_ridge, linear=linear)[1]
            theta = theta - solve(hessian, gradient, assume_a="pos")
    if not np.isfinite(theta).all():
        raise ValueError("nonfinite fitted model")
    return theta


def metrics(theta: Array, X: Array, y: Array) -> tuple[float, float]:
    """Unregularized test CE and accuracy in [0,1]."""
    logits = X @ theta
    ce = float(np.mean(np.logaddexp(0, logits) - y * logits))
    accuracy = float(np.mean((logits > 0.0) == (y > 0.5)))
    if not np.isfinite([ce, accuracy]).all():
        raise ValueError("nonfinite test utility")
    return ce, accuracy
