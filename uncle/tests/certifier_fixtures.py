"""Small logistic helpers used only by the certifier tests.

Copied from the removed training wrappers so the certificate tests can
build a full-data optimum and a one-step Newton candidate on their own.
"""

from typing import Optional

import numpy as np
from scipy.linalg import solve as la_solve
from scipy.optimize import minimize
from scipy.special import expit

# Tight universal bound on |γ_i(θ)| = |p(1-p)(1-2p)| for logistic regression.
LOGISTIC_GAMMA_MAX = 1.0 / (6.0 * np.sqrt(3.0))


def logistic_loss(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> float:
    """Regularised logistic regression loss."""
    z = X @ theta
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return loss + 0.5 * lam * np.dot(theta, theta)


def logistic_gradient(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Gradient of regularised logistic loss. Returns shape (d,)."""
    n = X.shape[0]
    p = expit(X @ theta)
    grad = (X.T @ (p - y)) / n
    if lam > 0:
        grad = grad + lam * theta
    return grad


def logistic_hessian(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Hessian of regularised logistic loss. Returns shape (d, d)."""
    n = X.shape[0]
    p = expit(X @ theta)
    w = p * (1 - p)
    H = (X.T * w) @ X / n
    if lam > 0:
        H = H + lam * np.eye(X.shape[1])
    return H


def _per_sample_gradients(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample gradients (unregularised). Returns shape (n, d)."""
    p = expit(X @ theta)
    residuals = p - y
    return residuals[:, None] * X


def remove_samples(
    X: np.ndarray, y: np.ndarray, T: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples indexed by T from a dataset."""
    n = X.shape[0]
    for idx in T:
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {n}")
    mask = np.ones(n, dtype=bool)
    mask[list(T)] = False
    return X[mask], y[mask]


def newton_step_hypothesis(
    theta_hat: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    T: set[int],
    lam: float,
    n_steps: int = 1,
    *,
    X_ret: Optional[np.ndarray] = None,
    y_ret: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One-step Newton hypothesis on the retained loss."""
    del X_ret, y_ret
    if n_steps != 1:
        raise ValueError(f"Only n_steps=1 is supported, got {n_steps}")

    n = X.shape[0]
    T_sorted = sorted(T)
    H_full = logistic_hessian(theta_hat, X, y, lam)
    X_T = X[T_sorted]
    p_T = expit(X_T @ theta_hat)
    w_T = p_T * (1.0 - p_T)
    H_ret = H_full - (X_T.T * w_T) @ X_T / n
    g_T = _per_sample_gradients(theta_hat, X, y)
    grad_sum = g_T[T_sorted].sum(axis=0) / n
    try:
        delta = la_solve(H_ret, grad_sum, assume_a="pos")
    except np.linalg.LinAlgError:
        delta = la_solve(H_ret, grad_sum, assume_a="sym")
    return theta_hat + delta


def retrain_exact(
    X_ret: np.ndarray,
    y_ret: np.ndarray,
    lam: float,
    d: int,
    n: Optional[int] = None,
) -> np.ndarray:
    """Exact ERM on retained data via L-BFGS-B.

    The objective is averaged over the original ``n`` when ``n`` is given.
    """
    n_ret = X_ret.shape[0]
    lam_eff = lam if n is None else lam * n / n_ret

    def _loss(weights: np.ndarray) -> float:
        return logistic_loss(weights, X_ret, y_ret, lam_eff)

    def _grad(weights: np.ndarray) -> np.ndarray:
        return logistic_gradient(weights, X_ret, y_ret, lam_eff)

    result = minimize(
        fun=_loss,
        x0=np.zeros(d),
        jac=_grad,
        method="L-BFGS-B",
        options={"maxiter": 50000, "gtol": 1e-12},
    )
    grad_norm = np.linalg.norm(_grad(result.x))
    if not result.success and grad_norm > 1e-6:
        raise RuntimeError(
            f"Retraining failed: {result.message} (grad_norm={grad_norm:.2e})"
        )
    return result.x


def default_R2(k: int, d: int, n: int, safety_factor: float = 1.0) -> float:
    """Default R_2 = safety_factor * k * d / n**1.5."""
    return safety_factor * k * d / n**1.5
