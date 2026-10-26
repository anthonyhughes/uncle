import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .cached_curvature import CachedCurvature


def calc_woodbury_constants(
    cache: CachedCurvature,
    T: np.ndarray,
    X_T: np.ndarray | None = None,
) -> dict:
    """Per-query Woodbury blocks for the exact retained Hessian inverse.

    The retained Hessian is the exact rank-k downdate of the cached mean-form
    full-data Hessian:

        H_{R,0} = H_full − (1/n) X_Tᵀ W_T X_T,

    and ``_apply_H_T_inv`` applies its inverse via

        H_{R,0}⁻¹ = H⁻¹ + H⁻¹ U_T K_T⁻¹ U_Tᵀ H⁻¹,
        U_T = (1/√n) X_Tᵀ W_T^{1/2},
        K_T = I_k − (1/n) W_T^{1/2} G[T,T] W_T^{1/2}.

    The small-system factor is retained so every inverse application reuses
    the same factorisation. ``K_T_inv`` remains available for the Gram-bound
    calculations and compatibility with the diagnostic helpers.

    Returned blocks:
        w_T_sqrt : W_T^{1/2}, shape (k,)
        K_T_chol : triangular factorisation of K_T
        K_T_inv  : K_T⁻¹,     shape (k, k)
        P_T      : H_full⁻¹ U_T, shape (d, k), when ``X_T`` is supplied

    Cost: O(k²) for the G[T,T] slice + O(k³) for K_T⁻¹ — independent of n.
    """
    n = cache.n
    k = T.size

    # guards to check
    if k == 0:
        raise ValueError("FullPass verifier requires |T| ≥ 1")

    # Dim checks
    if T.ndim != 1 or T.dtype.kind not in "iu":
        raise ValueError("indices_to_remove must be a 1-D integer array")
    
    if T.min() < 0 or T.max() >= n:
        raise ValueError(f"indices_to_remove out of range [0, {n})")

    # the original cache's full-data weights, indexed to the deleted set T
    w_T_sqrt = np.sqrt(cache.w_full[T])  # (k,)

    # K_T = I − (1/n) W_T^{1/2} G[T,T] W_T^{1/2}  
    # a normlaisation factor
    inv_n = 1.0 / n

    # we slice the cached gram matrix here to get an exact rank-k downdate of the full-data Hessian
    # which is needed for the exact retained Gram G⁰ and thus the tightest certificate. 
    G_TT = cache.G[np.ix_(T, T)]  # (k, k)

    # Start from the identity and subtract off how much the removed points collectively distort the curvature
    # to get an exact rank-k downdate of the full-data Hessian
    K_T = np.eye(k) - inv_n * (w_T_sqrt[:, None] * G_TT * w_T_sqrt[None, :])

    # K_T is SPD for |T| ≪ n; Cholesky inverse, general-solve fallback.
    try:
        L, lower = cho_factor(K_T, lower=True, check_finite=False)
        K_T_inv = cho_solve((L, lower), np.eye(k), check_finite=False)
        K_T_chol = (L, lower)
    except np.linalg.LinAlgError:
        K_T_inv = np.linalg.solve(K_T, np.eye(k))
        K_T_chol = None

    state = {
        "w_T_sqrt": w_T_sqrt,
        "K_T_chol": K_T_chol,
        "K_T": K_T,
        "K_T_inv": K_T_inv,
    }
    if X_T is not None:
        U_T = X_T.T * (w_T_sqrt[None, :] / np.sqrt(n))
        state["P_T"] = cho_solve(
            (cache.H_chol_L, True), U_T, check_finite=False)
    return state
