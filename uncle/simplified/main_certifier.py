"""Cached-curvature certifier.

Curvature is evaluated at θ₀ (the full-data optimum) via a precomputed
``CachedCurvature``. For a deletion set T the retained Hessian

    H_{R,0} = (1/n) X_Rᵀ W(θ₀) X_R + λI

is an exact rank-k Woodbury downdate of the cached full-data Hessian
``H_full = (1/n) Xᵀ W X + λI`` (``H_{R,0} = H_full − (1/n) X_Tᵀ W_T X_T``),
so it is obtained in O(d² + dk) per right-hand side without a refactor.

The certificate predicate is ``model_certify.certify``. Its Gram inputs are
fast bounds on the retained Gram G⁰ = X_R H_{R,0}⁻¹ X_Rᵀ, computed in
O(n·k² + k³) without forming G⁰:

    C_D^(T)  = max_i (G_ii + q_i)                                 (exact)
    C_op^(T) ≤ C_op^full + λ_max(A^{1/2} ZᵀZ A^{1/2})             (bound)
    C_O^(T)  ≤ C_O^full + q_max                                   (bound)

each scaled by 1/n (mean-form Gram, converted to the sum-form constants the
predicate consumes). The curvature-Lipschitz coefficient ``gamma`` is a kwarg
on ``certify_at_radius`` / ``main_certifier`` (default 1.0 leaves the drift
products unscaled; 1/(6√3) is the tight logistic value). It multiplies the
drift products inside ``model_certify``, not these Gram constants. The
off-diagonal action bound is

    Λ_off = min{C_O^full·S₂, R_G^full·S∞} + min{q_max·S₂, R_Δ·S∞}.
"""

from typing import Optional

import numpy as np
from scipy.linalg import cho_solve, eigvalsh
from scipy.sparse.linalg import ArpackError, eigsh as sparse_eigsh
from scipy.special import expit

from .cached_curvature import CachedCurvature
from .certify_trace import build_certify_trace, print_certify_trace
from .model_certify import certify, find_R_inf_witness
from .woodbury import calc_woodbury_constants

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_certificate(
    X_ret: np.ndarray,
    y_ret: np.ndarray,
    indices_to_remove: np.ndarray,
    theta_1: np.ndarray,
    lam: float,
    cache: CachedCurvature,
    *,
    X_full: Optional[np.ndarray] = None,
    woodbury: Optional[dict] = None,
) -> dict:
    """Prepare all radius-independent inputs to the certificate predicate.

    Anchors Hessian/Gram at θ₀ = ``cache.theta_0`` (the full-data optimum) and
    computes the expensive per-query geometry once. The returned state may be
    passed to ``certify_at_radius`` for any number of candidate radii.

    Args:
        X_ret, y_ret: Retained data (X minus rows at ``indices_to_remove``).
        indices_to_remove: 1-D int array of original-row indices being deleted.
        theta_1: Initial candidate (e.g. one-step Newton hypothesis).
        lam: Regularisation coefficient. Must equal ``cache.lam``.
        cache: Pre-built ``CachedCurvature`` at θ₀ (full-data optimum).
        X_full: The full pre-deletion design matrix (n, d) used to build the
            cache; required to look up X[T] for the Woodbury downdate.
        woodbury: Optional state already built by ``cached_newton_step``.
    Returns:
        Radius-independent prepared certificate state.
    """
    # guards to prevent incorrect use

    # check the cache matches the original lambda
    if cache.lam != lam:
        raise ValueError(
            f"cache built with lam={cache.lam}, but verify_and_correct_"
            f"full_pass called with lam={lam}. Rebuild the cache or "
            f"pass the matching lambda."
        )

    # we suport logistic only
    if cache.loss_fn_id != "logistic":
        raise NotImplementedError(
            f"FullPass only supports logistic loss; cache says " f"{cache.loss_fn_id!r}"
        )

    # Required for post-deletion Hessian inverse via Woodbury: v = H_T⁻¹ g
    if X_full is None:
        raise ValueError(
            "X_full is required for the FullPass Woodbury solve; pass "
            "the full design matrix from the original dataset (this is the "
            "pre-deletion X used to build the cache)."
        )

    # the sample we want to remove
    T = np.asarray(indices_to_remove, dtype=np.intp).reshape(-1)

    # Rows being deleted, for the Woodbury downdate.
    X_T = X_full[T]  # (k, d)

    # Build per-query Woodbury objects (paid once per query)
    # One cached d×d solve plus a k×k correction
    # / we can use the Woodbury matrix identity to efficiently update the relevant statistics for any drop-set 
    # at the same initial point theta_0.
    if woodbury is None:
        woodbury = calc_woodbury_constants(cache, T, X_T=X_T)
    n_ret = cache.n - T.size

    # Exact retained Gram constants C_D / C_O / C_op (scaled by 1/n), via the
    # rank-k downdate of the cached Gram — no d-dimensional work.
    calculated_constants = compute_constants(cache, woodbury, T)

    # Curvature anchor is θ_0 (cache.theta_0, the full-data optimum); the
    # candidate being certified is the input θ_1 (the Newton-step hypothesis).
    # On accept we output that certified candidate.
    theta_1  # θ₁: Newton-step candidate for the retained optimum
    prediction_displacement = X_ret @ (theta_1 - cache.theta_0)  # X_R(θ₁ − θ₀)
    A_2 = float(np.linalg.norm(prediction_displacement, 2))  # ‖X_R(θ₁ − θ₀)‖₂
    A_inf = float(np.linalg.norm(prediction_displacement, np.inf))  # ‖X_R(θ₁ − θ₀)‖∞

    # Retained-loss gradient at the candidate, ∇L_R(θ₁) = (1/n) X_Rᵀ(p−y) + λθ.
    # The 1/n (original n) matches the cached mean-form Hessian used by the
    # Woodbury downdate — so v = H_{R,0}⁻¹ g is the correct Newton step.
    p = expit(X_ret @ theta_1)  # σ(X_R θ₁): predicted probabilities on retained set at the candidate
    g = (X_ret.T @ (p - y_ret)) / cache.n + lam * theta_1  # ∇L_R(θ₁): retained-loss gradient at the candidate

      # H_{R,0}⁻¹ ∇L_R(θ₁): Newton correction using H_{R,0} = ∇²L_R(θ₀)
    v = _apply_H_T_inv(cache, X_T, woodbury, g)

    Xv = X_ret @ v  # X_R · H_{R,0}⁻¹ ∇L_R(θ₁): prediction-space Newton residual
    B_2 = float(np.linalg.norm(Xv, 2))  # ‖X_R H_{R,0}⁻¹ ∇L_R(θ₁)‖₂: Euclidean norm of prediction-space residual
    B_inf = float(np.linalg.norm(Xv, np.inf))  # ‖X_R H_{R,0}⁻¹ ∇L_R(θ₁)‖∞: infinity norm of prediction-space residual

    # Closed-form fallback ∞-radius: R_∞ = 4·(B_∞ + G_O·(A_2 + B_2)²).
    # Used when the search finds no feasible witness. ``certify_at_radius``
    # searches for any R_∞ > 0 that makes the predicate pass; that witness is
    # not part of the released radius (noise uses R_2 only).
    G_O_eff = calculated_constants["C_O_eff"]
    R_inf = 4.0 * (B_inf + G_O_eff * (A_2 + B_2) ** 2)

    return {
        "theta_1": theta_1,
        "B_2": B_2,
        "B_inf": B_inf,
        "A_2": A_2,
        "A_inf": A_inf,
        "R_inf": R_inf,
        "n_ret": n_ret,
        "constants": calculated_constants,
    }


def certify_at_radius(
    prepared: dict,
    R_2: float,
    *,
    R_inf: Optional[float] = None,
    search_R_inf: bool = True,
    gamma: float = 1.0,
    log_certify: bool = False,
    certify_context: Optional[dict] = None,
) -> dict:
    """Evaluate a prepared certificate at one radius using scalar work only.

    Any R_∞ > 0 that makes the predicate pass is a valid witness (R_∞ is not
    released; noise is calibrated to R_2 only), so by default this searches
    for one via the exact per-branch cubic test in ``model_certify`` and
    self-checks it against ``certify()``. A witness found by search accepts
    whenever the closed-form fallback would.

    Args:
        R_inf: Explicit witness override (diagnostic hook). When given, no
            search runs and the predicate is evaluated exactly at this value.
        search_R_inf: Set False to use the fixed closed form
            ``prepared["R_inf"]`` = 4·(B_∞ + G_O·(A_2 + B_2)²).
        gamma: Curvature-Lipschitz constant. Default 1.0 leaves the drift
            products unscaled; 1/(6√3) is the tight logistic value.
            Raw constants are passed down; the scaling happens exactly once
            inside model_certify/certify_trace. The closed-form fallback
            ``prepared["R_inf"]`` is the gamma=1 expression — it is only used
            for trace continuity on infeasible rows, never as a certificate.
    """
    constants = prepared["constants"]
    A_2 = prepared["A_2"]
    A_inf = prepared["A_inf"]
    B_2 = prepared["B_2"]
    B_inf = prepared["B_inf"]
    S_2 = A_2 + R_2

    # Λ_off(S_∞) shared by the search and the final evaluation:
    # min(C_O^full·S_2, R_G^full·S_∞) + min(q_max·S_2, R_Δ·S_∞).
    lambda_off_branches = (
        (constants["C_O_full_eff"] * S_2, constants["R_G_full_eff"]),
        (constants["q_max_eff"] * S_2, constants["R_Delta_eff"]),
    )

    if R_inf is not None:
        R_inf_source = "override"
    elif search_R_inf:
        witness = find_R_inf_witness(
            G_op=constants["C_op_eff"], G_D=constants["C_D_eff"],
            A_inf=A_inf, B_2=B_2, B_inf=B_inf, R_2=R_2,
            lambda_off_branches=lambda_off_branches,
            gamma=gamma,
        )
        if witness is not None:
            R_inf = witness
            R_inf_source = "witness"
        else:
            # Infeasible for every R_∞ — evaluate at the closed form so the
            # trace/binding-condition attribution stays comparable.
            R_inf = prepared["R_inf"]
            R_inf_source = "closed_form_infeasible"
    else:
        R_inf = prepared["R_inf"]
        R_inf_source = "closed_form"

    S_inf = A_inf + R_inf
    Lambda_off = min(
        constants["C_O_full_eff"] * S_2,
        constants["R_G_full_eff"] * S_inf,
    ) + min(
        constants["q_max_eff"] * S_2,
        constants["R_Delta_eff"] * S_inf,
    )

    quantities = {
        "C_D_eff": constants["C_D_eff"],
        "C_O_eff": constants["C_O_eff"],
        "C_op_eff": constants["C_op_eff"],
        "r_2": B_2,
        "r_inf": B_inf,
        "A_2": A_2,
        "A_inf": A_inf,
        "n_ret": prepared["n_ret"],
        "C_D": constants["C_D"],
        "C_O": constants["C_O"],
        "C_op": constants["C_op"],
        "Lambda_off": Lambda_off,
        "q_max_eff": constants["q_max_eff"],
        "R_Delta_eff": constants["R_Delta_eff"],
        "C_O_full_eff": constants["C_O_full_eff"],
        "R_G_full_eff": constants["R_G_full_eff"],
        "S_2": S_2,
        "S_inf": S_inf,
        "R_inf": R_inf,
        "R_inf_source": R_inf_source,
        "R_inf_closed_form": prepared["R_inf"],
        "gamma": gamma,
    }
    accepted = certify(
        G_op=constants["C_op_eff"], G_D=constants["C_D_eff"],
        A_inf=A_inf, B_2=B_2, B_inf=B_inf, R_2=R_2, R_inf=R_inf,
        Lambda_off=Lambda_off, gamma=gamma,
    )
    certify_trace = build_certify_trace(
        G_op=constants["C_op_eff"], G_D=constants["C_D_eff"],
        G_O=constants["C_O_eff"], A_2=A_2, A_inf=A_inf, B_2=B_2,
        B_inf=B_inf, R_2=R_2, R_inf=R_inf, Lambda_off=Lambda_off,
        gamma=gamma,
    )
    if log_certify:
        ctx = dict(certify_context or {})
        ctx.setdefault("n_ret", prepared["n_ret"])
        ctx.setdefault("R_2", R_2)
        print_certify_trace(certify_trace, context=ctx)
    return {
        "accepted": accepted,
        "theta_2": prepared["theta_1"],
        "B_2": B_2,
        "B_inf": B_inf,
        "R_inf": R_inf,
        "R_inf_source": R_inf_source,
        "quantities": quantities,
        "variant": "fullpass",
        "certify_trace": certify_trace,
    }


def main_certifier(
    X_ret: np.ndarray,
    y_ret: np.ndarray,
    indices_to_remove: np.ndarray,
    theta_1: np.ndarray,
    R_2: float,
    lam: float,
    cache: CachedCurvature,
    *,
    X_full: Optional[np.ndarray] = None,
    R_inf: Optional[float] = None,
    search_R_inf: bool = True,
    gamma: float = 1.0,
    log_certify: bool = False,
    certify_context: Optional[dict] = None,
    woodbury: Optional[dict] = None,
) -> dict:
    """Backward-compatible prepare-and-evaluate certifier entry point."""
    prepared = prepare_certificate(
        X_ret, y_ret, indices_to_remove, theta_1, lam, cache,
        X_full=X_full, woodbury=woodbury,
    )
    return certify_at_radius(
        prepared, R_2, R_inf=R_inf, search_R_inf=search_R_inf, gamma=gamma,
        log_certify=log_certify, certify_context=certify_context,
    )


def compute_constants(
    cache: CachedCurvature,
    woodbury: dict,
    indices_to_remove: np.ndarray,
) -> dict:
    """Fast Gram summaries for the retained set at θ₀.

    Never forms the |R|×|R| retained Gram.
    From the cached Gram slices and the per-query Woodbury blocks:

        Z   = (1/√n) · G[R,T] · W_T^{1/2}          (n_ret × k)
        q_i = z_i K_T⁻¹ z_iᵀ                       (diag of the update Z K_T⁻¹ Zᵀ)
        Γ   = Zᵀ Z                                 (k × k)

        C_D^(T)  = max_i (G_ii + q_i)                              (exact)
        C_op^(T) ≤ C_op^full + λ_max(LᵀΓL),  K_T⁻¹ = L Lᵀ          (bound)
        C_O^(T)  ≤ C_O^full + q_max                                (bound, CS)
        R_Δ      = max_i ‖row_i(Z K_T⁻¹ Zᵀ)‖₂ = max_i √(z_i K⁻¹ Γ K⁻¹ z_iᵀ)

    each *_eff value scaled by 1/n (original n). ``C_O_full_eff`` and
    ``R_G_full_eff`` (global cached-Gram summaries) are returned for the
    caller's two-term Λ_off.

    Cost: O(n_ret·k² + k³), plus a one-time O(n²) pass for the memoized
    off-diagonal row norm of the full cached Gram.
    """
    n = cache.n
    T = np.asarray(indices_to_remove, dtype=np.intp).reshape(-1)
    mask = np.ones(n, dtype=bool)
    mask[T] = False
    R_idx = np.flatnonzero(mask)

    w_T_sqrt = woodbury["w_T_sqrt"]  # (k,)
    A = woodbury["K_T_inv"]  # (k, k), SPD
    G = cache.G

    # Z = (1/√n) G[R,T] W_T^{1/2}: retained-deleted coupling, read off the cache.
    Z = G[np.ix_(R_idx, T)] * (w_T_sqrt[None, :] / np.sqrt(n))  # (n_ret, k)

    # Diagonal of the rank-k update Z A Zᵀ, rowwise: q_i = z_i A z_iᵀ.
    ZA = Z @ A  # (n_ret, k)
    q = np.einsum("ij,ij->i", ZA, Z)  # (n_ret,), ≥ 0
    q_max = float(np.max(q)) if q.size else 0.0

    # Small k×k Gram of the coupling, shared by R_Δ and the operator bound.
    Gamma = Z.T @ Z  # (k, k)

    # R_Δ = max row ℓ₂-norm of Z A Zᵀ:  √(z_i A Γ A z_iᵀ).
    B_small = A @ Gamma @ A  # (k, k)
    row_sq = np.einsum("ij,ij->i", Z @ B_small, Z)
    R_Delta = float(np.sqrt(max(float(np.max(row_sq)), 0.0))) if row_sq.size else 0.0

    # Exact post-deletion diagonal: max_i (G_ii + q_i) over retained rows.
    g_diag = cache.G_diag if cache.G_diag is not None else np.diag(G)
    C_D = float(np.max(g_diag[R_idx] + q)) if R_idx.size else 0.0

    # Operator bound: λ_max(A^{1/2} Γ A^{1/2}) = λ_max(LᵀΓL) with A = L Lᵀ.
    try:
        L = np.linalg.cholesky(A)
        op_update = _lambda_max_psd(L.T @ Gamma @ L)
    except np.linalg.LinAlgError:
        # A is SPD in exact arithmetic; on numerical failure fall back to the
        # looser product bound λ_max(A^{1/2}ΓA^{1/2}) ≤ λ_max(A)·λ_max(Γ).
        op_update = _lambda_max_psd(0.5 * (A + A.T)) * _lambda_max_psd(Gamma)
    C_op = float(cache.C_op) + op_update

    # Entrywise bound via Cauchy–Schwarz on the PSD update: |(ZAZᵀ)_ij| ≤ q_max.
    C_O = float(cache.C_O) + q_max

    # Global off-diagonal row norm of the full cached Gram (memoized once).
    R_G_full = _off_diag_row_norm_full(cache)

    inv_n = 1.0 / n

    # *_eff constants are the ones fed to the predicate / Λ_off.
    return {
        "C_D_eff": inv_n * C_D,
        "C_O_eff": inv_n * C_O,
        "C_op_eff": inv_n * C_op,
        "q_max_eff": inv_n * q_max,
        "R_Delta_eff": inv_n * R_Delta,
        "C_O_full_eff": inv_n * float(cache.C_O),
        "R_G_full_eff": inv_n * R_G_full,
        # Unscaled raw constants — diagnostics/tests.
        "C_D": C_D,
        "C_O": C_O,
        "C_op": C_op,
        "q_max": q_max,
        "R_Delta": R_Delta,
    }


def _off_diag_row_norm_full(cache: CachedCurvature) -> float:
    """max_i ‖(G_ij)_{j≠i}‖₂ over the full cached Gram.

    A global summary valid for every retained subset (dropping columns can
    only shrink a row's off-diagonal norm). Computed once per loaded cache
    with an O(n²) pass and memoized on the instance; not persisted, so no
    cache-file version bump is needed.
    """
    val = getattr(cache, "_off_diag_row_norm_full_", None)
    if val is None:
        G = cache.G
        g_diag = cache.G_diag if cache.G_diag is not None else np.diag(G)
        row_sq = np.einsum("ij,ij->i", G, G) - g_diag**2
        val = float(np.sqrt(max(float(np.max(row_sq)), 0.0))) if row_sq.size else 0.0
        cache._off_diag_row_norm_full_ = val
    return val


def _lambda_max_psd(G0: np.ndarray) -> float:
    """Largest eigenvalue of a symmetric PSD matrix (= its operator norm).

    G⁰ = X_R H_{R,0}⁻¹ X_Rᵀ is PSD (H_{R,0} ≻ 0), so ‖G⁰‖_op = λ_max and only
    the single extreme eigenvalue is needed — not the full spectrum. For large
    matrices Lanczos (ARPACK) finds it to machine precision in O(m²·iters)
    instead of the O(m³) dense eigendecomposition (≈27× faster at m = 4000).
    A fixed start vector keeps runs deterministic; small matrices and ARPACK
    convergence failures fall back to the dense solve.
    """
    m = G0.shape[0]
    if m == 0:
        return 0.0
    if m < 500:
        return float(np.max(np.abs(eigvalsh(G0))))
    try:
        v0 = np.full(m, 1.0 / np.sqrt(m))
        vals = sparse_eigsh(G0, k=1, which="LA", v0=v0, return_eigenvectors=False)
        return float(vals[0])
    except (ArpackError, np.linalg.LinAlgError):
        return float(np.max(np.abs(eigvalsh(G0))))


def cached_newton_step(
    cache: CachedCurvature,
    X_T: np.ndarray,
    y_T: np.ndarray,
    indices_to_remove: np.ndarray,
    *,
    woodbury: Optional[dict] = None,
    return_woodbury: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """One-step Newton hypothesis via the cached Woodbury inverse.

    θ₁ = θ₀ − H_{R,0}⁻¹ ∇L_R(θ₀). Since θ₀ minimises the full loss,
    ∇L_R(θ₀) = −(1/n) Σ_{i∈T} (σ(xᵢᵀθ₀) − yᵢ) xᵢ — only the deleted samples
    enter the gradient. The solve reuses the cached
    Hessian factor through the Woodbury identity, so no fresh Hessian is
    formed: cost O(d² + dk + k³) instead of the O(nd² + d³) of
    ``newton_step_hypothesis``. Identical value (same retained Hessian, same
    gradient); only the solve differs.

    Args:
        cache: Pre-built ``CachedCurvature`` at θ₀ (full-data optimum).
        X_T: Deleted rows X[T], shape (k, d).
        y_T: Deleted labels y[T], shape (k,).
        indices_to_remove: 1-D int array of deleted original-row indices.
    Returns:
        theta_1: One-step Newton approximation to the retained ERM, shape (d,).
    """
    T = np.asarray(indices_to_remove, dtype=np.intp).reshape(-1)

    # grad_sum = (1/n) Σ_{i∈T} (σ(xᵢᵀθ₀) − yᵢ) xᵢ = −∇L_R(θ₀)  (mean-form).
    p_T = expit(X_T @ cache.theta_0)
    grad_sum = X_T.T @ (p_T - y_T) / cache.n  # (d,)

    if woodbury is None:
        woodbury = calc_woodbury_constants(cache, T, X_T=X_T)
    delta = _apply_H_T_inv(cache, X_T, woodbury, grad_sum)  # H_{R,0}⁻¹ grad_sum
    theta_1 = cache.theta_0 + delta
    if return_woodbury:
        return theta_1, woodbury
    return theta_1


def _apply_H_T_inv(
    cache: CachedCurvature, X_T: np.ndarray, woodbury: dict, v: np.ndarray
) -> np.ndarray:
    """Apply H_{R,0}⁻¹ to ``v`` via the Woodbury identity:

        H_{R,0}⁻¹ = H⁻¹ + H⁻¹ U_T K_T⁻¹ U_Tᵀ H⁻¹

    where U_T = (1/√n) X_Tᵀ W_T^{1/2} so that U_T U_Tᵀ = (1/n) X_Tᵀ W_T X_T,
    the exact mean-form rank-k downdate (H_{R,0} = H_full − U_T U_Tᵀ).

    ``v`` may be a vector (d,) or a matrix (d, m); the matrix form is used to
    build the exact retained Gram. Cost: O(d² + dk) per column.
    """
    H_chol = cache.H_chol_L
    w_T_sqrt = woodbury["w_T_sqrt"]  # (k,)
    K_T_chol = woodbury.get("K_T_chol")
    inv_n_full = 1.0 / cache.n

    # Broadcast W_T^{1/2} along the leading (k) axis for both v and matrix RHS.
    w = w_T_sqrt if v.ndim == 1 else w_T_sqrt[:, None]

    # First term: H⁻¹ v
    a = cho_solve((H_chol, True), v, check_finite=False)  # (d,) or (d, m)

    # Second term: P_T K_T⁻¹ P_Tᵀv, with P_T = H⁻¹U_T prepared once.
    P_T = woodbury.get("P_T")
    if P_T is not None:
        UT_a = P_T.T @ v
    else:
        UT_a = inv_n_full**0.5 * (w * (X_T @ a))
    if K_T_chol is not None:
        Kinv_UT_a = cho_solve(K_T_chol, UT_a, check_finite=False)
    else:
        Kinv_UT_a = np.linalg.solve(woodbury["K_T"], UT_a)

    if P_T is not None:
        correction = P_T @ Kinv_UT_a
    else:
        correction_rhs = inv_n_full**0.5 * (X_T.T @ (w * Kinv_UT_a))
        correction = cho_solve(
            (H_chol, True), correction_rhs, check_finite=False)
    return a + correction
