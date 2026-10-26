from typing import TypedDict

import numpy as np
from scipy.linalg import eigvalsh
from scipy.special import expit

from .certify_trace import build_certify_trace, print_certify_trace
from .model_certify import certify, find_R_inf_witness


class GramBounds(TypedDict):
    """Constants of the retained Gram G⁰ = X_R H_{R,0}⁻¹ X_Rᵀ."""

    C_D: float
    C_O: float
    C_op: float


class PredictionDisplacement(TypedDict):
    """A₂ and A∞ norms of the prediction-space displacement X_R(θ₁ − θ₀)."""

    A_2: float
    A_inf: float


class NewtonResidual(TypedDict):
    """Newton-step residual at θ₁ using H_{R,0} = ∇²L_S(θ₀)."""

    B_2: float
    B_inf: float


class CertificateRadii(TypedDict):
    """Target radii and off-diagonal spillover bound for ``certify()``."""

    R_inf: float
    R_inf_source: str
    Lambda_off: float


def _fixed_ridge_matrix(n: int, lam: float, d: int) -> np.ndarray:
    """Ridge Λ = n λ I. n is fixed when samples are removed."""
    return n * lam * np.eye(d)


def retained_hessian_inverse_at_theta0(
    theta_0: np.ndarray,
    X_ret: np.ndarray,
    lam: float, # l2 reg
    n: int,
) -> np.ndarray:
    """H_{R,0} = ∇²L_S(θ₀) under sum-form, inverted directly.

    H_{R,0} = X_Rᵀ diag(w(θ₀)) X_R + Λ,  Λ = n λ I.
    """
    p0 = expit(X_ret @ theta_0) # sigmoid
    w0 = p0 * (1.0 - p0) # var
    d = X_ret.shape[1] # feature dimension

    # is n or n_ret here?
    H_R0 = (X_ret.T * w0) @ X_ret + _fixed_ridge_matrix(n, lam, d)
    return np.linalg.inv(H_R0) # inv


def retained_gram_constants(
    X_ret: np.ndarray,
    H_inv: np.ndarray,
) -> GramBounds:
    """C_D, C_O, C_op of G⁰ = X_R H_{R,0}⁻¹ X_Rᵀ."""
    G0 = X_ret @ H_inv @ X_ret.T

    diag = np.diag(G0)
    C_D = float(np.max(diag)) if diag.size else 0.0

    absG = np.abs(G0)
    np.fill_diagonal(absG, 0.0)
    C_O = float(np.max(absG)) if absG.size else 0.0

    # note: more efficient ways to compute the top eigenvalue of a symmetric matrix than full eigvalsh, but this is simple and works for small d
    C_op = float(np.max(np.abs(eigvalsh(G0)))) if G0.shape[0] else 0.0

    return GramBounds(C_D=C_D, C_O=C_O, C_op=C_op)

# note: consistency of calcs
def prediction_displacement_from_theta0(
    theta_0: np.ndarray,
    theta_1: np.ndarray,
    X_ret: np.ndarray,
) -> PredictionDisplacement:
    """A₂, A∞ — norms of X_R(θ₁ − θ₀) on the retained set."""
    prediction_displacement = X_ret @ (theta_1 - theta_0)
    return PredictionDisplacement(
        A_2=float(np.linalg.norm(prediction_displacement, 2)),
        A_inf=float(np.linalg.norm(prediction_displacement, np.inf)),
    )


def newton_residual_at_candidate(
    theta_1: np.ndarray,
    y_ret: np.ndarray,
    X_ret: np.ndarray,
    H_inv: np.ndarray,
    lam: float,
    n: int,
) -> NewtonResidual:
    """B₂, B∞ — ‖X_R H_{R,0}⁻¹ ∇L_S(θ₁)‖ under sum-form."""
    d = X_ret.shape[1] # feature dimension
    p1 = expit(X_ret @ theta_1) # sigmoid

    g = X_ret.T @ (p1 - y_ret) + _fixed_ridge_matrix(n, lam, d) @ theta_1

    Xv = X_ret @ (H_inv @ g)

    return NewtonResidual(
        B_2=float(np.linalg.norm(Xv, 2)),
        B_inf=float(np.linalg.norm(Xv, np.inf)),
    )


def certificate_radii(
    *,
    B_inf: float,
    C_O: float,
    A_2: float,
    B_2: float,
    R_2: float,
    C_op: float | None = None,
    C_D: float | None = None,
    A_inf: float | None = None,
    gamma: float = 1.0,
) -> CertificateRadii:
    """R_∞ and Λ_off for ``certify()``.

    Any R_∞ > 0 that makes the predicate pass is enough, so when the Gram
    constants are supplied this searches for a per-query witness (exact
    cubic feasibility test + ``certify()`` self-check). The closed form
    4·(B_∞ + C_O·(A_2+B_2)²) is the fallback when no witness exists (the
    predicate then rejects at it) or when the Gram constants are omitted.
    """
    Lambda_off = C_O * (A_2 + R_2)
    if C_op is not None and C_D is not None and A_inf is not None:
        witness = find_R_inf_witness(
            G_op=C_op, G_D=C_D, A_inf=A_inf, B_2=B_2, B_inf=B_inf, R_2=R_2,
            # Λ_off = C_O·S_2 is constant in R_∞ here: one constant-only branch
            # (the inf sentinel is gamma-safe: scaling preserves it).
            lambda_off_branches=((Lambda_off, np.inf),),
            gamma=gamma,
        )
        if witness is not None:
            return CertificateRadii(
                R_inf=witness, R_inf_source="witness", Lambda_off=Lambda_off)
        source = "closed_form_infeasible"
    else:
        source = "closed_form"
    R_inf = 4.0 * (B_inf + C_O * (A_2 + B_2) ** 2)
    return CertificateRadii(
        R_inf=R_inf, R_inf_source=source, Lambda_off=Lambda_off)


def simple_certifier(
    X_ret: np.ndarray,
    y_ret: np.ndarray,
    theta_0: np.ndarray,
    theta_1: np.ndarray,
    R_2: float,
    lam: float,
    n: int,
    *,
    gamma: float = 1.0,
    log_certify: bool = False,
    certify_context: dict | None = None,
) -> dict:
    """Certifier via direct sum-form Hessian inversion.

    Args:
        X_ret, y_ret: Retained data after deletion.
        theta_0: Full-data ERM optimum θ₀.
        theta_1: Candidate to certify.
        R_2: Target L2 certificate radius in prediction space.
        lam: Scalar regularisation. The ridge is Λ = n λ I.
        n: Original pre-deletion sample count.

    Returns dict (matching ``main_certifier`` shape for pipeline compatibility).
    """
    X_ret = np.asarray(X_ret, dtype=np.float64)
    y_ret = np.asarray(y_ret, dtype=np.float64)
    theta_0 = np.asarray(theta_0, dtype=np.float64)
    theta_1 = np.asarray(theta_1, dtype=np.float64)

    # note: to check n vs. n_ret throughout.
    n_ret = X_ret.shape[0]

    H_inv = retained_hessian_inverse_at_theta0(theta_0, X_ret, lam, n)
    gram = retained_gram_constants(X_ret, H_inv)
    displacement = prediction_displacement_from_theta0(theta_0, theta_1, X_ret)
    # is residual correct? newton step
    residual = newton_residual_at_candidate(
        theta_1, y_ret, X_ret, H_inv, lam, n,
    )
    radii = certificate_radii(
        B_inf=residual["B_inf"],
        C_O=gram["C_O"],
        A_2=displacement["A_2"],
        B_2=residual["B_2"],
        R_2=R_2,
        C_op=gram["C_op"],
        C_D=gram["C_D"],
        A_inf=displacement["A_inf"],
        gamma=gamma,
    )

    # C_*_eff keys match main_certifier quantities schema (same values here).
    quantities = {
        "C_D_eff": gram["C_D"],
        "C_O_eff": gram["C_O"],
        "C_op_eff": gram["C_op"],
        "r_2": residual["B_2"],
        "r_inf": residual["B_inf"],
        "A_2": displacement["A_2"],
        "A_inf": displacement["A_inf"],
        "n_ret": n_ret,
        "C_D": gram["C_D"],
        "C_O": gram["C_O"],
        "C_op": gram["C_op"],
        "Lambda_off": radii["Lambda_off"],
        "R_inf": radii["R_inf"],
        "R_inf_source": radii["R_inf_source"],
        "gamma": gamma,
    }

    certify_trace = build_certify_trace(
        G_op=gram["C_op"],
        G_D=gram["C_D"],
        G_O=gram["C_O"],
        A_2=displacement["A_2"],
        A_inf=displacement["A_inf"],
        B_2=residual["B_2"],
        B_inf=residual["B_inf"],
        R_2=R_2,
        R_inf=radii["R_inf"],
        Lambda_off=radii["Lambda_off"],
        gamma=gamma,
    )
    if log_certify:
        ctx = dict(certify_context or {})
        ctx.setdefault("n_ret", n_ret)
        ctx.setdefault("lam", lam)
        ctx.setdefault("R_2", R_2)
        print_certify_trace(certify_trace, context=ctx)

    accepted = certify(
        G_op=gram["C_op"],
        G_D=gram["C_D"],
        A_inf=displacement["A_inf"],
        B_2=residual["B_2"],
        B_inf=residual["B_inf"],
        R_2=R_2,
        R_inf=radii["R_inf"],
        Lambda_off=radii["Lambda_off"],
        gamma=gamma,
    )

    return {
        "accepted": accepted,
        "theta_2": theta_1,
        "B_2": residual["B_2"],
        "B_inf": residual["B_inf"],
        "R_inf": radii["R_inf"],
        "R_inf_source": radii["R_inf_source"],
        "quantities": quantities,
        "variant": "simple",
        "certify_trace": certify_trace,
    }
