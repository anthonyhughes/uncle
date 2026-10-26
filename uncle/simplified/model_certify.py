from typing import Optional, Sequence

import numpy as np


def _validate_gamma(gamma: float) -> float:
    """Validate the curvature-Lipschitz constant.

    The predicate is only sound for gamma in (0, 1]: gamma <= 0 or NaN would
    make the drift terms vanish or the conditions more permissive than the
    gamma = 1 predicate, and gamma = 0 turns the simple certifier's
    constant-only branch sentinel (c_lin = inf) into 0*inf = NaN, which the
    fail-closed gate would misreport as an exact infeasibility. gamma > 1
    is outside the range this predicate accepts.
    """
    g = float(gamma)
    if not (np.isfinite(g) and 0.0 < g <= 1.0):
        raise ValueError(f"gamma must be finite and in (0, 1]; got {gamma!r}")
    return g


def _scale_branches(
    branches: Sequence[tuple[float, float]], gamma: float
) -> tuple[tuple[float, float], ...]:
    """Scale Lambda_off branch pairs by gamma, preserving the inf sentinel.

    Each branch is (c_const, c_lin) meaning min(c_const, c_lin * S_inf);
    gamma multiplies Lambda_off wholesale at its consumption site in C4
    (min is positively homogeneous), and c_lin = inf marks a constant-only
    branch that must survive scaling as inf, never gamma*inf.
    """
    return tuple(
        (gamma * c_const, c_lin if np.isinf(c_lin) else gamma * c_lin)
        for c_const, c_lin in branches
    )

# theta_1 is the candidate theta (newton step candidate)
# theta_0 is the model at the full-data optimum

# G_* is the Gram matrix of the model theta_0 at the retained set
# We want the retained set's geometry (G) and our candidate's quality (A, B) 
# to guarantee that the true retained optimum θ* lies within radius R₂ of our candidate theta_1
# in prediction space.

# A_* — ‖X_R(θ_1 − θ_0)‖_{2,∞}: prediction-space displacement from θ₀ to θ₁
# prediction_displacement = X_ret @ (theta_1 - theta_0)

# B_* tell us how far theta_1 is from being the retained optimum

# R_* is the target L2/L∞ certificate radius for ‖X_R(θ* − θ_1)‖_*

def certify(
    # Gram bounds of (theta_0):
    G_op: float,  # upper bound operator norm of the retained theta_0 Gram
    G_D: float,  # upper bound max diagonal entry of the retained theta_0 Gram

    # Hypothesis:
    A_inf: float,  # infinity norm in prediction-space from theta_0 to theta_1

    # Candidate (theta_1):
    B_2: float,  # euclidean norm in prediction-space Newton residual at the candidate (theta_1)
    B_inf: float,  # euclidean infinity norm in prediction-space Newton residual at the candidate (theta_1)
    
    # Certificate radii:
    R_2: float,  # target euc norm certificate radius for ‖X_R(θ* − θ₁)‖₂
    R_inf: float,  # target euc max certificate radius for ‖X_R(θ* − θ₁)‖∞

    # Lambda off-diagonal:
    Lambda_off: float,  # max off-diagonal

    # Curvature-Lipschitz constant. 1.0 leaves drift products unscaled.
    gamma: float = 1.0,  # logistic upper bound: 1/(6*sqrt(3))
) -> bool:
    """The certificate predicate.

    Returns True iff the four conditions below all hold.

        ‖X_Rᵀ(θ* − θ_1)‖_2 ≤ R_2    and    ‖X_Rᵀ(θ* − θ_1)‖_∞ ≤ R_∞.

    All inputs are evaluated on the retained set at theta_0:
        G_op = ‖G⁰‖_op,  G_D = max_i G⁰_ii,  G_O = max_{i≠j} |G⁰_ij|. (Properties of the gram matrix)
        A_2, A_inf = ‖X_Rᵀ(θ_1 − θ_0)‖_{2,∞}            (Hessian changes)
        B_2, B_inf = ‖X_Rᵀ H_{R,0}⁻¹ ∇L_R(θ_1)‖_{2,∞}   (Newton-step residual)
        R_2, R_inf = target certificate radii. (Radius of the ball in prediction space around theta_1 that we want to certify contains the true retained optimum theta*)

    gamma is the curvature-Lipschitz constant. It multiplies every drift
    product (G_op·S_inf, G_D·S_inf, and Lambda_off where condition 4 consumes
    it). Inputs are raw constants; gamma is applied here exactly once (same
    single-scaling contract as the witness search).
    """
    gamma = _validate_gamma(gamma)

    # Guard for obviously failing certificates, and to NaN-like issues...
    if R_2 <= 0 or R_inf <= 0 or not (np.isfinite(R_2) and np.isfinite(R_inf)):
        return False

    # Apply the drift constant once, up front: every appearance of G_op, G_D
    # and Lambda_off in conditions (1)-(4) is a drift term.
    G_op = gamma * G_op
    G_D = gamma * G_D
    Lambda_off = gamma * Lambda_off

    #  S_∞ = A_∞ + R_∞.
    S_inf = A_inf + R_inf

    # Condition (1): G_op · S_∞ < 1
    #   Keeps the local Hessian positive-definite on the certificate region.
    if not (G_op * S_inf < 1.0):
        return False

    # Condition (2): G_D · S_∞ < 1
    #   Keeps the ∞-norm diagonal contraction (1 − G_D·S_∞) positive.
    #   Redundant whenever G_D ≤ G_op (the max diagonal of a PSD matrix never
    #   exceeds its operator norm), so condition (1) already implies it for
    #   exact inputs. Kept because G_op may be a bound computed independently
    #   of G_D, and the check is free.
    if not (G_D * S_inf < 1.0):
        return False

    # Condition (3): B_2 / (1 − G_op · S_∞) < R_2
    #   The L2 Newton-flow update stays within the L2 radius.
    #   (1 − G_op·S_∞) > 0 here because condition (1) already passed.
    # the norm of the Newton step inside the certificate region is 
    # smaller than R₂. The Newton map does not leave the ball we are certifying.
    if not (B_2 / (1.0 - G_op * S_inf) < R_2):
        return False

    # Condition (4): (B_∞ + Λ_off · B_2 / (1 − G_op·S_∞)) / (1 − G_D·S_∞) < R_∞
    #   The ∞-norm Newton-flow update stays within the ∞ radius.
    #   Both denominators are > 0 here because conditions (1) and (2) passed.
    c_4 = (B_inf + Lambda_off * B_2 / (1.0 - G_op * S_inf)) / (1.0 - G_D * S_inf)
    if not (c_4 < R_inf):
        return False

    return True


# ---------------------------------------------------------------------------
# Existential R_∞: exact feasibility test and per-query witness search
# ---------------------------------------------------------------------------
#
# The predicate holds for arbitrary target radii R_2, R_∞ > 0. Any single
# passing witness certifies the released L2 claim, and R_∞ never enters the
# privacy calibration (noise uses R_2 only). So instead of pinning R_∞ to
# the closed form 4·(B_∞ + C_O·(A_2 + B_2)²), the certifier may search
# for any feasible R_∞.
#
# The search is exact, not a scan. Write x = R_∞ and
#     p = 1 − G_op·A_∞,   q = 1 − G_D·A_∞.
# Conditions (1) and (3) reduce to 0 < x < (p − B_2/R_2)/G_op and condition
# (2) to x < q/G_D (redundant whenever G_D ≤ G_op, which holds structurally
# for a PSD Gram, but kept in the bound for unconditional soundness), so
# (1)–(3) hold iff 0 < x < U with
#     U = min( (p − B_2/R_2)/G_op ,  q/G_D ).
# On any sub-interval where Λ_off(x) is affine (α + β·x — Λ_off is a sum of
# two-term minima of constants and linear functions of S_∞ = A_∞ + x, hence
# piecewise affine with at most two breakpoints), condition (4) multiplies
# out to a cubic inequality φ(x) > 0 with
#     φ(x) = G_D·G_op·x³ − (q·G_op + p·G_D)·x²
#            + (p·q + B_∞·G_op − β·B_2)·x − (B_∞·p + α·B_2),
# so the exact feasible set is a union of root-bounded intervals (≤ 3 per
# branch piece). NOTE the accept indicator is NOT monotone in x (too small
# fails condition (4), too large fails condition (1)) — plain bisection on
# it is unsound; and the feasible set need not lie below R_2, so the search
# must not be capped at x = R_2.
#
# With the curvature-Lipschitz constant gamma made explicit, the γ-predicate
# is the γ=1 predicate evaluated at the scaled constants γ·G_op, γ·G_D,
# γ·Λ_off (γ multiplies every drift product and nothing else), so the
# reduction above is reused after scaling the Gram constants and the Λ_off
# branch pairs at entry.


def _lambda_off_at(
    branches: Sequence[tuple[float, float]], A_inf: float, x: float
) -> float:
    """Λ_off(x) = Σ_b min(c_const_b, c_lin_b·(A_∞ + x)); c_lin = inf means a
    constant-only contribution of c_const."""
    S_inf = A_inf + x
    total = 0.0
    for c_const, c_lin in branches:
        if np.isfinite(c_lin):
            total += min(c_const, c_lin * S_inf)
        else:
            total += c_const
    return total


def feasible_R_inf_intervals(
    *,
    G_op: float,
    G_D: float,
    A_inf: float,
    B_2: float,
    B_inf: float,
    R_2: float,
    lambda_off_branches: Sequence[tuple[float, float]],
    gamma: float = 1.0,
) -> tuple[list[tuple[float, float]], float]:
    """Exact feasible set {x > 0 : certify(..., R_inf=x, gamma=gamma) passes}.

    ``lambda_off_branches`` describes Λ_off(x) as a sum of two-term minima:
    each (c_const, c_lin) pair contributes min(c_const, c_lin·(A_∞ + x)),
    with c_lin = np.inf meaning the constant c_const alone. Inputs are RAW
    constants; gamma (the curvature-Lipschitz constant) is applied
    internally exactly once.

    Returns (intervals, U): a list of disjoint open intervals whose union is
    the feasible set, and the (1)–(3) upper limit U. Validated against dense
    400k-point grids of the coded ``certify()`` (scripts/r_inf_deep_dive).
    """
    intervals, U, _ = _feasible_R_inf_intervals_impl(
        G_op=G_op, G_D=G_D, A_inf=A_inf, B_2=B_2, B_inf=B_inf, R_2=R_2,
        lambda_off_branches=lambda_off_branches, gamma=gamma,
    )
    return intervals, U


def _feasible_R_inf_intervals_impl(
    *,
    G_op: float,
    G_D: float,
    A_inf: float,
    B_2: float,
    B_inf: float,
    R_2: float,
    lambda_off_branches: Sequence[tuple[float, float]],
    gamma: float = 1.0,
) -> tuple[list[tuple[float, float]], float, bool]:
    """Interval computation with an ``exact`` flag.

    ``exact`` is True when every piece was resolved analytically — in that
    case ``intervals == []`` is a PROOF of infeasibility and callers may skip
    any fallback scan. It is False when a piece had to be skipped for
    numerical reasons (non-finite/degenerate cubic coefficients), in which
    case an empty result is inconclusive. Like ``certify()`` itself, this
    function fails closed: non-finite geometry inputs yield ([], 0.0, True)
    (certify() rejects them at every R_∞) and numerical trouble can only
    under-report feasibility, never invent it.

    gamma scales the drift terms (G_op, G_D, and every Λ_off branch) at
    entry; the interval algebra below is then unchanged — the γ-predicate is
    exactly the γ=1 predicate at the scaled constants.
    """
    gamma = _validate_gamma(gamma)

    # Fail closed on non-finite geometry — certify() rejects NaN/inf inputs
    # at every R_∞ via its strict comparisons, so the feasible set is empty.
    # (Without this guard NaN coefficients would reach np.roots and raise.)
    # Validate on the RAW inputs, then scale.
    scalars = (G_op, G_D, A_inf, B_2, B_inf)
    if not all(np.isfinite(s) for s in scalars):
        return [], 0.0, True
    if not all(
        np.isfinite(c_const) and (np.isfinite(c_lin) or np.isinf(c_lin))
        for c_const, c_lin in lambda_off_branches
    ):
        return [], 0.0, True
    if not (np.isfinite(R_2) and R_2 > 0) or B_2 >= R_2:
        return [], 0.0, True

    G_op = gamma * G_op
    G_D = gamma * G_D
    lambda_off_branches = _scale_branches(lambda_off_branches, gamma)

    p = 1.0 - G_op * A_inf
    q = 1.0 - G_D * A_inf

    # Upper limit from conditions (1)+(3), and (2) kept for unconditional
    # soundness (redundant when G_D ≤ G_op).
    U = np.inf
    if G_op > 0:
        U = (p - B_2 / R_2) / G_op
    if G_D > 0:
        U = min(U, q / G_D)
    if np.isinf(U):
        # Degenerate zero-Gram case (G_op = G_D = 0); pick a large finite
        # search window — certify() itself stays the authority.
        U = 1e15
    if U <= 0:
        return [], float(U), True

    # Breakpoints where a min-branch switches between linear and constant.
    breakpoints = []
    for c_const, c_lin in lambda_off_branches:
        if np.isfinite(c_lin) and c_lin > 0:
            xb = c_const / c_lin - A_inf
            if 0.0 < xb < U:
                breakpoints.append(xb)
    knots = sorted(set([0.0, float(U)] + breakpoints))

    exact = True
    intervals: list[tuple[float, float]] = []
    for lo, hi in zip(knots[:-1], knots[1:]):
        mid = 0.5 * (lo + hi)
        S_mid = A_inf + mid
        # Active affine form Λ_off(x) = alpha + beta·x on this piece.
        alpha, beta = 0.0, 0.0
        for c_const, c_lin in lambda_off_branches:
            if np.isfinite(c_lin) and c_lin * S_mid <= c_const:
                alpha += c_lin * A_inf
                beta += c_lin
            else:
                alpha += c_const
        coeffs = [
            G_D * G_op,
            -(q * G_op + p * G_D),
            p * q + B_inf * G_op - beta * B_2,
            -(B_inf * p + alpha * B_2),
        ]
        # Trim leading coefficients that are zero or negligible RELATIVE to
        # the rest — a subnormal leading term (e.g. G_D·G_op ~ 1e-320) makes
        # np.roots' companion-matrix normalisation overflow.
        max_abs = max(abs(c) for c in coeffs)
        while len(coeffs) > 1 and (
            coeffs[0] == 0.0 or abs(coeffs[0]) < 1e-30 * max_abs
        ):
            coeffs = coeffs[1:]
        if not all(np.isfinite(c) for c in coeffs):
            exact = False  # skip piece; fail closed (may under-report only)
            continue
        try:
            roots = np.roots(coeffs) if len(coeffs) > 1 else np.array([])
        except np.linalg.LinAlgError:
            exact = False
            continue
        real_roots = sorted(
            r.real for r in roots
            if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)))
        points = [lo] + [r for r in real_roots if lo < r < hi] + [hi]
        for a, b in zip(points[:-1], points[1:]):
            if np.polyval(coeffs, 0.5 * (a + b)) > 0:
                if intervals and abs(intervals[-1][1] - a) <= 1e-12 * max(
                        1.0, abs(a)):
                    intervals[-1] = (intervals[-1][0], b)  # merge contiguous
                else:
                    intervals.append((a, b))
    return intervals, float(U), exact


def find_R_inf_witness(
    *,
    G_op: float,
    G_D: float,
    A_inf: float,
    B_2: float,
    B_inf: float,
    R_2: float,
    lambda_off_branches: Sequence[tuple[float, float]],
    gamma: float = 1.0,
) -> Optional[float]:
    """Per-query witness R_∞ making ``certify()`` pass, or None.

    Uses the exact interval test, then re-runs the coded ``certify()`` at
    each candidate as a self-check — only a value the actual predicate
    accepts is ever returned, so a bug in the interval algebra can cause a
    spurious reject but never a spurious accept. Falls back to a log-grid
    scan of (0, U) if no interval candidate survives the self-check.

    Single-scaling contract: inputs are RAW constants; gamma is handed to
    the interval impl and to certify(), each of which applies it internally
    exactly once. The self-check therefore computes Λ_off from the RAW
    branches — certify() scales it — so the intervals and the self-check
    always agree on the same γ-predicate (a double-scaled Λ_off would make
    condition 4 spuriously easy).
    """
    gamma = _validate_gamma(gamma)

    def _passes(x: float) -> bool:
        if not (x > 0 and np.isfinite(x)):
            return False
        lam_off = _lambda_off_at(lambda_off_branches, A_inf, x)
        return certify(
            G_op=G_op, G_D=G_D, A_inf=A_inf, B_2=B_2, B_inf=B_inf,
            R_2=R_2, R_inf=x, Lambda_off=lam_off, gamma=gamma,
        )

    intervals, U, exact = _feasible_R_inf_intervals_impl(
        G_op=G_op, G_D=G_D, A_inf=A_inf, B_2=B_2, B_inf=B_inf, R_2=R_2,
        lambda_off_branches=lambda_off_branches, gamma=gamma,
    )
    for a, b in sorted(intervals, key=lambda iv: iv[1] - iv[0], reverse=True):
        for t in (0.5, 0.25, 0.75, 0.1, 0.9):
            x = a + t * (b - a)
            if _passes(x):
                return float(x)

    # An exact empty result is a PROOF of infeasibility — skip the scan (it
    # would burn ~400 certify() calls per reject and distort per-query
    # timing). Scan only when a piece was skipped numerically (inconclusive)
    # or interval candidates failed the certify() self-check at the edges.
    if exact and not intervals:
        return None

    # Numerical fallback: coarse log+linear scan of (0, U).
    if U > 0 and np.isfinite(U):
        lo = max(U * 1e-12, 1e-300)
        candidates = np.concatenate([
            np.geomspace(lo, U, 200)[:-1],
            np.linspace(0.0, U, 202)[1:-1],
        ])
        for x in candidates:
            if _passes(float(x)):
                return float(x)
    return None
