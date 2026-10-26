"""Phase 1 regression tests for the existential R_∞ witness search.

R_∞ is existentially quantified in Theorem A.6 (main.tex:1278 states the
predicate for arbitrary positive radii; the closed form 4(B_∞+C_O(A_2+B_2)²)
is only the Theorem A.5 proof's "we will choose" witness). These tests pin
the three Phase 1 guarantees from R_INF_TODO.md:

  (a) search-accept ⊇ fixed-accept, and the search strictly recovers cases
      on a seeded problem grid;
  (b) every accepted certificate is SOUND: the ground-truth retained optimum
      (exact-Newton-polished, original-n fixed-ridge convention) satisfies
      the released L₂ claim ‖X_R(θ* − θ₁)‖₂ ≤ R₂;
  (c) the exact cubic feasibility test agrees with a dense grid scan of the
      coded ``certify()`` predicate.
"""

import numpy as np
import pytest
from scipy.special import expit

from uncle.simplified.cached_curvature import CachedCurvature
from uncle.simplified.main_certifier import (
    cached_newton_step,
    certify_at_radius,
    prepare_certificate,
)
from uncle.simplified.model_certify import (
    _lambda_off_at,
    certify,
    feasible_R_inf_intervals,
    find_R_inf_witness,
)
from tests.certifier_fixtures import default_R2, remove_samples, retrain_exact


def _gaussian_pac(n, d, rng):
    theta_star = rng.standard_normal(d)
    theta_star /= np.linalg.norm(theta_star)
    X = rng.standard_normal((n, d))
    y = (rng.uniform(size=n) < expit(X @ theta_star)).astype(np.float64)
    return X, y


def _prepared_case(seed, n=120, d=8, k=5, lam=0.1):
    rng = np.random.default_rng(seed)
    X, y = _gaussian_pac(n, d, rng)
    theta_0 = retrain_exact(X, y, lam, d)
    cache = CachedCurvature.build(X, y, theta_0, lam)
    T = rng.choice(n, size=k, replace=False).astype(np.intp)
    X_ret, y_ret = remove_samples(X, y, set(T.tolist()))
    theta_1, woodbury = cached_newton_step(
        cache, X[T], y[T], T, return_woodbury=True)
    prepared = prepare_certificate(
        X_ret, y_ret, T, theta_1, lam, cache, X_full=X, woodbury=woodbury)
    return X, y, X_ret, y_ret, T, lam, theta_1, prepared


def _newton_polish(X_ret, y_ret, lam, n, theta, max_iter=50, tol=1e-13):
    """Exact-Newton polish of the retained mean-form fixed-ridge objective.

    L_R(θ) = (1/n) Σ_{i∈R} ℓ_i(θ) + (λ/2)‖θ‖² with the ORIGINAL n — the
    convention shared by retrain_exact(..., n=n) and the certifier. L-BFGS
    tolerance noise alone produced false violations in the deep-dive probe,
    so ground-truth checks must polish to machine precision.
    """
    d = X_ret.shape[1]
    for _ in range(max_iter):
        p = expit(X_ret @ theta)
        g = X_ret.T @ (p - y_ret) / n + lam * theta
        if np.linalg.norm(g) <= tol:
            break
        w = p * (1.0 - p)
        H = (X_ret.T * w) @ X_ret / n + lam * np.eye(d)
        theta = theta - np.linalg.solve(H, g)
    return theta


# ---------------------------------------------------------------------------
# (a) Search dominates the fixed closed form
# ---------------------------------------------------------------------------

SEED_GRID = [(seed, k, sf) for seed in range(12) for k in (1, 5, 10)
             for sf in (5.0, 50.0)]


def test_search_accept_superset_of_fixed_accept_and_recovers_cases():
    n_fixed = n_search = 0
    for seed, k, sf in SEED_GRID:
        X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(
            seed, k=k)
        R_2 = default_R2(k, X.shape[1], X.shape[0], safety_factor=sf)
        fixed = certify_at_radius(prepared, R_2, search_R_inf=False)
        searched = certify_at_radius(prepared, R_2)
        if fixed["accepted"]:
            assert searched["accepted"], (
                f"search lost a fixed-accept case: seed={seed} k={k} sf={sf}")
        n_fixed += fixed["accepted"]
        n_search += searched["accepted"]
    assert n_search >= n_fixed
    # The searched certifier must strictly recover rejects on this grid
    # (the deep-dive probe measured +12.7pt overall; this grid includes the
    # sf=50 regime where recovery is largest).
    assert n_search > n_fixed, (
        f"search recovered nothing: fixed={n_fixed} searched={n_search}")


def test_result_reports_witness_source_and_legacy_closed_form():
    X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(0, k=5)
    R_2 = default_R2(5, X.shape[1], X.shape[0], safety_factor=50.0)
    searched = certify_at_radius(prepared, R_2)
    assert searched["R_inf_source"] in (
        "witness", "closed_form_infeasible")
    assert searched["quantities"]["R_inf_closed_form"] == prepared["R_inf"]
    legacy = certify_at_radius(prepared, R_2, search_R_inf=False)
    assert legacy["R_inf_source"] == "closed_form"
    assert legacy["R_inf"] == prepared["R_inf"]


def test_explicit_R_inf_override_is_used_verbatim():
    X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(3, k=5)
    R_2 = default_R2(5, X.shape[1], X.shape[0], safety_factor=10.0)
    override = 0.123456
    result = certify_at_radius(prepared, R_2, R_inf=override)
    assert result["R_inf"] == override
    assert result["R_inf_source"] == "override"
    assert result["certify_trace"]["inputs"]["R_inf"] == override


# ---------------------------------------------------------------------------
# (b) Soundness: accepted ⇒ ground-truth L₂ claim holds
# ---------------------------------------------------------------------------


def test_accepted_witness_satisfies_ground_truth_L2_claim():
    n_accepted = 0
    for seed in range(20):
        for sf in (5.0, 50.0):
            k = 5
            X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(
                seed, k=k)
            n, d = X.shape
            R_2 = default_R2(k, d, n, safety_factor=sf)
            result = certify_at_radius(prepared, R_2)
            if not result["accepted"]:
                continue
            n_accepted += 1
            theta_star = retrain_exact(X_ret, y_ret, lam, d, n=n)
            theta_star = _newton_polish(X_ret, y_ret, lam, n, theta_star)
            err = float(np.linalg.norm(X_ret @ (theta_star - theta_1)))
            assert err <= R_2, (
                f"UNSOUND accept: seed={seed} sf={sf} ‖X_R(θ*−θ₁)‖₂={err:.3e}"
                f" > R₂={R_2:.3e} (witness R_∞={result['R_inf']:.3e})")
    # the grid must actually exercise accepts for this test to mean anything
    assert n_accepted >= 5, f"only {n_accepted} accepts on soundness grid"


# ---------------------------------------------------------------------------
# (c) Exact feasibility test agrees with a dense grid of certify()
# ---------------------------------------------------------------------------


def _random_predicate_case(rng):
    G_D = rng.uniform(0.01, 1.5)
    G_op = G_D * rng.uniform(1.0, 2.5)
    C_Of = rng.uniform(0, G_D)
    q_m = rng.uniform(0, 0.3 * G_D)
    A_2 = rng.uniform(0, 5)
    B_2 = rng.uniform(0.01, 2)
    return dict(
        G_op=G_op, G_D=G_D, A_inf=A_2 * rng.uniform(0.05, 1.0),
        B_2=B_2, B_inf=B_2 * rng.uniform(0.05, 1.0),
        R_2=rng.uniform(0.05, 5),
        branches_raw=(C_Of, q_m, C_Of * rng.uniform(1, 20),
                      q_m * rng.uniform(1, 20)),
        A_2=A_2,
    )


@pytest.mark.parametrize("case_seed", range(4))
def test_exact_intervals_match_dense_grid(case_seed):
    rng = np.random.default_rng(1000 + case_seed)
    for _ in range(25):
        c = _random_predicate_case(rng)
        S_2 = c["A_2"] + c["R_2"]
        C_Of, q_m, R_Gf, R_D = c["branches_raw"]
        branches = ((C_Of * S_2, R_Gf), (q_m * S_2, R_D))
        kw = dict(G_op=c["G_op"], G_D=c["G_D"], A_inf=c["A_inf"],
                  B_2=c["B_2"], B_inf=c["B_inf"], R_2=c["R_2"],
                  lambda_off_branches=branches)
        intervals, U = feasible_R_inf_intervals(**kw)

        def _passes(x):
            return certify(
                G_op=c["G_op"], G_D=c["G_D"], A_inf=c["A_inf"], B_2=c["B_2"],
                B_inf=c["B_inf"], R_2=c["R_2"], R_inf=x,
                Lambda_off=_lambda_off_at(branches, c["A_inf"], x))

        if U <= 0 or c["B_2"] >= c["R_2"]:
            assert intervals == []
            continue
        grid = np.linspace(1e-9, U * 0.9999999, 20001)
        passes = np.array([_passes(float(x)) for x in grid])

        # every grid point must agree with membership in the exact intervals
        tol = 5e-4 * max(1.0, U)
        for x, ok in zip(grid, passes):
            in_exact = any(a - tol <= x <= b + tol for a, b in intervals)
            near_boundary = any(
                abs(x - a) <= tol or abs(x - b) <= tol for a, b in intervals)
            if not near_boundary:
                assert ok == in_exact, (
                    f"grid/exact disagree at x={x:.6g}: certify={ok}, "
                    f"intervals={intervals}, U={U}")

        # and any witness returned must pass the coded predicate
        w = find_R_inf_witness(**kw)
        if intervals:
            assert w is not None and _passes(w)
        elif w is not None:
            assert _passes(w)


def test_nonfinite_geometry_fails_closed_not_crash():
    """NaN/inf geometry (e.g. a diverged upstream solver) must reject
    gracefully like the legacy predicate, never raise (review finding)."""
    for bad in (np.nan, np.inf):
        for field in ("G_op", "G_D", "A_inf", "B_2", "B_inf"):
            kw = dict(G_op=0.3, G_D=0.2, A_inf=0.1, B_2=0.05, B_inf=0.02,
                      R_2=1.0, lambda_off_branches=((0.1, 0.2),))
            kw[field] = bad
            assert find_R_inf_witness(**kw) is None
            assert feasible_R_inf_intervals(**kw) == ([], 0.0)


def test_nonfinite_candidate_end_to_end_rejects_not_crash():
    """theta_1 full of NaN/inf through the whole certifier → accepted=False."""
    for bad in (np.nan, np.inf):
        X, y, X_ret, y_ret, T, lam, theta_1, _ = _prepared_case(1, k=5)
        n, d = X.shape
        theta_bad = np.full(d, bad)
        cache_theta0 = theta_1  # unused; rebuild prepared with bad candidate
        rng = np.random.default_rng(1)
        theta_0 = retrain_exact(X, y, lam, d)
        cache = CachedCurvature.build(X, y, theta_0, lam)
        theta1_good, woodbury = cached_newton_step(
            cache, X[T], y[T], T, return_woodbury=True)
        prepared = prepare_certificate(
            X_ret, y_ret, T, theta_bad, lam, cache,
            X_full=X, woodbury=woodbury)
        R_2 = default_R2(5, d, n, safety_factor=5.0)
        result = certify_at_radius(prepared, R_2)
        assert result["accepted"] is False


def test_subnormal_gram_constants_do_not_crash():
    """Subnormal G_D·G_op product must not overflow np.roots' companion
    matrix (review finding); any witness returned must pass certify()."""
    kw = dict(G_op=1e-160, G_D=1e-160, A_inf=0.0, B_2=0.05, B_inf=0.02,
              R_2=1.0, lambda_off_branches=((0.5, 1.0), (0.1, 0.2)))
    w = find_R_inf_witness(**kw)
    if w is not None:
        lam_off = _lambda_off_at(kw["lambda_off_branches"], 0.0, w)
        assert certify(G_op=1e-160, G_D=1e-160, A_inf=0.0, B_2=0.05,
                       B_inf=0.02, R_2=1.0, R_inf=w, Lambda_off=lam_off)


def test_structurally_infeasible_returns_none_immediately():
    """G_op·A_inf ≥ 1 makes every R_∞ fail condition (1): exact proof of
    infeasibility, witness must be None (and no fallback scan is needed)."""
    kw = dict(G_op=2.0, G_D=1.0, A_inf=0.6, B_2=0.05, B_inf=0.02, R_2=1.0,
              lambda_off_branches=((0.1, 0.2),))
    intervals, U = feasible_R_inf_intervals(**kw)
    assert intervals == [] and U <= 0
    assert find_R_inf_witness(**kw) is None


def test_witness_not_capped_at_R2():
    """Feasible sets can live entirely above R₂ — the search must not cap
    there (documented trap from the deep dive)."""
    # Construct: B_inf large relative to R_2 so condition (4) needs x > R_2,
    # while G_op is small enough that condition (1) still allows it.
    kw = dict(G_op=0.05, G_D=0.05, A_inf=0.0, B_2=0.05, B_inf=2.0, R_2=1.0,
              lambda_off_branches=((0.0, 0.0),))
    intervals, U = feasible_R_inf_intervals(**kw)
    assert intervals, "expected a feasible interval"
    assert all(a >= kw["R_2"] for a, _ in intervals), (
        "this construction should only be feasible above R_2")
    w = find_R_inf_witness(**kw)
    assert w is not None and w > kw["R_2"]
    assert certify(G_op=0.05, G_D=0.05, A_inf=0.0, B_2=0.05, B_inf=2.0,
                   R_2=1.0, R_inf=w, Lambda_off=0.0)


# ---------------------------------------------------------------------------
# Phase 2: explicit curvature-Lipschitz constant gamma (Assumption 1)
# ---------------------------------------------------------------------------

from tests.certifier_fixtures import LOGISTIC_GAMMA_MAX  # noqa: E402

GAMMAS = (LOGISTIC_GAMMA_MAX, 0.37, 1.0)


def _raw_case(seed, k=5, sf=5.0):
    """Raw certify() inputs + branches from a prepared campaign-style case."""
    X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(seed, k=k)
    n, d = X.shape
    R_2 = default_R2(k, d, n, safety_factor=sf)
    c = prepared["constants"]
    S_2 = prepared["A_2"] + R_2
    branches = (
        (c["C_O_full_eff"] * S_2, c["R_G_full_eff"]),
        (c["q_max_eff"] * S_2, c["R_Delta_eff"]),
    )
    return dict(G_op=c["C_op_eff"], G_D=c["C_D_eff"], A_inf=prepared["A_inf"],
                B_2=prepared["B_2"], B_inf=prepared["B_inf"], R_2=R_2,
                branches=branches, prepared=prepared,
                data=(X, y, X_ret, y_ret, T, lam, theta_1))


def test_gamma_default_is_identity():
    """gamma=1.0 must reproduce the coefficient-1 predicate bit-for-bit."""
    for seed in range(6):
        case = _raw_case(seed)
        for x in np.geomspace(1e-6, 1e3, 40):
            lam_off = _lambda_off_at(case["branches"], case["A_inf"], x)
            base = certify(G_op=case["G_op"], G_D=case["G_D"],
                           A_inf=case["A_inf"], B_2=case["B_2"],
                           B_inf=case["B_inf"], R_2=case["R_2"], R_inf=x,
                           Lambda_off=lam_off)
            withg = certify(G_op=case["G_op"], G_D=case["G_D"],
                            A_inf=case["A_inf"], B_2=case["B_2"],
                            B_inf=case["B_inf"], R_2=case["R_2"], R_inf=x,
                            Lambda_off=lam_off, gamma=1.0)
            assert base == withg


@pytest.mark.parametrize("gamma", GAMMAS)
def test_gamma_equivalence_oracle_certify(gamma):
    """certify(gamma=g, raw) == certify(gamma=1, pre-scaled constants).

    The pre-scaled gamma=1 form IS the restated predicate by definition, so
    this catches any wrong gamma placement (e.g. on B_2/R_2) and any
    double-scaling, at every R_inf.
    """
    for seed in range(8):
        case = _raw_case(seed)
        for x in np.geomspace(1e-6, 1e3, 60):
            lam_off = _lambda_off_at(case["branches"], case["A_inf"], x)
            got = certify(G_op=case["G_op"], G_D=case["G_D"],
                          A_inf=case["A_inf"], B_2=case["B_2"],
                          B_inf=case["B_inf"], R_2=case["R_2"], R_inf=x,
                          Lambda_off=lam_off, gamma=gamma)
            want = certify(G_op=gamma * case["G_op"], G_D=gamma * case["G_D"],
                           A_inf=case["A_inf"], B_2=case["B_2"],
                           B_inf=case["B_inf"], R_2=case["R_2"], R_inf=x,
                           Lambda_off=gamma * lam_off)
            assert got == want, f"seed={seed} gamma={gamma} R_inf={x}"


@pytest.mark.parametrize("gamma", GAMMAS)
def test_gamma_equivalence_oracle_intervals(gamma):
    """feasible intervals under gamma == intervals of pre-scaled constants."""
    for seed in range(8):
        case = _raw_case(seed)
        got_iv, got_U = feasible_R_inf_intervals(
            G_op=case["G_op"], G_D=case["G_D"], A_inf=case["A_inf"],
            B_2=case["B_2"], B_inf=case["B_inf"], R_2=case["R_2"],
            lambda_off_branches=case["branches"], gamma=gamma)
        scaled_branches = tuple(
            (gamma * c0, c1 if np.isinf(c1) else gamma * c1)
            for c0, c1 in case["branches"])
        want_iv, want_U = feasible_R_inf_intervals(
            G_op=gamma * case["G_op"], G_D=gamma * case["G_D"],
            A_inf=case["A_inf"], B_2=case["B_2"], B_inf=case["B_inf"],
            R_2=case["R_2"], lambda_off_branches=scaled_branches)
        assert np.isclose(got_U, want_U, rtol=1e-12), f"seed={seed}"
        assert len(got_iv) == len(want_iv), f"seed={seed}"
        for (a, b), (a2, b2) in zip(got_iv, want_iv):
            assert np.isclose(a, a2, rtol=1e-9) and np.isclose(b, b2, rtol=1e-9)


def test_gamma_accept_superset_of_gamma1_accept():
    """The gamma-predicate is monotone: every gamma=1 accept stays accepted."""
    n_flip = 0
    for seed, k, sf in SEED_GRID:
        X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(
            seed, k=k)
        n, d = X.shape
        R_2 = default_R2(k, d, n, safety_factor=sf)
        base = certify_at_radius(prepared, R_2)
        withg = certify_at_radius(prepared, R_2, gamma=LOGISTIC_GAMMA_MAX)
        if base["accepted"]:
            assert withg["accepted"], f"gamma lost an accept: {seed},{k},{sf}"
        if withg["accepted"] and not base["accepted"]:
            n_flip += 1
    assert n_flip > 0, "gamma recovered nothing on the seeded grid"


def test_gamma_accepts_are_sound_against_ground_truth():
    """Every gamma-accepted query satisfies the released L2 claim vs the
    Newton-polished exact retrain — the test that defends Phase 2."""
    n_accepted = 0
    for seed in range(20):
        for sf in (5.0, 50.0):
            X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(
                seed, k=5)
            n, d = X.shape
            R_2 = default_R2(5, d, n, safety_factor=sf)
            result = certify_at_radius(prepared, R_2,
                                       gamma=LOGISTIC_GAMMA_MAX)
            if not result["accepted"]:
                continue
            n_accepted += 1
            theta_star = retrain_exact(X_ret, y_ret, lam, d, n=n)
            theta_star = _newton_polish(X_ret, y_ret, lam, n, theta_star)
            err = float(np.linalg.norm(X_ret @ (theta_star - theta_1)))
            assert err <= R_2, (
                f"UNSOUND gamma accept: seed={seed} sf={sf} "
                f"err={err:.3e} > R_2={R_2:.3e}")
    assert n_accepted >= 10, f"only {n_accepted} gamma accepts on grid"


def test_gamma_rejects_when_B2_exceeds_R2():
    """B_2 >= R_2 must reject at every gamma (kills gamma-on-B2/R2 bugs)."""
    assert not certify(G_op=0.5, G_D=0.4, A_inf=0.1, B_2=1.0, B_inf=0.1,
                       R_2=1.0, R_inf=1.0, Lambda_off=0.2,
                       gamma=LOGISTIC_GAMMA_MAX)
    assert find_R_inf_witness(
        G_op=0.5, G_D=0.4, A_inf=0.1, B_2=1.0, B_inf=0.1, R_2=1.0,
        lambda_off_branches=((0.2, np.inf),),
        gamma=LOGISTIC_GAMMA_MAX) is None


def test_gamma_witness_consistent_with_independent_lambda_off():
    """A returned witness must pass certify() with Lambda_off computed
    INDEPENDENTLY from the raw branches (a double-scaled self-check would
    hide a gamma^2 path)."""
    n_checked = 0
    for seed in range(12):
        case = _raw_case(seed)
        w = find_R_inf_witness(
            G_op=case["G_op"], G_D=case["G_D"], A_inf=case["A_inf"],
            B_2=case["B_2"], B_inf=case["B_inf"], R_2=case["R_2"],
            lambda_off_branches=case["branches"], gamma=LOGISTIC_GAMMA_MAX)
        if w is None:
            continue
        n_checked += 1
        lam_off_raw = _lambda_off_at(case["branches"], case["A_inf"], w)
        assert certify(G_op=case["G_op"], G_D=case["G_D"],
                       A_inf=case["A_inf"], B_2=case["B_2"],
                       B_inf=case["B_inf"], R_2=case["R_2"], R_inf=w,
                       Lambda_off=lam_off_raw, gamma=LOGISTIC_GAMMA_MAX)
    assert n_checked >= 5


def test_gamma_trace_matches_and_records():
    """Trace under gamma: certify_matches True, gamma recorded, and a
    gamma-flipped case shows C1 passed in ITS trace."""
    found_flip = False
    for seed, k, sf in SEED_GRID:
        X, y, X_ret, y_ret, T, lam, theta_1, prepared = _prepared_case(
            seed, k=k)
        n, d = X.shape
        R_2 = default_R2(k, d, n, safety_factor=sf)
        base = certify_at_radius(prepared, R_2)
        withg = certify_at_radius(prepared, R_2, gamma=LOGISTIC_GAMMA_MAX)
        assert withg["certify_trace"]["certify_matches"] is True
        assert withg["quantities"]["gamma"] == LOGISTIC_GAMMA_MAX
        assert withg["certify_trace"]["inputs"]["gamma"] == LOGISTIC_GAMMA_MAX
        if withg["accepted"] and not base["accepted"] and not found_flip:
            conds = withg["certify_trace"]["conditions"]
            assert conds[0]["name"] == "G_op * S_inf < 1"
            assert conds[0]["passed"] is True
            found_flip = True
    assert found_flip, "no gamma-flipped case found to check the trace on"


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_gamma_domain_validation(bad):
    """gamma outside (0, 1] must raise, not silently change the predicate."""
    with pytest.raises(ValueError):
        certify(G_op=0.5, G_D=0.4, A_inf=0.1, B_2=0.1, B_inf=0.1,
                R_2=1.0, R_inf=1.0, Lambda_off=0.2, gamma=bad)
    with pytest.raises(ValueError):
        find_R_inf_witness(
            G_op=0.5, G_D=0.4, A_inf=0.1, B_2=0.1, B_inf=0.1, R_2=1.0,
            lambda_off_branches=((0.2, np.inf),), gamma=bad)


@pytest.mark.parametrize("case_seed", range(4))
def test_gamma_exact_intervals_match_dense_grid(case_seed):
    """Dense-grid cross-validation of the interval algebra UNDER gamma:
    gamma threads as a kwarg to both sides (never pre-scaled here)."""
    case = _raw_case(case_seed, k=5, sf=5.0)
    gamma = LOGISTIC_GAMMA_MAX
    intervals, U = feasible_R_inf_intervals(
        G_op=case["G_op"], G_D=case["G_D"], A_inf=case["A_inf"],
        B_2=case["B_2"], B_inf=case["B_inf"], R_2=case["R_2"],
        lambda_off_branches=case["branches"], gamma=gamma)
    hi = U if np.isfinite(U) and U > 0 else 1.0
    xs = np.concatenate([
        np.geomspace(max(hi * 1e-9, 1e-300), hi * 1.5, 4000),
        np.linspace(0.0, hi * 1.5, 4001)[1:],
    ])
    for x in xs:
        lam_off = _lambda_off_at(case["branches"], case["A_inf"], float(x))
        pred = certify(G_op=case["G_op"], G_D=case["G_D"],
                       A_inf=case["A_inf"], B_2=case["B_2"],
                       B_inf=case["B_inf"], R_2=case["R_2"], R_inf=float(x),
                       Lambda_off=lam_off, gamma=gamma)
        inside = any(a < x < b for a, b in intervals)
        boundary = any(
            abs(x - a) <= 1e-9 * max(1.0, abs(a))
            or abs(x - b) <= 1e-9 * max(1.0, abs(b))
            for a, b in intervals) or abs(x - U) <= 1e-9 * max(1.0, abs(U))
        if not boundary:
            assert pred == inside, f"x={x}: certify={pred} intervals={inside}"
