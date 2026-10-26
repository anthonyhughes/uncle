import numpy as np

from tests.certifier_fixtures import default_R2, remove_samples, retrain_exact
from uncle.simplified.cached_curvature import CachedCurvature
from uncle.simplified.main_certifier import (
    cached_newton_step,
    certify_at_radius,
    main_certifier,
    prepare_certificate,
)


def _problem(seed: int = 7):
    rng = np.random.default_rng(seed)
    n, d, lam = 80, 6, 0.2
    X = rng.normal(size=(n, d))
    theta_star = rng.normal(size=d)
    probs = 1.0 / (1.0 + np.exp(-(X @ theta_star)))
    y = (rng.random(n) < probs).astype(float)
    theta_0 = retrain_exact(X, y, lam, d)
    cache = CachedCurvature.build(X, y, theta_0, lam)
    T = rng.choice(n, size=5, replace=False).astype(np.intp)
    X_ret, y_ret = remove_samples(X, y, set(T.tolist()))
    return X, y, X_ret, y_ret, T, lam, cache


def test_prepared_radius_evaluation_matches_compatibility_wrapper():
    X, y, X_ret, y_ret, T, lam, cache = _problem()
    theta_1, woodbury = cached_newton_step(
        cache, X[T], y[T], T, return_woodbury=True)
    prepared = prepare_certificate(
        X_ret, y_ret, T, theta_1, lam, cache,
        X_full=X, woodbury=woodbury)

    for safety_factor in (0.5, 2.0, 20.0):
        radius = default_R2(
            T.size, X.shape[1], X.shape[0], safety_factor=safety_factor)
        fast = certify_at_radius(prepared, radius)
        wrapped = main_certifier(
            X_ret, y_ret, T, theta_1, radius, lam, cache,
            X_full=X, woodbury=woodbury)
        assert fast["accepted"] == wrapped["accepted"]
        assert fast["quantities"] == wrapped["quantities"]
        assert fast["certify_trace"]["inputs"] == wrapped["certify_trace"]["inputs"]
        assert fast["certify_trace"]["accepted"] == wrapped["certify_trace"]["accepted"]


def test_cached_newton_can_return_reusable_woodbury_state():
    X, y, _, _, T, _, cache = _problem(seed=11)
    theta_plain = cached_newton_step(cache, X[T], y[T], T)
    theta_shared, woodbury = cached_newton_step(
        cache, X[T], y[T], T, return_woodbury=True)

    np.testing.assert_allclose(theta_shared, theta_plain, rtol=0.0, atol=0.0)
    assert woodbury["K_T_chol"] is not None
    assert woodbury["P_T"].shape == (X.shape[1], T.size)
