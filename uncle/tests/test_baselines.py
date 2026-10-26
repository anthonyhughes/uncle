"""Numerical migration checks; execute on GCP, not a laptop experiment worker.

Historical imports are deliberate test oracles. The standalone package itself
must not import them. These tests cover the active scientific paths only.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from uncle.guo import GuoGeometry, GuoModel
from uncle.sekhari import SekhariGeometry, SekhariModel, noise_scale


def data(n=36, d=7):
    rng = np.random.default_rng(8423)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, size=n).astype(float)
    return X, y


@pytest.mark.migration
def test_guo_training_matches_production_objective_perturbation():
    guo_train_with_perturbation = pytest.importorskip(
        "unerring.core.guo_certified_removal").guo_train_with_perturbation

    X, y = data()
    warm = np.linspace(-0.2, 0.3, X.shape[1])
    actual = GuoModel.build(X, y, 0.13, 1.7, 947, warm_start=warm)
    theta, b = guo_train_with_perturbation(X, y, 0.13, 1.7, seed=947, warm_start=warm)
    assert_allclose(actual.perturbation, b, rtol=0, atol=0)
    assert_allclose(actual.theta, theta, rtol=0, atol=2e-14)


@pytest.mark.migration
def test_guo_request_spectral_cache_preserves_twenty_step_policy():
    guo_precompute_request_spectral_norms = pytest.importorskip(
        "unerring.core.guo_certified_removal").guo_precompute_request_spectral_norms

    X, _ = data()
    prefixes = np.array([[0, 2, 7, 3], [4, 8, 6, 1]])
    geometry = GuoGeometry.build(X)
    actual = [[geometry.spectral_norm(row[:k]) for row in prefixes] for k in [1, 4]]
    expected = guo_precompute_request_spectral_norms(X, prefixes, [1, 4])
    assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n,d", [(36, 7), (9, 15)])
@pytest.mark.migration
def test_guo_single_and_batched_updates_and_bounds_match_production(n, d):
    original_guo = pytest.importorskip("unerring.core.guo_certified_removal")
    guo_batched_newton_removal = original_guo.guo_batched_newton_removal
    guo_certification_diagnostics_from_norms = original_guo.guo_certification_diagnostics_from_norms

    X, y = data(n, d)
    prefixes = np.array([[0, 2, 7, 3], [4, 8, 6, 1]])
    sizes = [1, 3, 4]
    geometry = GuoGeometry.build(X)
    model = GuoModel.build(X, y, 0.13, 1.7, 947, geometry=geometry)
    spectra = np.array([[geometry.spectral_norm(row[:k]) for row in prefixes] for k in sizes])
    requests = model.prepare_requests(prefixes, sizes, spectra)
    expected = guo_batched_newton_removal(model.theta, X, y, prefixes, sizes, 0.13)
    for ki, k in enumerate(sizes):
        for ri, prefix in enumerate(prefixes):
            request = requests[ki][ri]
            single = model.prepare_request(prefix[:k], spectra[ki, ri])
            assert_allclose(request.theta_newton, expected[ki, ri], rtol=1e-12, atol=1e-14)
            assert_allclose(single.theta_newton, request.theta_newton, rtol=1e-12, atol=1e-14)
            change = expected[ki, ri] - model.theta
            prediction = X @ change
            prediction_sq = prediction @ prediction - prediction[prefix[:k]] @ prediction[prefix[:k]]
            original = guo_certification_diagnostics_from_norms(
                change, prediction_norm=float(np.sqrt(max(0, prediction_sq))),
                sqrt_spectral_norm=spectra[ki, ri], sigma=1.7, epsilon=1, delta=1e-4,
            )
            accepted, beta, threshold = request.certify(1, 1e-4)
            assert_allclose(beta, original["beta"], rtol=1e-11, atol=1e-15)
            assert threshold == original["threshold"]
            assert accepted == original["passed"]


def test_guo_woodbury_matches_fixed_original_n_dense_hessian():
    X, y = data()
    model = GuoModel.build(X, y, 0.13, 1.7, 947)
    indices = np.array([1, 4, 9, 12])
    retained = np.ones(len(X), dtype=bool)
    retained[indices] = False
    p = expit(X @ model.theta)
    weights = p * (1 - p)
    hessian = (X[retained].T * weights[retained]) @ X[retained] / len(X) + 0.13 * np.eye(X.shape[1])
    rhs = X[indices].T @ (p[indices] - y[indices]) / len(X)
    dense = model.theta + np.linalg.solve(hessian, rhs)
    request = model.prepare_request(indices)
    assert_allclose(request.theta_newton, dense, rtol=1e-12, atol=1e-14)
    # There is no +k*ridge*theta term in Guo's fixed-original-n objective.


@pytest.mark.migration
def test_guo_rejected_release_keeps_same_noise_and_original_n_objective():
    guo_retrain_perturbed = pytest.importorskip(
        "unerring.core.guo_certified_removal").guo_retrain_perturbed

    X, y = data()
    model = GuoModel.build(X, y, 0.13, 1.7, 947)
    indices = np.array([1, 4, 9, 12])
    request = model.prepare_request(indices)
    result = request.release(1e-16, 1e-4)
    assert not result.accepted
    expected = guo_retrain_perturbed(
        X, y, indices, 0.13, model.perturbation, warm_start=request.theta_newton,
    )
    assert_allclose(result.theta, expected, rtol=0, atol=2e-14)


def test_guo_accepted_release_does_not_add_output_noise():
    X, y = data()
    model = GuoModel.build(X, y, 0.13, 10, 947)
    request = model.prepare_request(np.array([1]))
    result = request.release(1, 1e-4)
    assert result.accepted
    assert_allclose(result.theta, request.theta_newton, rtol=0, atol=0)


@pytest.mark.parametrize("n,d", [(20, 5), (5, 9)])
@pytest.mark.migration
def test_sekhari_geometry_matches_original_exact_spectral_targets(n, d):
    full_data_hessian_lipschitz_bounds = pytest.importorskip(
        "unerring.core.hessian_lipschitz").full_data_hessian_lipschitz_bounds

    X, _ = data(n, d)
    actual = SekhariGeometry.build(X)
    expected = full_data_hessian_lipschitz_bounds(X)
    assert actual.gram_op == expected.gram_op
    assert actual.squared_gram_op == expected.hadamard_gram_op
    assert actual.M_sum == expected.M_sum
    assert actual.loss_lipschitz == expected.data_loss_lipschitz


@pytest.mark.migration
def test_sekhari_newton_noise_and_scale_match_original():
    full_data_hessian_lipschitz_bounds = pytest.importorskip(
        "unerring.core.hessian_lipschitz").full_data_hessian_lipschitz_bounds
    SekhariLogisticCache = pytest.importorskip("unerring.core.sekhari_removal").SekhariLogisticCache

    X, y = data()
    theta = np.linspace(-0.2, 0.3, X.shape[1])
    original_geometry = full_data_hessian_lipschitz_bounds(X)
    old = SekhariLogisticCache.build(
        X, y, theta, 0.13, loss_lipschitz=original_geometry.data_loss_lipschitz,
        geometry=original_geometry,
    )
    new = SekhariModel.build(X, y, theta, 0.13, SekhariGeometry.build(X))
    indices = np.array([1, 4, 9, 12])
    actual = new.remove(indices, 1, 1e-4, np.random.default_rng(672))
    expected = old.remove(X[indices], y[indices], epsilon=1, delta=1e-4, rng=np.random.default_rng(672))
    assert_allclose(actual.theta_newton, expected.theta_newton, rtol=0, atol=0)
    assert_allclose(actual.theta, expected.theta_noisy, rtol=0, atol=0)
    assert_allclose(actual.noise, expected.noise, rtol=0, atol=0)
    assert actual.sigma == expected.sigma
    assert actual.sensitivity == expected.sensitivity
    assert new.full_gradient_norm == old.full_gradient_norm


def test_sekhari_retained_mean_normalization_and_per_k_factor_reuse():
    X, y = data()
    theta = np.linspace(-0.2, 0.3, X.shape[1])
    ridge = 0.13
    model = SekhariModel.build(X, y, theta, ridge)
    indices = np.array([1, 4, 9, 12])
    p = expit(X @ theta)
    retained = np.ones(len(X), dtype=bool)
    retained[indices] = False
    hessian_sum = (X[retained].T * (p[retained] * (1 - p[retained]))) @ X[retained]
    hessian_sum += (len(X) - len(indices)) * ridge * np.eye(X.shape[1])
    rhs = X[indices].T @ (p[indices] - y[indices]) + len(indices) * ridge * theta
    expected = theta + np.linalg.solve(hessian_sum, rhs)
    assert_allclose(model.newton_step(indices), expected, rtol=1e-12, atol=1e-14)
    factor = model.factors[4]
    model.newton_step(np.array([0, 2, 3, 7]))
    assert model.factors[4] is factor


def test_sekhari_noise_is_isotropic_prescribed_and_not_tuned():
    X, y = data()
    model = SekhariModel.build(X, y, np.zeros(X.shape[1]), 0.13)
    result = model.remove(np.array([1, 3]), 1, 1e-4, np.random.default_rng(38))
    assert_allclose(result.noise, result.sigma * np.random.default_rng(38).standard_normal(X.shape[1]),
                    rtol=0, atol=0)
    sensitivity, sigma = noise_scale(len(X), 2, 0.13, model.geometry.M_sum,
                                     model.geometry.loss_lipschitz, 1, 1e-4)
    assert sensitivity == result.sensitivity
    assert sigma == result.sigma
    assert not hasattr(result, "accepted")  # Algorithm 1 has no pass/retrain predicate.


@pytest.mark.parametrize("epsilon,delta", [(0, 1e-4), (2, 1e-4), (1, 0), (1, 1), (np.nan, 1e-4)])
def test_sekhari_rejects_invalid_noise_budget(epsilon, delta):
    with pytest.raises(ValueError):
        noise_scale(20, 2, 0.1, 2, 3, epsilon, delta)


def test_geometry_reuse_rejects_different_data_with_same_shape():
    X, y = data()
    with pytest.raises(ValueError, match="different feature"):
        GuoModel.build(X + 0.1, y, 0.13, 1.7, 947, geometry=GuoGeometry.build(X))
    with pytest.raises(ValueError, match="different feature"):
        SekhariModel.build(X + 0.1, y, np.zeros(X.shape[1]), 0.13, SekhariGeometry.build(X))


@pytest.mark.parametrize("indices", [np.array([1, 1]), np.array([1.2]), np.array([-1]), np.array([36])])
def test_both_baselines_reject_ambiguous_requests(indices):
    X, y = data()
    guo = GuoModel.build(X, y, 0.13, 1.7, 947)
    sekhari = SekhariModel.build(X, y, np.zeros(X.shape[1]), 0.13)
    with pytest.raises(ValueError):
        guo.prepare_request(indices)
    with pytest.raises(ValueError):
        sekhari.newton_step(indices)
