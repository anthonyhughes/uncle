"""Parity of the shared scientific primitives and frozen random streams."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from uncle import logistic, randomness
from uncle.privacy import Privacy


@pytest.mark.parametrize("epsilon,delta", [(1, 1e-4), (0.5, 0.01), (1, 0.8), (0.1, 1e-8)])
@pytest.mark.parametrize("radius", [0, 0.001, 0.7, 20])
@pytest.mark.migration
def test_analytic_gaussian_calibration_matches_existing(epsilon, delta, radius):
    analytic_gaussian_std = pytest.importorskip("unerring.core.utils").analytic_gaussian_std

    assert Privacy(epsilon, delta).gaussian_std(radius) == analytic_gaussian_std(epsilon, delta, radius)


def test_metrics_use_ce_only_and_preserve_strict_logit_tie_convention():
    theta = np.array([1.0, -1.0])
    X = np.array([[1, 1], [2, 2], [2, 1], [-2, 1]], dtype=float)
    y = np.array([0, 0, 1, 0], dtype=float)
    actual = logistic.metrics(theta, X, y)
    assert actual[1] == 1.0
    assert_allclose(actual[0], np.mean(np.logaddexp(0, X @ theta) - y * (X @ theta)), rtol=0, atol=0)


@pytest.mark.migration
def test_loss_gradient_preserves_original_n_and_linear_perturbation():
    old = pytest.importorskip("unerring.core.guo_certified_removal")

    rng = np.random.default_rng(111)
    X, y = rng.normal(size=(13, 4)), rng.integers(0, 2, 13)
    theta, b = rng.normal(size=4), rng.normal(size=4)
    value, gradient = logistic.loss_gradient(theta, X, y, 0.2, normalization_n=20, linear=b)
    assert_allclose(value, old._logistic_loss_perturbed(theta, X, y, 0.2, b, normalization_n=20), rtol=1e-15)
    assert_allclose(gradient, old._logistic_gradient_perturbed(theta, X, y, 0.2, b, normalization_n=20), rtol=1e-14)


@pytest.mark.parametrize("normalization_n", [None, 40])
@pytest.mark.migration
def test_unpolished_fit_matches_existing_retrain(normalization_n):
    retrain_exact = pytest.importorskip("unerring.core.unerring").retrain_exact

    rng = np.random.default_rng(117)
    X, y = rng.normal(size=(24, 5)), rng.integers(0, 2, 24)
    expected = retrain_exact(X, y, 0.2, 5, n=normalization_n)
    actual = logistic.fit(X, y, 0.2, normalization_n=normalization_n)
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.migration
def test_polished_fit_matches_sekhari_production_fitter():
    fit_warm_polished = pytest.importorskip("scripts.fine_lambda_grid_scan").fit_warm_polished

    rng = np.random.default_rng(119)
    X, y = rng.normal(size=(24, 5)), rng.integers(0, 2, 24)
    expected, _ = fit_warm_polished(X, y, 0.2, np.zeros(5))
    assert_allclose(logistic.fit(X, y, 0.2, polish=True), expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("dataset", ["mnist", "dogfish", "cifar10_catdog_resnet18", "imdb"])
@pytest.mark.migration
def test_trial_and_nested_request_streams_match_production(dataset):
    old = pytest.importorskip("scripts.guo_pareto_sweep")

    seeds = randomness.trial_seeds(20260918, dataset, 3)
    assert_array_equal(seeds, old._make_trial_seeds(20260918, dataset, 3))
    prefixes = np.asarray([randomness.deletion_requests(int(seed), 500, 100, 10) for seed in seeds])
    assert_array_equal(prefixes, old._make_drop_requests(seeds, 500, 100, 10))
    for seed in seeds:
        for k in [1, 3, 10, 100]:
            assert randomness.selected_request(int(seed), k, 10) == old._evaluation_request_index(int(seed), k, 10)


@pytest.mark.migration
def test_uncle_mask_stream_matches_existing_common_radius_stream():
    _shared_mask_rng = pytest.importorskip("scripts.unerring_r2_pareto_sweep")._shared_mask_rng

    assert_array_equal(randomness.mask_rng(1485, 10, 4).standard_normal(27),
                       _shared_mask_rng(1485, 10, 4).standard_normal(27))


@pytest.mark.parametrize("dataset,canonical", [("catdog", "cifar10_catdog_resnet18"), ("imdb", "imdb")])
@pytest.mark.migration
def test_sekhari_noise_stream_matches_active_campaign(dataset, canonical):
    noise_seed = pytest.importorskip("scripts.sekhari_lambda_sweep").noise_seed

    config = dict(master_seed=20260918, dataset=dataset)
    expected = np.random.default_rng(noise_seed(config, 8765, 100)).standard_normal(27)
    assert_array_equal(randomness.sekhari_rng(20260918, canonical, 8765, 100).standard_normal(27), expected)


def test_fresh_master_seed_changes_all_replication_streams():
    first = randomness.trial_seeds(20260918, "dogfish", 100)
    second = randomness.trial_seeds(20260919, "dogfish", 100)
    assert not set(first).intersection(second)


def test_standalone_gradient_matches_finite_differences():
    rng = np.random.default_rng(142)
    X, y = rng.normal(size=(13, 4)), rng.integers(0, 2, 13)
    theta, linear = rng.normal(size=4), rng.normal(size=4)
    _, gradient = logistic.loss_gradient(theta, X, y, 0.2, normalization_n=20, linear=linear)
    directions = np.eye(4) * 1e-5
    numerical = [(logistic.loss_gradient(theta + step, X, y, 0.2, 20, linear)[0]
                  - logistic.loss_gradient(theta - step, X, y, 0.2, 20, linear)[0]) / 2e-5
                 for step in directions]
    assert_allclose(gradient, numerical, rtol=1e-8, atol=1e-10)


def test_standalone_polished_optimizer_stationarity_and_improvement():
    rng = np.random.default_rng(144)
    X, y = rng.normal(size=(24, 5)), rng.integers(0, 2, 24)
    theta = logistic.fit(X, y, 0.2, polish=True)
    loss, gradient = logistic.loss_gradient(theta, X, y, 0.2)
    assert np.linalg.norm(gradient) < 1e-8
    assert loss < logistic.loss_gradient(np.zeros(5), X, y, 0.2)[0]


def test_standalone_gaussian_calibration_satisfies_hockey_stick_equation():
    from scipy.special import ndtr

    epsilon, delta, radius = 1.0, 1e-4, 0.7
    sigma = Privacy(epsilon, delta).gaussian_std(radius)
    actual_delta = ndtr(radius / (2 * sigma) - epsilon * sigma / radius) - np.exp(epsilon) * ndtr(
        -radius / (2 * sigma) - epsilon * sigma / radius)
    assert_allclose(actual_delta, delta, rtol=2e-10, atol=1e-15)
    assert Privacy(epsilon, delta).gaussian_std(0) == 0.0
