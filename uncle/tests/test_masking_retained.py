"""Covariance, certificate-to-release binding, and noise-stream migration tests.

The covariance oracle evaluates every basis vector of the actual sampler's
Gaussian input map. It is deterministic algebra, not a Monte Carlo test.
Run on the cloud gate; this module does not write campaign results.
"""

import math

import numpy as np
import pytest
from scipy.special import ndtr

from uncle.masking import AugmentedGeometry, PreparedRelease
from uncle.privacy import Privacy
from uncle.retained import UncleModel


class FixedNormalInputs:
    """Supply a chosen vector in place of the d+k independent N(0,1) inputs."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.cursor = 0

    def standard_normal(self, size):
        result = self.values[self.cursor:self.cursor + size]
        self.cursor += size
        assert len(result) == size
        return result


def design(shape, kind="full"):
    X = np.random.default_rng(109).normal(size=shape)
    if kind == "zero":
        X[:] = 0
    elif kind == "dependent":
        X[:, -1], X[:, -2] = 0, X[:, 0]
    return X


@pytest.mark.parametrize("shape,k", [((17, 5), 0), ((17, 5), 2), ((9, 14), 3), ((107, 9), 100)])
@pytest.mark.parametrize("kind", ["full", "zero", "dependent"])
@pytest.mark.parametrize("tau", [.01, 2.0])
def test_actual_sampler_linear_map_has_exact_retained_inverse_covariance(shape, k, kind, tau):
    X = design(shape, kind)
    indices = np.arange(k, dtype=int)
    geometry = AugmentedGeometry.build(X, tau).prepare(indices)
    d = shape[1]
    eye = np.eye(d+k)
    columns = []
    for column in eye.T:
        source = FixedNormalInputs(column)
        columns.append(geometry.draw(source))
        assert source.cursor == d+k
    linear_map = np.column_stack(columns)
    covariance = linear_map @ linear_map.T
    Z = X[k:]
    precision = Z.T @ Z + tau * np.eye(d)
    expected = np.linalg.solve(precision, np.eye(d))
    np.testing.assert_allclose(covariance, expected, rtol=2e-9, atol=2e-10)
    # Q^(1/2) is d-dimensional; the longer [Z;sqrt(tau)I] noise is NOT isotropic.
    lower = np.linalg.cholesky(precision)
    np.testing.assert_allclose(lower.T @ covariance @ lower, np.eye(d), rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("tau", [0., -1., np.nan, np.inf, -np.inf])
def test_nonpositive_or_nonfinite_tau_is_rejected(tau):
    with pytest.raises(ValueError, match="ridge"):
        AugmentedGeometry.build(np.eye(3), tau)


@pytest.mark.parametrize("indices", [[0, 0], [-1], [8], [0.5], list(range(8))])
def test_invalid_deletion_identity_is_rejected(indices):
    full = AugmentedGeometry.build(design((8, 3)), .2)
    with pytest.raises(ValueError, match="deleted"):
        full.prepare(np.asarray(indices))


def test_full_factor_data_and_indices_cannot_be_mutated_accidentally():
    X = design((8, 3))
    full = AugmentedGeometry.build(X, .2)
    geometry = full.prepare(np.array([1, 4]))
    for array in (full.X, full.cholesky, geometry.indices):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 1.


def test_mask_preparation_is_lazy_reused_and_does_not_change_tau(monkeypatch):
    full = AugmentedGeometry.build(design((8, 3)), .2)
    calls = []
    original = np.linalg.cholesky
    def count(matrix):
        calls.append(matrix.shape)
        return original(matrix)
    monkeypatch.setattr(np.linalg, "cholesky", count)
    geometry = full.prepare(np.array([1, 4]))
    assert not calls
    geometry.draw(np.random.default_rng(1))
    geometry.draw(np.random.default_rng(2))
    assert calls == [(2, 2)]
    assert geometry.tau == .2
    assert geometry.full is full


def test_factor_failure_stops_without_a_jitter_or_covariance_fallback(monkeypatch):
    calls = []
    def fail(matrix):
        calls.append(matrix.copy())
        raise np.linalg.LinAlgError("deliberate numerical failure")
    monkeypatch.setattr(np.linalg, "cholesky", fail)
    X = design((8, 3))
    with pytest.raises(np.linalg.LinAlgError):
        AugmentedGeometry.build(X, .2)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], X.T @ X + .2 * np.eye(3))


def nullspace_request(nonzero=False):
    # The nonzero case deliberately has a residual in a feature-null direction,
    # so C3 must reject a small R2 even though the raw prediction norm is zero.
    theta = np.array([1., 0., 0.]) if nonzero else np.zeros(3)
    model = UncleModel.build(np.zeros((7, 3)), np.zeros(7), theta, .5)
    return model.prepare_request(np.array([1, 4]), (.1, .01))


def test_release_uses_exact_certificate_geometry_candidate_and_analytic_scale():
    request = nullspace_request()
    geometry = AugmentedGeometry.build(request.model.X, 2.).prepare(request.indices)
    expected = geometry.draw(np.random.default_rng(72))
    prepared = PreparedRelease.build(request, geometry, np.random.default_rng(72))
    privacy = Privacy(1., 1e-4)
    for radius in (.1, .2):
        output = prepared.release(radius, privacy)
        assert output.certificate.accepted
        assert output.certificate.geometry is prepared.geometry is geometry
        assert output.noise_std == privacy.gaussian_std(radius)
        np.testing.assert_array_equal(output.theta, request.candidate + output.noise_std * expected)
    assert "retrained" not in prepared.__dict__


def test_prepared_release_geometry_cannot_be_rebound():
    from dataclasses import FrozenInstanceError
    request = nullspace_request()
    geometry = AugmentedGeometry.build(request.model.X, 2.).prepare(request.indices)
    prepared = PreparedRelease.build(request, geometry, np.random.default_rng(72))
    with pytest.raises(FrozenInstanceError):
        prepared.geometry = AugmentedGeometry.build(request.model.X, 3.).prepare(request.indices)


def test_rejected_release_retrains_original_n_once_and_still_adds_matching_noise(monkeypatch):
    request = nullspace_request(nonzero=True)
    geometry = AugmentedGeometry.build(request.model.X, 1.).prepare(request.indices)
    prepared = PreparedRelease.build(request, geometry, np.random.default_rng(72))
    fitted = np.array([.2, .3, .4])
    calls = []
    def fit(X, y, ridge, normalization_n=None):
        calls.append((X.shape, y.shape, ridge, normalization_n))
        return fitted
    monkeypatch.setattr("uncle.masking.fit", fit)
    privacy = Privacy(1., 1e-4)
    for radius in (.1, .2):
        output = prepared.release(radius, privacy)
        assert not output.certificate.accepted
        np.testing.assert_array_equal(output.theta, fitted + output.noise_std * prepared.direction)
    assert calls == [((5, 3), (5,), .5, 7)]


def test_supplied_fallback_is_shared_across_tau_but_noise_is_not(monkeypatch):
    request = nullspace_request(nonzero=True)
    fallback = np.array([.1, .2, .3])
    def forbidden(*args, **kwargs):
        raise AssertionError("supplied fallback was refitted")
    monkeypatch.setattr("uncle.masking.fit", forbidden)
    outputs = []
    for tau in (.5, 2.):
        geometry = AugmentedGeometry.build(request.model.X, tau).prepare(request.indices)
        prepared = PreparedRelease.build(request, geometry, np.random.default_rng(72), fallback_theta=fallback)
        output = prepared.release(.1, Privacy(1., 1e-4))
        assert not output.certificate.accepted
        np.testing.assert_array_equal(output.theta, fallback + output.noise_std * prepared.direction)
        outputs.append(output)
    assert not np.array_equal(outputs[0].theta, outputs[1].theta)


def test_release_guard_rejects_different_data_or_deleted_rows():
    request = nullspace_request()
    copied = AugmentedGeometry.build(request.model.X.copy(), .2).prepare(request.indices)
    different_rows = AugmentedGeometry.build(request.model.X, .2).prepare(np.array([0, 4]))
    for geometry in (copied, different_rows):
        with pytest.raises(ValueError, match="same full design"):
            PreparedRelease.build(request, geometry, np.random.default_rng(1))


@pytest.mark.parametrize("fallback", [np.zeros(2), np.array([1., np.nan, 2.])])
def test_invalid_supplied_fallback_fails_before_release(fallback):
    request = nullspace_request()
    geometry = AugmentedGeometry.build(request.model.X, .2).prepare(request.indices)
    with pytest.raises(ValueError, match="fallback"):
        PreparedRelease.build(request, geometry, np.random.default_rng(1), fallback_theta=fallback)


@pytest.mark.parametrize("epsilon,delta", [(1., 1e-4), (.5, 1e-6), (1., .3), (1., .8)])
def test_analytic_gaussian_calibration_satisfies_gaussian_privacy_identity(epsilon, delta):
    privacy = Privacy(epsilon, delta)
    radius = .7
    sigma = privacy.gaussian_std(radius)
    # Exact hockey-stick divergence between equal-variance Gaussians distance R2.
    actual_delta = ndtr(radius/(2*sigma)-epsilon*sigma/radius) - math.exp(epsilon)*ndtr(-radius/(2*sigma)-epsilon*sigma/radius)
    assert actual_delta == pytest.approx(delta, rel=2e-9, abs=2e-13)
    assert privacy.gaussian_std(2*radius) == pytest.approx(2*sigma, rel=2e-15)
    assert privacy.gaussian_std(0) == 0


@pytest.mark.migration
@pytest.mark.parametrize("epsilon,delta", [(1., 1e-4), (.5, 1e-6), (1., .3), (1., .8)])
def test_analytic_gaussian_calibration_matches_production(epsilon, delta):
    old = pytest.importorskip("unerring.core.utils")
    privacy = Privacy(epsilon, delta)
    for radius in (0., 1e-5, .1, 10.):
        assert privacy.gaussian_std(radius) == old.analytic_gaussian_std(epsilon, delta, radius)


@pytest.mark.migration
@pytest.mark.parametrize("shape,indices", [((17, 5), [1, 4]), ((9, 14), [0, 3, 7])])
def test_fixed_seed_draw_matches_existing_augmented_sampler(tmp_path, shape, indices):
    old = pytest.importorskip("scripts.augmented_geometry_masking")
    X = design(shape)
    y = np.zeros(len(X))
    tau = .3
    old.build_factor_bank(X, y, tmp_path / "bank", geometry_ridge=tau, dataset="toy")
    bank = old.AugmentedMaskFactorBank(tmp_path / "bank", geometry_ridge=tau)
    old_sampler = old.AugmentedWoodburyMaskSampler(X, y, bank=bank)
    removed = np.asarray(indices)
    new = AugmentedGeometry.build(X, tau).prepare(removed)
    old_request = old_sampler.prepare(removed, lam=.7)
    for seed in (4, 17, 809):
        np.testing.assert_allclose(new.draw(np.random.default_rng(seed)), old_request.draw(np.random.default_rng(seed)), rtol=2e-12, atol=2e-13)
