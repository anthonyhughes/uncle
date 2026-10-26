"""Dense oracles and migration checks; execute in the cloud test gate."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.special import expit

from uncle import curvature
from uncle.logistic import fit
from uncle.masking import AugmentedGeometry
from uncle.ab2 import UncleModel


def prepared(kind="full", seed=42, radii=(10.0, 1.0, 0.1)):
    rng = np.random.default_rng(seed)
    n, d = (11, 16) if kind == "wide" else (19, 5)
    X = rng.normal(scale=0.3, size=(n, d))
    if kind == "duplicate-zero":
        X[:, -1], X[:, -2] = 0, X[:, 0]
    elif kind == "zero":
        X[:] = 0
    elif kind == "ill-conditioned":
        X[:, -1] = X[:, 0] + 1e-9 * X[:, -1]
    y, theta = rng.integers(0, 2, n).astype(float), rng.normal(scale=0.04, size=d)
    model = UncleModel.build(X, y, theta, 0.4, radii)
    return model.prepare_request(np.array([1, 4]))


def geometry(request, tau=0.3):
    # Geometry construction does not need a masking draw to audit a certificate.
    return AugmentedGeometry.build(request.model.X, tau).prepare(request.indices)


def dense(request):
    model, R = request.model, request.retained
    Z, removed = model.X[R], model.X[request.indices]
    n, d = model.X.shape
    p0 = expit(Z @ model.theta)
    H = (Z.T * (p0 * (1.0 - p0))) @ Z / n + model.training_ridge * np.eye(d)
    deleted_gradient = removed.T @ (expit(removed @ model.theta) - model.y[request.indices]) / n
    candidate = model.theta + np.linalg.solve(H, deleted_gradient)
    gradient = Z.T @ (expit(Z @ request.candidate) - model.y[R]) / n + model.training_ridge * request.candidate
    v = np.linalg.solve(H, gradient)
    B = Z @ np.linalg.solve(H, Z.T) / n
    return Z, H, candidate, v, B


@pytest.mark.parametrize("kind", ["full", "wide", "duplicate-zero", "zero", "ill-conditioned"])
def test_newton_and_gradient_solves_match_dense_original_n(kind):
    request = prepared(kind)
    Z, H, candidate, v, B = dense(request)
    np.testing.assert_allclose(request.candidate, candidate, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(request.inverse_gradient, v, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(request.b, Z @ v, rtol=3e-12, atol=3e-14)
    assert request.b2 == pytest.approx(np.linalg.norm(Z @ v), abs=3e-14)
    displacement = Z @ (request.candidate - request.model.theta)
    assert request.a2 == pytest.approx(np.linalg.norm(displacement), abs=3e-14)
    if kind != "zero":
        assert not np.allclose(v, v * len(request.model.X) / len(Z))


@pytest.mark.parametrize("kind", ["full", "wide", "duplicate-zero", "zero", "ill-conditioned"])
def test_b2_bounds_dominate_dense_retained_spectral_targets(kind):
    request = prepared(kind)
    Z, H, _, v, B = dense(request)
    for state in request.states:
        if state.r_inf <= 0:
            continue
        D = np.diag(np.sqrt(state.delta))
        expected_op = np.linalg.eigvalsh(D @ B @ D)[-1]
        assert state.g_op >= expected_op - 3e-14
        assert state.g_mix >= np.linalg.norm(B @ D, 2) - 3e-14
        assert state.g_diag >= np.max(state.delta * np.diag(B)) - 3e-14
        summary = next(s for s in request.model.summaries if s.s_inf == state.s_inf)
        assert state.g_op == pytest.approx(request.eta * summary.a_s, rel=3e-14)
        assert state.g_diag == pytest.approx(request.eta * summary.g_s, rel=3e-14)
        assert state.g_mix == pytest.approx(request.eta * math.sqrt(request.model.k0 * summary.a_s), rel=3e-14)
        assert state.b2_local == pytest.approx(np.linalg.norm(D @ (Z @ v)), rel=3e-12, abs=3e-14)


@pytest.mark.parametrize("kind", ["full", "wide", "duplicate-zero", "zero"])
@pytest.mark.parametrize("tau", [1e-6, 0.3, 10.0])
def test_augmented_c3_and_prediction_c4_have_the_prescribed_geometry(kind, tau):
    request = prepared(kind)
    Z, H, _, v, B = dense(request)
    certificate = request.certify(0.7, geometry(request, tau))
    n, d = request.model.X.shape
    J = np.vstack((Z, math.sqrt(tau) * np.eye(d)))
    for state, checks in zip(request.states, certificate.branches):
        if state.r_inf <= 0:
            assert not checks.accepted
            continue
        D = np.diag(np.sqrt(state.delta))
        assert checks.b2_aug == pytest.approx(np.linalg.norm(J @ v), rel=4e-12, abs=3e-14)
        actual_mix = np.linalg.norm(J @ np.linalg.solve(H, Z.T @ D) / n, 2)
        assert checks.g_mix_aug >= actual_mix - 3e-12
        expected_mix = math.sqrt(state.g_mix**2 + tau * state.g_op / (n * request.model.training_ridge))
        assert checks.g_mix_aug == pytest.approx(expected_mix, rel=3e-14, abs=3e-14)
        expected_off = min(state.off_l2 * (request.a2 + 0.7) / state.s_inf, state.g_mix)
        assert checks.lambda_off == pytest.approx(expected_off, rel=3e-14, abs=3e-14)
        assert checks.c1.passed == (state.g_op < 1)
        assert checks.c2.passed == (state.g_diag < 1)


def test_fixed_intervals_cover_theta0_and_derived_candidate_radius():
    request = prepared()
    Z = request.model.X[request.retained]
    a0, a1 = Z @ request.model.theta, Z @ request.candidate
    for state in request.states:
        if state.r_inf <= 0:
            continue
        t = np.linspace(a0-state.s_inf, a0+state.s_inf, 401)
        assert np.all(a1-state.r_inf >= a0-state.s_inf-1e-14)
        assert np.all(a1+state.r_inf <= a0+state.s_inf+1e-14)
        p = expit(t)
        derivative = np.abs(p * (1-p) * (1-2*p))
        assert np.all(np.max(derivative, axis=0) <= state.m)
        np.testing.assert_array_equal(state.delta, np.nextafter(state.s_inf * state.m, np.inf))


def test_nullspace_is_charged_to_c3_not_ignored():
    model = UncleModel.build(np.zeros((7, 3)), np.zeros(7), np.array([1., 0., 0.]), 0.5, (0.1, 0.01))
    request = model.prepare_request(np.array([1]))
    assert request.b2 == 0
    small = request.certify(0.5, geometry(request, 0.01))
    large = request.certify(0.5, geometry(request, 1.0))
    assert small.accepted and not large.accepted
    assert all(branch.c1.passed and branch.c2.passed and branch.c4.passed for branch in large.branches)
    assert all(not branch.c3.passed for branch in large.branches)
    assert large.branches[0].b2_aug == pytest.approx(1.0)


def test_checks_are_strict_and_tau_values_are_never_ORed():
    model = UncleModel.build(np.zeros((7, 3)), np.zeros(7), np.array([1., 0., 0.]), 0.5, (0.1, 0.01))
    request = model.prepare_request(np.array([1]))
    geom = geometry(request, 1.0)
    bound = request.certify(10., geom).branches[0].c3.lhs
    assert not request.certify(bound, geom).accepted
    assert request.certify(np.nextafter(bound, np.inf), geom).accepted
    for key in ("g_op", "g_diag"):
        state = replace(request.states[0], **{key: 1.0})
        assert not replace(request, states=(state,)).certify(10., geom).accepted
    values = [request.certify(.5, geometry(request, tau)).branches[0].c3.lhs for tau in (1e-8, .01, 1., 100.)]
    assert np.all(np.diff(values) > 0)


def test_geometry_mismatch_is_an_error_and_certificate_retains_exact_geometry():
    request = prepared()
    geom = geometry(request)
    assert request.certify(1., geom).geometry is geom
    with pytest.raises(ValueError, match="geometry"):
        request.certify(1., AugmentedGeometry.build(request.model.X.copy(), .3).prepare(request.indices))
    with pytest.raises(ValueError, match="geometry"):
        request.certify(1., AugmentedGeometry.build(request.model.X, .3).prepare(np.array([2, 4])))


def test_radius_and_geometry_scans_do_not_repeat_preparation(monkeypatch):
    request = prepared()
    def forbidden(*args, **kwargs):
        raise AssertionError("radius/tau certification repeated spectral preparation")
    monkeypatch.setattr("uncle.ab2.full_weighted_norm", forbidden)
    monkeypatch.setattr("uncle.ab2.largest_psd_eigenvalue", forbidden)
    for tau in (.01, .1, 1.):
        geom = geometry(request, tau)
        for radius in (.01, .1, 1.):
            certificate = request.certify(radius, geom)
            assert len(certificate.branches) == 3
            assert certificate.accepted == any(b.accepted for b in certificate.branches)


@pytest.mark.parametrize("shape", [(31, 5), (12, 18)])
@pytest.mark.parametrize("tau", [1e-3, 0.3, 10.])
def test_accepted_augmented_distance_against_retained_optimizer(shape, tau):
    rng = np.random.default_rng(87)
    X = rng.normal(scale=.25, size=shape)
    y = rng.integers(0, 2, len(X)).astype(float)
    theta = fit(X, y, .2, polish=True)
    request = UncleModel.build(X, y, theta, .2).prepare_request(np.array([1, 4]))
    Z, labels = X[request.retained], y[request.retained]
    optimum = fit(Z, labels, .2, normalization_n=len(X), polish=True)
    error = request.candidate - optimum
    distance = math.sqrt(np.linalg.norm(Z @ error)**2 + tau * np.linalg.norm(error)**2)
    geom = geometry(request, tau)
    accepted = 0
    for radius in np.geomspace(1e-5, 10., 30):
        if request.certify(radius, geom).accepted:
            accepted += 1
            assert distance < radius * (1.0 + 1e-7)
    assert accepted > 0


@pytest.mark.parametrize("indices", [[1, 1], [-1], [19], [], [0.1]])
def test_invalid_deletions_fail(indices):
    request = prepared()
    with pytest.raises(ValueError):
        request.model.prepare_request(np.asarray(indices))


@pytest.mark.parametrize("radii", [[], [0.1, .1], [0], [-1], [np.inf], [np.nan]])
def test_invalid_radius_grids_fail(radii):
    request = prepared()
    with pytest.raises(ValueError):
        UncleModel.build(request.model.X, request.model.y, request.model.theta, .4, radii)


def test_numerical_policy_uses_dense_then_deterministic_iterative_eigenvalues():
    for n in (7, 501):
        diagonal = np.linspace(1.0, 2.0, n)
        assert curvature.largest_psd_eigenvalue(n, lambda v: diagonal*v) == pytest.approx(2.0, rel=2e-9)


def test_numerical_policy_retains_deterministic_power_fallback(monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["which"] == "LA"
        assert kwargs["tol"] == 1e-10
        np.testing.assert_array_equal(kwargs["v0"], np.full(501, 1 / math.sqrt(501)))
        raise np.linalg.LinAlgError("exercise retained fallback")
    monkeypatch.setattr(curvature, "eigsh", fail)
    diagonal = np.ones(501)
    diagonal[0] = 3.
    assert curvature.largest_psd_eigenvalue(501, lambda v: diagonal*v) == pytest.approx(3., rel=1e-8)


@pytest.mark.migration
def test_migration_fixed_envelope_is_identical_to_production():
    original = pytest.importorskip("unerring.core.local_curvature_certification")
    centers = np.array([-1000., -8., -1.4, -1.317, -.01, 0., 1.3, 8., 1000.])
    errors = np.full(len(centers), 1e-13)
    for s in (10., 1., .1, .01, .001):
        np.testing.assert_array_equal(curvature.fixed_coefficients(centers, errors, s),
            original.local_third_derivative_upper(centers, s, center_error_upper=errors))


@pytest.mark.parametrize("kind", ["full", "wide", "duplicate-zero", "zero", "ill-conditioned"])
@pytest.mark.migration
def test_migration_coefficients_and_decisions_match_production_ab2(kind):
    original = pytest.importorskip("unerring.core.augmented_geometry_certification")
    certifier = pytest.importorskip("unerring.core.main_certifier")
    request = prepared(kind)
    model = request.model
    values = [s.s_inf for s in model.summaries]
    old_model = original.prepare_ab2_model(model.X, model.y, model.theta, model.training_ridge, s_inf_values=values)
    old_candidate = certifier.cached_newton_step(old_model.cache, model.X[request.indices], model.y[request.indices], request.indices)
    np.testing.assert_allclose(request.candidate, old_candidate, rtol=3e-13, atol=3e-15)
    old_request = original.prepare_request(old_model, request.indices, old_candidate)
    old_states = original.prepare_radius_grid(old_request, values, certificate_mode="a-b2")
    for old, new in zip(old_states, request.states):
        np.testing.assert_allclose(new.m, old.m, rtol=3e-13, atol=3e-15)
        for key, old_key in (("g_op", "G_op"), ("g_mix", "G_mix"), ("g_diag", "G_diag_max"), ("b2_local", "B_2_local"), ("off_l2", "lambda_off_l2_coefficient")):
            assert getattr(new, key) == pytest.approx(getattr(old, old_key), rel=3e-12, abs=3e-14)
    for tau in (.001, .1, 10.):
        geom = geometry(request, tau)
        for radius in np.geomspace(1e-5, 10., 15):
            old = original.certify_radius_or(old_states, radius, tau)
            new = request.certify(radius, geom)
            assert new.accepted == old["accepted"]
            assert new.witness_s_inf == old["selected_S_inf"]
            assert new.witness_r_inf == pytest.approx(old["selected_R_inf"], abs=3e-14)
            for branch, old_branch in zip(new.branches, old["per_radius"]):
                for check, old_check in zip((branch.c1, branch.c2, branch.c3, branch.c4), old_branch["conditions"]):
                    assert check.lhs == pytest.approx(old_check["lhs"], rel=3e-12, abs=3e-14)
                    assert check.rhs == pytest.approx(old_check["rhs"], abs=3e-14)
                    assert check.passed == old_check["passed"]


def test_query_only_solves_k_dimensional_eigenproblem(monkeypatch):
    request = prepared()
    eigenvalue = curvature.largest_psd_eigenvalue
    sizes = []

    def small_only(size, matvec):
        sizes.append(size)
        assert size == 2
        return eigenvalue(size, matvec)

    def forbidden(*args):
        raise AssertionError("deletion query tried to initialize a full-data spectrum")

    monkeypatch.setattr("uncle.ab2.full_weighted_norm", forbidden)
    monkeypatch.setattr("uncle.ab2.fixed_coefficients", forbidden)
    monkeypatch.setattr("uncle.ab2.largest_psd_eigenvalue", small_only)
    for indices in ([1, 2], [3, 4]):
        request.model.prepare_request(np.array(indices))
    assert sizes == [2, 2]


def test_invalid_s_branch_is_retained_as_rejection():
    request = prepared(radii=(10., 1e-30))
    checks = request.certify(1., geometry(request)).branches
    assert len(checks) == 2
    assert checks[-1].r_inf < 0
    assert not checks[-1].accepted
    assert not checks[-1].c1.passed


@pytest.mark.parametrize("seed", range(5))
def test_precomputed_full_summaries_match_dense_targets(seed):
    request = prepared(seed=seed)
    model = request.model
    B = model.gram / len(model.X)
    assert model.k0 == pytest.approx(np.linalg.eigvalsh(B)[-1], rel=3e-13)
    assert model.d0 == pytest.approx(np.diag(B).max(), rel=3e-13)
    for summary in model.summaries:
        D = np.diag(np.sqrt(summary.delta))
        assert summary.a_s == pytest.approx(np.linalg.eigvalsh(D @ B @ D)[-1], rel=3e-13)
        assert summary.g_s == pytest.approx(np.max(summary.delta * np.diag(B)), rel=3e-13)
        assert not summary.m.flags.writeable
        assert not summary.delta.flags.writeable
