"""Opt-in feature-scale smoke against the current production UNCLE.

Set UNCLE_REAL_DATA_ROOT to the frozen real-data campaign on a cloud worker.
Each dataset uses a deterministic 600-row training slice with ALL original
feature columns. Full-data setup exercises the iterative eigensolver; deletion
queries use only the k-dimensional B2 inflation problem.
This checks implementation parity, NOT full-campaign utility equivalence.
"""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from uncle.logistic import fit
from uncle.masking import AugmentedGeometry
from uncle.ab2 import UncleModel


pytestmark = [pytest.mark.real_data, pytest.mark.migration]
LAMBDAS = {
    "mnist": .0013456462345536545,
    "dogfish": .00027114312398124335,
    "catdog": .0019396341886702534,
    "imdb": .0001773149992886697,
}
S_INF = (10., 1., .1, .01, .001)


@pytest.fixture(scope="module", params=list(LAMBDAS))
def real_model(request, tmp_path_factory):
    root = os.environ.get("UNCLE_REAL_DATA_ROOT")
    if not root:
        pytest.skip("set UNCLE_REAL_DATA_ROOT to enable the cloud real-feature smoke")
    if "BATCH_TASK_INDEX" not in os.environ:
        pytest.fail("real-data parity must run on a GCP Batch worker")
    dataset = request.param
    path = Path(root) / dataset / "input" / "dataset.npz"
    with np.load(path, allow_pickle=False) as saved:
        original_n = len(saved["y_train"])
        assert original_n >= 600
        # Different fixed streams per dataset; never select rows using labels.
        seed = 20260918 + list(LAMBDAS).index(dataset)
        rows = np.random.default_rng(seed).choice(original_n, 600, replace=False)
        X = np.asarray(saved["X_train"][rows], dtype=np.float64)
        y = np.asarray(saved["y_train"][rows], dtype=np.float64)
    lam = LAMBDAS[dataset]
    theta = fit(X, y, lam, polish=True)
    model = UncleModel.build(X, y, theta, lam, S_INF)
    # With opt-in enabled a missing production oracle is a failure, not a skip.
    original = importlib.import_module("unerring.core.augmented_geometry_certification")
    old_model = original.prepare_ab2_model(X, y, theta, lam, s_inf_values=S_INF)
    certifier = importlib.import_module("unerring.core.main_certifier")
    old_masking = importlib.import_module("scripts.augmented_geometry_masking")
    geometries = {c: AugmentedGeometry.build(X, c * len(X) * lam) for c in (.001, 1., 10.)}
    bank_path = tmp_path_factory.mktemp(f"{dataset}-augmented-oracle")
    old_masking.build_factor_bank(X, y, bank_path, geometry_ridge=len(X)*lam, dataset=dataset)
    bank = old_masking.AugmentedMaskFactorBank(bank_path, geometry_ridge=len(X)*lam)
    old_sampler = old_masking.AugmentedWoodburyMaskSampler(X, y, bank=bank)
    return dataset, model, original, old_model, certifier, geometries, old_sampler


@pytest.mark.parametrize("k", [10, 100])
def test_real_feature_iterative_certificate_and_masking_parity(real_model, k, record_property):
    dataset, model, original, old_model, certifier, geometries, old_sampler = real_model
    indices = np.random.default_rng(37091).permutation(len(model.X))[:k]
    request = model.prepare_request(indices)
    assert len(model.X) >= 500  # Exercise ARPACK during model initialization.
    old_candidate = certifier.cached_newton_step(
        old_model.cache, model.X[indices], model.y[indices], indices,
    )
    old_request = original.prepare_request(old_model, indices, old_candidate)
    old_states = original.prepare_radius_grid(old_request, S_INF, certificate_mode="a-b2")
    np.testing.assert_allclose(request.candidate, old_candidate, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(request.inverse_gradient, old_request.inverse_gradient, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(request.b, old_request.b, rtol=2e-9, atol=2e-11)
    for new, old in zip(request.states, old_states):
        np.testing.assert_allclose(new.m, old.m, rtol=2e-10, atol=2e-13)
        np.testing.assert_allclose(new.delta, old.delta, rtol=2e-10, atol=2e-13)
        for key, old_key in (("g_op", "G_op"), ("g_mix", "G_mix"),
                             ("g_diag", "G_diag_max"), ("b2_local", "B_2_local"),
                             ("off_l2", "lambda_off_l2_coefficient")):
            assert getattr(new, key) == pytest.approx(getattr(old, old_key), rel=2e-9, abs=2e-11)

    # Use the experiment's radius parametrization but do not tune to outcomes.
    radii = np.geomspace(.001, 1000., 10) * k * model.X.shape[1] / len(model.X)**1.5
    counts = {}
    for c, full_geometry in geometries.items():
        geometry = full_geometry.prepare(indices)
        accepted = 0
        for radius in radii:
            old = original.certify_radius_or(old_states, float(radius), geometry.tau)
            new = request.certify(float(radius), geometry)
            assert new.accepted == old["accepted"]
            assert new.witness_s_inf == old["selected_S_inf"]
            assert new.witness_r_inf == pytest.approx(old["selected_R_inf"], abs=2e-11)
            accepted += new.accepted
            for branch, old_branch in zip(new.branches, old["per_radius"]):
                assert branch.lambda_off == pytest.approx(old_branch["Lambda_off"], rel=2e-9, abs=2e-11)
                assert branch.b2_aug == pytest.approx(old_branch["B_2_aug"], rel=2e-9, abs=2e-11)
                assert branch.g_mix_aug == pytest.approx(old_branch["G_mix_aug"], rel=2e-9, abs=2e-11)
                for check, old_check in zip((branch.c1, branch.c2, branch.c3, branch.c4), old_branch["conditions"]):
                    assert check.passed == old_check["passed"]
                    assert check.lhs == pytest.approx(old_check["lhs"], rel=2e-9, abs=2e-11)
                    assert check.rhs == pytest.approx(old_check["rhs"], abs=2e-11)
        counts[str(c)] = accepted

    # Sample equality tests the implemented triangular orientations and stream,
    # while the smaller deterministic unit tests establish covariance algebra.
    new_mask = geometries[1.].prepare(indices)
    old_mask = old_sampler.prepare(indices, model.training_ridge)
    for seed in (19, 172):
        np.testing.assert_allclose(
            new_mask.draw(np.random.default_rng(seed)),
            old_mask.draw(np.random.default_rng(seed)), rtol=2e-9, atol=2e-10,
        )
    record_property("dataset", dataset)
    record_property("train_slice_shape", str(model.X.shape))
    record_property("k", k)
    record_property("accepted_radii_by_c", str(counts))
