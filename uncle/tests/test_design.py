"""Standalone design/statistics checks; execute in the cloud test gate."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from uncle import design, reporting


def raw_rows() -> pd.DataFrame:
    records = []
    for method in ("uncle", "guo", "sekhari"):
        for seed, (ce, acc, success) in enumerate(((.2, .9, .8), (.4, .7, 1.))):
            records.append(dict(method=method, lam=.1, k=1, sigma=.2 if method == "guo" else np.nan,
                geometry_multiplier=.5 if method == "uncle" else np.nan,
                radius_factor=2. if method == "uncle" else np.nan,
                tau=2. if method == "uncle" else np.nan, R2=.03 if method == "uncle" else np.nan,
                seed_index=seed, success_rate=1. if method == "sekhari" else success,
                test_ce=ce, test_accuracy=acc, ce_reference=.1, accuracy_reference=.95,
                n_removal_requests=10))
    return pd.DataFrame(records)


def summary_points() -> pd.DataFrame:
    return pd.DataFrame([
        dict(method=method, k=k, lam=lam, sigma=.1, geometry_multiplier=.5, radius_factor=1.,
             mean_ce_cost=ce, mean_accuracy_drop=acc, mean_success_rate=success,
             mean_test_ce=.1 + ce, mean_test_accuracy=.95 - acc,
             sem_ce_cost=.001, sem_accuracy_drop=.002, sem_success_rate=.01)
        for method in ("uncle", "guo", "sekhari") for k in (1, 100)
        for lam, ce, acc, success in ((.1, .01, .3, .8), (.2, .02, .1, .8), (.3, .03, .4, .7))
    ])


def guo_summary() -> pd.DataFrame:
    return pd.DataFrame([dict(k=k, lam=lam, sigma=sigma, mean_excess_test_loss=lam + sigma,
        success_rate=(lam + sigma) / 2, on_frontier=k == 1)
        for k in (1, 100) for lam in (.001, .01, .1) for sigma in (.01, .1, 1.)])


def sekhari_summary(grid: list[float]) -> pd.DataFrame:
    return pd.DataFrame([dict(dataset="mnist", k=k, lam=lam,
        mean_test_ce=1. + abs(i - (0 if k == 1 else len(grid) - 1)),
        mean_test_accuracy=.8 - abs(i - len(grid) // 2) / (2 * len(grid)))
        for k in (1, 100) for i, lam in enumerate(grid)])


def test_exact_pareto_dominance_preserves_ties_and_improvements():
    assert design.pareto_mask([1., 1., 2., 0., -.1], [.9, .9, .8, .7, .6]).tolist() == [True, True, False, True, True]
    assert design.pareto_mask([], []).tolist() == []
    with pytest.raises(ValueError):
        design.pareto_mask([math.nan], [.5])


def test_point_ids_use_only_actual_method_parameters_and_exclude_derived_tau():
    for row in raw_rows().to_dict("records"):
        original = design.point_id(row)
        assert design.point_id({**row, "tau": 999., "R2": 999., "seed_index": 50}) == original
        relevant = {"uncle": "radius_factor", "guo": "sigma", "sekhari": "lam"}[row["method"]]
        assert design.point_id({**row, relevant: row[relevant] * 2}) != original
        irrelevant = "sigma" if row["method"] != "guo" else "geometry_multiplier"
        assert design.point_id({**row, irrelevant: 999.}) == original


def test_summary_uses_seed_level_sem_and_preserves_reference_costs():
    result = design.summarize_seeds(raw_rows(), expected_seeds=2)
    np.testing.assert_allclose(result.mean_test_ce, .3)
    np.testing.assert_allclose(result.mean_test_accuracy, .8)
    np.testing.assert_allclose(result.mean_ce_cost, .2)
    np.testing.assert_allclose(result.mean_accuracy_drop, .15)
    np.testing.assert_allclose(result.sem_test_ce, .1)
    np.testing.assert_allclose(result.sem_test_accuracy, .1)
    np.testing.assert_allclose(result.loc[result.method.ne("sekhari"), "sem_success_rate"], .1)
    assert result.n_seeds.eq(2).all() and result.n_utility_trials.eq(2).all()
    assert result.loc[result.method.ne("sekhari"), "n_request_trials"].eq(20).all()
    assert result.loc[result.method.eq("sekhari"), "n_request_trials"].eq(0).all()
    sekhari = result[result.method.eq("sekhari")].iloc[0]
    assert sekhari.mean_success_rate == 1. and sekhari.sem_success_rate == 0.
    assert sekhari.mean_retrain_rate == 0.
    pd.testing.assert_frame_equal(result, design.summarize_seeds(raw_rows().sample(frac=1., random_state=9), [0, 1]))


@pytest.mark.parametrize("change", [
    lambda rows: rows.iloc[1:],
    lambda rows: pd.concat([rows, rows.iloc[:1]]),
    lambda rows: rows.assign(test_ce=np.nan),
    lambda rows: rows.assign(success_rate=1.1),
    lambda rows: rows.assign(seed_index=.5),
    lambda rows: rows.assign(n_removal_requests=1),
    lambda rows: rows.assign(accuracy_reference=[.8] + [.95] * (len(rows) - 1)),
    lambda rows: rows.assign(point_id="incorrect"),
])
def test_summary_rejects_missing_duplicated_or_invalid_evidence(change):
    with pytest.raises(ValueError):
        design.summarize_seeds(change(raw_rows()), expected_seeds=2)


def test_single_seed_has_unknown_sem_and_explicit_seed_sets_are_supported():
    rows = raw_rows()
    one = rows[rows.seed_index.eq(0)]
    summary = design.summarize_seeds(one, expected_seeds=[0])
    assert summary.sem_test_ce.isna().all() and summary.sem_success_rate.isna().all()
    shifted = rows.assign(seed_index=rows.seed_index + 10)
    assert design.summarize_seeds(shifted, expected_seeds=[10, 11]).n_seeds.eq(2).all()
    with pytest.raises(ValueError, match="seed set"):
        design.summarize_seeds(shifted, expected_seeds=2)


def test_frontier_union_is_per_method_k_and_independent_of_input_order():
    points = summary_points()
    selected = design.freeze_frontiers(points)
    assert len(selected) == 12
    for _, group in selected.groupby(["method", "k"]):
        assert set(group.lam) == {.1, .2}
        assert group.loc[group.selected_by_ce, "lam"].tolist() == [.1]
        assert group.loc[group.selected_by_accuracy, "lam"].tolist() == [.2]
    pd.testing.assert_frame_equal(selected,
        design.freeze_frontiers(points.sample(frac=1., random_state=5)))
    fresh = selected.copy()
    fresh["mean_ce_cost"], fresh["mean_accuracy_drop"] = fresh.lam, fresh.lam
    marked = design.mark_fresh_frontiers(fresh)
    assert len(marked) == len(selected)
    assert marked.fresh_pareto_ce.sum() == 6 and marked.fresh_pareto_accuracy.sum() == 6
    assert marked.selected_by_accuracy.equals(selected.selected_by_accuracy)


def test_guo_polyline_corner_cutting_preserves_endpoints():
    path = np.array([[0., 0.], [1., 2.]])
    result = design.smooth_polyline(path, iterations=1)
    np.testing.assert_array_equal(result, [[0., 0.], [.25, .5], [.75, 1.5], [1., 2.]])


def test_guo_tube_matches_active_budget_rng_bounds_and_path_contract():
    frame = guo_summary()
    generated, metadata, raw, smooth = design.build_refinement_design(frame, dataset="mnist", master_seed=19)
    repeated, metadata2, raw2, smooth2 = design.build_refinement_design(
        frame.sample(frac=1., random_state=7), dataset="mnist", master_seed=19)
    pd.testing.assert_frame_equal(generated, repeated)
    assert metadata == metadata2
    np.testing.assert_array_equal(raw, raw2)
    np.testing.assert_array_equal(smooth, smooth2)
    assert len(generated) == 27 and metadata["first_stage_parameter_points"] == 9
    assert generated["lambda"].between(.001 * (1 - 1e-14), .1 * (1 + 1e-14)).all()
    assert generated.sigma.between(.01 * (1 - 1e-14), 1. * (1 + 1e-14)).all()
    assert metadata["bounds_policy"].startswith("reject")
    assert "Gaussian" in metadata["sampling"]
    np.testing.assert_allclose(np.log10(generated["lambda"]),
        generated.base_log10_lambda + generated.noise_log10_lambda, atol=1e-14)
    assert generated.dataset_name.eq("mnist").all()


def test_guo_singleton_is_the_existing_fixed_base_gaussian_tube():
    frame = guo_summary().assign(on_frontier=False)
    frame.loc[0, "on_frontier"] = True
    with pytest.raises(ValueError, match="at least two"):
        design.build_refinement_design(frame, dataset="dogfish")
    generated, metadata, raw, smooth = design.build_refinement_design(
        frame, dataset="dogfish", allow_singleton_frontier=True)
    assert metadata["singleton_frontier_used"] is True
    np.testing.assert_array_equal(raw, smooth)
    np.testing.assert_allclose(generated.base_log10_lambda, -3.)
    np.testing.assert_allclose(generated.base_log10_sigma, -2.)
    assert metadata["rejection_rounds"] > 0


def test_guo_summary_adapter_preserves_exact_design_for_an_explicit_budget():
    old = guo_summary()
    current = old.rename(columns={"mean_excess_test_loss": "mean_ce_cost", "success_rate": "mean_success_rate"})
    current["method"] = "guo"
    got, metadata, _, _ = design.guo_refinement_design(current, dataset="catdog", count=37, master_seed=23)
    expected, old_metadata, _, _ = design.build_refinement_design(old, dataset="catdog",
        budget_multiplier=37 / 9, master_seed=23, allow_singleton_frontier=True)
    pd.testing.assert_frame_equal(got, expected)
    assert metadata == old_metadata and len(got) == 37


def test_sekhari_coarse_budget_retains_anchors_without_expanding_guo_bounds():
    config = dict(guo_lambdas=[.001, .01, 1.],
                  reference_lambdas=dict(test_ce=.00231, test_accuracy=.017))
    coarse = design.coarse_lambdas(config)
    assert len(coarse) == len(set(coarse)) == 32
    assert coarse[0] == .001 and coarse[-1] == 1.
    assert {config["reference_lambdas"][key] for key in ("test_ce", "test_accuracy")} <= set(coarse)
    config["reference_lambdas"] = dict(test_ce=.0001, test_accuracy=10.)
    np.testing.assert_allclose(np.diff(np.log(design.coarse_lambdas(config))), math.log(1000) / 31)


def test_sekhari_refinement_retains_original_all_k_objectives_and_boundary_policy():
    coarse = np.geomspace(.001, 1., 32).tolist()
    summary = sekhari_summary(coarse)
    refined, metadata = design.refinement_lambdas(summary, coarse)
    assert len(refined) == len(set(refined)) == 64
    assert not set(refined).intersection(coarse)
    assert all(coarse[0] < value < coarse[-1] for value in refined)
    assert len(metadata["objective_winners"]) == 4
    assert {row["boundary"] for row in metadata["boundary_winners"]} == {"lower", "upper"}
    assert metadata["range_expanded"] is False
    assert (refined, metadata) == design.refinement_lambdas(summary.sample(frac=1., random_state=3), coarse[::-1])
    summary["mean_test_ce"], summary["mean_test_accuracy"] = .4, .9
    _, tied = design.refinement_lambdas(summary, coarse)
    assert all(row["lam"] == coarse[0] for row in tied["objective_winners"])


def test_report_thresholds_choose_requested_points_and_preserve_infeasible_gaps():
    frame = design.freeze_frontiers(summary_points())
    frame.loc[frame.lam.eq(.2) & frame.k.eq(1), "mean_success_rate"] = .9
    selected = reporting.summary_points(frame)
    assert selected.loc[selected.k.eq(1), "accuracy_drop"].eq(.1).all()
    assert selected.loc[selected.k.eq(100), "accuracy_drop"].isna().all()
    np.testing.assert_allclose(selected.retrain_rate, .2)
    assert selected.retrain_point_id.str.len().gt(0).all()


def test_reporting_emits_only_simple_plots_and_explicit_sem_metadata(tmp_path):
    summary = design.summarize_seeds(raw_rows(), expected_seeds=2)
    paths = reporting.plot_pareto(summary, tmp_path, dataset="mnist")
    paths += reporting.plot_summary(summary, tmp_path, dataset="mnist")
    assert len(paths) == 4 and all(path.is_file() for path in paths)
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    metadata = json.loads((tmp_path / "mnist_summary.json").read_text())
    assert "one SEM" in metadata["uncertainty"] and "by construction" in metadata["sekhari"]
    assert metadata["success_threshold"] == .9 and metadata["ce_cost_threshold"] == .01
