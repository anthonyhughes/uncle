"""Tiny end-to-end experiment tests. Run these on the GCP test worker."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from uncle import design, experiment, randomness


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(9081)
    X, Xt = rng.normal(size=(24, 3)), rng.normal(size=(18, 3))
    y, yt = rng.integers(0, 2, len(X)), rng.integers(0, 2, len(Xt))
    data_path = tmp_path / "data.npz"
    np.savez(data_path, X_train=X, y_train=y, X_test=Xt, y_test=yt)
    config = dict(
        dataset="catdog", rng_dataset="cifar10_catdog_resnet18", data_npz=str(data_path),
        privacy=dict(epsilon=1.0, delta=1e-4), master_seed=20260918,
        num_seeds=2, removal_requests=2, ks=[1, 3],
        reference_metrics=dict(test_ce=0.5, test_accuracy=0.8),
        reference_lambdas=dict(test_ce=0.2, test_accuracy=0.2),
        uncle_lambdas=[0.2], guo_lambdas=[0.2], sekhari_lambdas=[0.2],
        geometry_multipliers=[0.1, 1.0], safety_factors=[0.2, 2.0],
        sigmas=[1.0], s_inf_values=[1.0, 0.1],
        uncle_variant="theta0-A-B2-augmented-v1",
    )
    config_path, points_path = tmp_path / "config.json", tmp_path / "points.csv"
    config_path.write_text(json.dumps(config))
    points = experiment.initial_points(config, ["uncle", "guo", "sekhari"])
    points.to_csv(points_path, index=False)
    return config, config_path, points_path, points


def read_rows(output):
    return pd.read_csv(output / "runs.csv", float_precision="round_trip").sort_values(
        ["point_id", "seed_index"]
    ).reset_index(drop=True)


def test_three_method_end_to_end_records_matching_geometry_and_budget(inputs, tmp_path):
    config, config_path, points_path, points = inputs
    output = tmp_path / "complete"
    experiment.run(config_path, points_path, output)
    rows = read_rows(output)
    assert len(rows) == len(points) * 2
    assert set(rows.method) == {"uncle", "guo", "sekhari"}
    assert not rows.duplicated(["point_id", "seed_index"]).any()
    assert rows.epsilon.eq(1).all() and rows.delta.eq(1e-4).all()
    assert rows.ce_reference.eq(0.5).all() and rows.accuracy_reference.eq(0.8).all()
    assert np.isfinite(rows[["test_ce", "test_accuracy", "success_rate"]]).all().all()
    assert rows.test_ce.ge(0).all()
    assert rows.test_accuracy.between(0, 1).all() and rows.success_rate.between(0, 1).all()
    for row in rows.itertuples():
        trace = json.loads(row.trace)
        if row.method == "uncle":
            assert trace["tau"] == row.geometry_multiplier * 24 * row.lam
            assert trace["R2"] == row.radius_factor * row.k * 3 / 24 ** 1.5
            assert trace["precision"] == "X_ret.T@X_ret+tau*I"
            assert "inverse(precision)" in trace["covariance"]
            assert len(trace["branches"]) == 2
            assert trace["uncle_variant"] == "theta0-A-B2-augmented-v1"
            assert [b["s_inf"] for b in trace["branches"]] == config["s_inf_values"]
        elif row.method == "guo":
            assert trace["normalization_n"] == 24
            assert trace["noise"] == "objective perturbation; no output masking"
        else:
            assert trace["normalization_n"] == 24 - row.k
            assert trace["covariance"] == "sigma^2*I"
            assert "NOT empirical" in trace["success_kind"]
            assert row.success_rate == 1.0
    # A seed is the independent replication unit, not each request or R2.
    summary = design.summarize_seeds(rows, expected_seeds=2, requests_per_seed=2)
    assert len(summary) == len(points)
    assert summary.n_seeds.eq(2).all() and summary.n_utility_trials.eq(2).all()
    assert summary.loc[summary.method.ne("sekhari"), "n_request_trials"].eq(4).all()
    assert summary.loc[summary.method.eq("sekhari"), "n_request_trials"].eq(0).all()


def test_completion_receipt_and_manifest_bind_outputs_and_inputs(inputs, tmp_path):
    config, config_path, points_path, _ = inputs
    output = tmp_path / "complete"
    experiment.run(config_path, points_path, output)
    complete = json.loads((output / "complete.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert complete["status"] == "pass"
    assert complete["rows"] == len(read_rows(output))
    assert complete["runs_sha256"] == experiment.file_hash(output / "runs.csv")
    assert manifest["config_sha256"] == experiment.file_hash(config_path)
    assert manifest["points_sha256"] == experiment.file_hash(points_path)
    assert manifest["data_sha256"] == experiment.file_hash(Path(config["data_npz"]))
    assert {"guo.py", "sekhari.py", "ab2.py", "masking.py"} <= set(manifest["sources"])
    assert len(manifest["trial_seeds"]) == 2
    with pytest.raises(FileExistsError):
        experiment.run(config_path, points_path, output)


def test_seed_shards_reproduce_unsharded_rows_and_pair_all_methods(inputs, tmp_path):
    config, config_path, points_path, _ = inputs
    full, first, second = (tmp_path / name for name in ("full", "first", "second"))
    experiment.run(config_path, points_path, full)
    experiment.run(config_path, points_path, first, seed_start=0, seed_stop=1)
    experiment.run(config_path, points_path, second, seed_start=1, seed_stop=2)
    combined = pd.concat([read_rows(first), read_rows(second)], ignore_index=True).sort_values(
        ["point_id", "seed_index"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(read_rows(full), combined, check_exact=False, rtol=1e-11, atol=1e-12)
    for (_, k), group in combined.groupby(["seed_index", "k"]):
        assert group.trial_seed.nunique() == 1
        assert group.selected_request_index.nunique() == 1
        seed = int(group.trial_seed.iloc[0])
        assert group.selected_request_index.iloc[0] == randomness.selected_request(seed, int(k), 2)
    expected = randomness.trial_seeds(config["master_seed"], config["rng_dataset"], 2)
    assert set(combined.trial_seed) == set(expected)


def test_guo_rows_index_batches_by_actual_k_not_list_position(inputs, tmp_path):
    _, config_path, points_path, points = inputs
    points[points.method.eq("guo")].to_csv(points_path, index=False)
    output = tmp_path / "guo-only"
    experiment.run(config_path, points_path, output)
    rows = read_rows(output)
    assert set(rows.k) == {1, 3}
    assert len(rows) == 4


def test_failed_shard_has_no_success_receipt(inputs, tmp_path, monkeypatch):
    _, config_path, points_path, points = inputs
    points[points.method.eq("guo")].to_csv(points_path, index=False)

    def fail(*args):
        raise RuntimeError("deliberate test interruption")

    monkeypatch.setattr(experiment, "_guo_rows", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="interruption"):
        experiment.run(config_path, points_path, output)
    assert (output / "manifest.json").exists()
    assert not (output / "complete.json").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_identity", "unknown_method", "wrong_k"])
def test_invalid_point_table_rejected_before_output(inputs, tmp_path, mutation):
    _, config_path, points_path, points = inputs
    if mutation == "duplicate":
        points = pd.concat([points, points.iloc[:1]])
    elif mutation == "wrong_identity":
        points.loc[0, "lam"] *= 2
    elif mutation == "unknown_method":
        points.loc[0, "method"] = "legacy"
    else:
        points.loc[0, "k"] = 4
        points.loc[0, "point_id"] = design.point_id(points.iloc[0])
    points.to_csv(points_path, index=False)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        experiment.run(config_path, points_path, output)
    assert not output.exists()


@pytest.mark.parametrize("start,stop", [(-1, 1), (1, 1), (0, 3), (2, 2)])
def test_invalid_seed_shard_rejected(inputs, tmp_path, start, stop):
    _, config_path, points_path, _ = inputs
    with pytest.raises(ValueError, match="seed shard"):
        experiment.run(config_path, points_path, tmp_path / "invalid", start, stop)


@pytest.mark.parametrize("change", [dict(num_seeds=0), dict(removal_requests=True), dict(ks=[1, 1]),
                                    dict(privacy=dict(epsilon=1, delta=0))])
def test_invalid_config_rejected(inputs, change):
    config, config_path, _, _ = inputs
    config.update(change)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        experiment.read_config(config_path)


def test_dataset_loader_preserves_features_and_rejects_bad_labels(inputs, tmp_path):
    config, _, _, _ = inputs
    loaded = experiment.Dataset.load(Path(config["data_npz"]))
    assert loaded.X.shape == (24, 3)
    assert not loaded.X.flags.writeable
    with np.load(config["data_npz"]) as raw:
        assert_allclose(loaded.X, raw["X_train"], rtol=0, atol=0)
    invalid = tmp_path / "bad.npz"
    np.savez(invalid, X_train=loaded.X, y_train=np.full(24, 2),
             X_test=loaded.X_test, y_test=loaded.y_test)
    with pytest.raises(ValueError, match="invalid frozen"):
        experiment.Dataset.load(invalid)


def test_complete_discovery_summary_freeze_and_fresh_seed_inference(inputs, tmp_path):
    from uncle.__main__ import read_csv, summarize, write_table

    config, config_path, points_path, points = inputs
    points[points.method.eq("guo")].to_csv(points_path, index=False)
    discovery, discovery_summary = tmp_path / "discovery", tmp_path / "discovery-summary"
    experiment.run(config_path, points_path, discovery)
    summarize([discovery], discovery_summary)
    discovered = read_csv(discovery_summary / "summary.csv")
    data_hash = experiment.file_hash(Path(config["data_npz"]))
    experiment.validate_summary_context(config, discovered, data_hash, fresh=False)
    for key, value in experiment.summary_context(config, data_hash).items():
        assert discovered[key].eq(value).all()
    frozen = design.freeze_frontiers(discovered)
    frontier_path = tmp_path / "frozen.csv"
    write_table(frozen, frontier_path)
    config["master_seed"] += 1
    experiment.validate_summary_context(config, frozen, data_hash, fresh=True)
    fresh_config = tmp_path / "fresh-config.json"
    fresh_config.write_text(json.dumps(config))
    inference, inference_summary = tmp_path / "fresh", tmp_path / "fresh-summary"
    experiment.run(fresh_config, frontier_path, inference)
    summarize([inference], inference_summary)
    discovery_rows, fresh_rows = read_rows(discovery), read_rows(inference)
    assert set(fresh_rows.point_id) == set(frozen.point_id)
    assert not set(discovery_rows.trial_seed).intersection(fresh_rows.trial_seed)
    summary = read_csv(inference_summary / "summary.csv")
    experiment.validate_summary_context(config, summary, data_hash, fresh=False)
    assert "selected_by_ce" in summary and "selected_by_accuracy" in summary
    assert set(summary.point_id) == set(frozen.point_id)
    marked = design.mark_fresh_frontiers(summary)
    assert "fresh_pareto_ce" in marked and len(marked) == len(summary)


def test_summary_rejects_modified_ledgers(inputs, tmp_path):
    from uncle.__main__ import summarize

    _, config_path, points_path, points = inputs
    points[points.method.eq("guo")].to_csv(points_path, index=False)
    output = tmp_path / "run"
    experiment.run(config_path, points_path, output)
    rows = read_rows(output)
    rows.loc[0, "delta"] = 0.01
    rows.to_csv(output / "runs.csv", index=False)
    with pytest.raises(ValueError, match="incomplete or changed"):
        summarize([output], tmp_path / "must-not-create")


@pytest.mark.parametrize("key,wrong", [
    ("epsilon", 0.5), ("delta", 0.01), ("dataset", "imdb"),
    ("data_sha256", "not-the-data-hash"), ("ce_reference", 0.7),
    ("accuracy_reference", 0.9), ("rng_dataset", "imdb"),
    ("request_max_k", 100), ("n_removal_requests", 3),
    ("s_inf_values_json", "[0.01]"),
    ("uncle_variant", "theta0-non-A-retained-augmented-v1"),
])
def test_summary_context_rejects_changed_scientific_context(inputs, key, wrong):
    config, _, _, _ = inputs
    digest = experiment.file_hash(Path(config["data_npz"]))
    frame = pd.DataFrame([{**experiment.summary_context(config, digest),
                           "master_seed": config["master_seed"]}])
    frame[key] = wrong
    fresh_config = {**config, "master_seed": config["master_seed"] + 1}
    with pytest.raises(ValueError, match=key):
        experiment.validate_summary_context(fresh_config, frame, digest, fresh=True)


def test_summary_context_requires_fresh_seeds_but_same_seed_for_refinement(inputs):
    config, _, _, _ = inputs
    digest = experiment.file_hash(Path(config["data_npz"]))
    frame = pd.DataFrame([{**experiment.summary_context(config, digest),
                           "master_seed": config["master_seed"]}])
    experiment.validate_summary_context(config, frame, digest, fresh=False)
    with pytest.raises(ValueError, match="fresh inference needs new seeds"):
        experiment.validate_summary_context(config, frame, digest, fresh=True)
    fresh_config = {**config, "master_seed": config["master_seed"] + 1}
    experiment.validate_summary_context(fresh_config, frame, digest, fresh=True)
    with pytest.raises(ValueError, match="refinement needs discovery seeds"):
        experiment.validate_summary_context(fresh_config, frame, digest, fresh=False)


def test_fresh_run_rejects_reusing_discovery_seed_before_output(inputs, tmp_path):
    config, config_path, points_path, points = inputs
    points = points[points.method.eq("guo")].copy()
    for key, value in experiment.summary_context(config, experiment.file_hash(Path(config["data_npz"]))).items():
        points[key] = value
    points["master_seed"] = config["master_seed"]
    points["selected_by_ce"] = True
    points["selected_by_accuracy"] = True
    points.to_csv(points_path, index=False)
    with pytest.raises(ValueError, match="fresh inference needs new seeds"):
        experiment.run(config_path, points_path, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("change", [
    dict(uncle_variant="theta0-non-A-retained-augmented-v1"),
    dict(r_inf_values=[1., .1]), dict(s_inf_values=[]), dict(s_inf_values=[.1, .1]),
])
def test_non_ab2_or_invalid_s_config_fails(inputs, change):
    config, path, _, _ = inputs
    config.update(change)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        experiment.read_config(path)
