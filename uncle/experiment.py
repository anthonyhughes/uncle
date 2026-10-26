"""Explicit experiment loops: setup -> independent requests -> selected utility.

No engine replacement, callback injection, CLI monkeypatching or old imports.
The point table IS the experiment: fresh inference simply supplies the frozen
frontier table and a config with a different master seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import ab2, design, randomness, retained
from .guo import GuoGeometry, GuoModel
from .logistic import fit, metrics
from .masking import AugmentedGeometry, PreparedRelease
from .privacy import Privacy
from .sekhari import SekhariGeometry, SekhariModel

Array = NDArray[np.float64]
RETAINED_VARIANT = "theta0-non-A-retained-augmented-v1"
AB2_VARIANT = ab2.CERTIFICATE_VARIANT
CANONICAL_NAMES = {"catdog": "cifar10_catdog_resnet18", "mnist": "mnist",
                   "dogfish": "dogfish", "imdb": "imdb",
                   "auto_truck": "cifar10_auto_truck_resnet18", "esc50": "esc50"}


@dataclass(frozen=True)
class Dataset:
    X: Array
    y: Array
    X_test: Array
    y_test: Array

    @classmethod
    def load(cls, path: Path) -> Dataset:
        """Consume frozen arrays; never silently normalize, project or add bias."""
        with np.load(path, allow_pickle=False) as saved:
            arrays = [np.asarray(saved[key], dtype=np.float64) for key in
                      ("X_train", "y_train", "X_test", "y_test")]
        X, y, Xt, yt = arrays
        if (X.ndim != 2 or Xt.ndim != 2 or X.shape[1] != Xt.shape[1]
                or y.shape != (len(X),) or yt.shape != (len(Xt),)
                or not all(np.isfinite(a).all() for a in arrays)
                or not all(np.isin(a, [0, 1]).all() for a in (y, yt))):
            raise ValueError("invalid frozen binary classification arrays")
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays)


def read_config(path: Path) -> dict:
    """A plain explicit JSON config; no inherited defaults or engine overrides."""
    config = json.loads(path.read_text())
    privacy = Privacy(**config["privacy"])
    if (privacy.epsilon, privacy.delta) != (1.0, 1e-4):
        raise ValueError("current paired experiments require epsilon=1, delta=1e-4")
    for name in ("num_seeds", "removal_requests"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not config["ks"] or any(type(k) is not int or k <= 0 for k in config["ks"]):
        raise ValueError("positive integer removal sizes required")
    if len(set(config["ks"])) != len(config["ks"]):
        raise ValueError("removal sizes must be distinct")
    if not np.isfinite(list(config["reference_metrics"].values())).all():
        raise ValueError("finite metric-specific references required")
    variant = config.get("uncle_variant")
    if variant == RETAINED_VARIANT:
        if "s_inf_values" in config:
            raise ValueError("Sinf is not a retained-spectrum witness; configure r_inf_values instead")
        if "r_inf_values" not in config:
            raise ValueError("retained-spectrum UNCLE requires r_inf_values")
    elif variant == AB2_VARIANT:
        if "r_inf_values" in config:
            raise ValueError("Rinf is derived from Sinf-Ainf; configure s_inf_values instead")
        ab2._s_grid(config["s_inf_values"])
    else:
        raise ValueError("uncle_variant must name retained-spectrum or A+B2 augmented UNCLE")
    return config


def export_config(source: Path, variant: str) -> dict:
    """Translate a frozen matched-Guo native config; retain exact float values."""
    if variant == "retained":
        witnesses = dict(uncle_variant=RETAINED_VARIANT, r_inf_values=[1., .1, .01, .001, .0001])
    elif variant == "ab2":
        witnesses = dict(uncle_variant=AB2_VARIANT, s_inf_values=[10., 1., .1, .01, .001])
    else:
        raise ValueError("export variant must be retained or ab2")
    old = json.loads(source.read_text())
    privacy = Privacy(**old["privacy"])
    if (privacy.epsilon, privacy.delta) != (1.0, 1e-4):
        raise ValueError("export expects the current matched-privacy campaign")
    keys = ("dataset", "data_npz", "master_seed", "num_seeds", "removal_requests", "ks",
            "reference_metrics", "reference_lambdas", "guo_lambdas", "sigmas", "safety_factors")
    result = {key: old[key] for key in keys}
    result.update(privacy=old["privacy"], rng_dataset=CANONICAL_NAMES[old["dataset"]],
        uncle_lambdas=old["unerring_lambdas"], geometry_multipliers=[.001, .01, .1, 1., 10.],
        sekhari_lambdas=design.coarse_lambdas(old),
        guo_refinement_points=old.get("guo_v5_parameter_points", 3 * len(old["guo_lambdas"]) * len(old["sigmas"])),
        sekhari_refinement_points=64, inherited_config=str(source), inherited_sha256=file_hash(source),
        **witnesses)
    return result


def initial_points(config: dict, methods: list[str]) -> pd.DataFrame:
    """UNCLE's tau and R2 are derived at runtime from explicit c/factor values."""
    points = []
    for method in methods:
        if method == "uncle":
            for lam in config["uncle_lambdas"]:
                for c in config["geometry_multipliers"]:
                    for factor in config["safety_factors"]:
                        for k in config["ks"]:
                            points.append(dict(method=method, lam=lam, k=k,
                                geometry_multiplier=c, radius_factor=factor))
        elif method == "guo":
            for lam in config["guo_lambdas"]:
                for sigma in config["sigmas"]:
                    for k in config["ks"]:
                        points.append(dict(method=method, lam=lam, sigma=sigma, k=k))
        elif method == "sekhari":
            points.extend(dict(method=method, lam=lam, k=k) for lam in config["sekhari_lambdas"] for k in config["ks"])
        else:
            raise ValueError(f"unknown method {method}")
    frame = pd.DataFrame(points)
    frame["point_id"] = frame.apply(design.point_id, axis=1)
    return frame


def _row(config: dict, point: pd.Series, seed_index: int, seed: int, selected: int,
         success: float, utility: tuple[float, float], trace: dict) -> dict:
    # Identical schema for all methods; inapplicable parameters stay blank.
    row = {key: point.get(key, np.nan) for key in
           ("point_id", "method", "lam", "k", "sigma", "geometry_multiplier", "radius_factor")}
    row.update(dataset=config["dataset"], seed_index=seed_index, trial_seed=seed,
        master_seed=config["master_seed"], selected_request_index=selected,
        n_removal_requests=config["removal_requests"], **config["privacy"],
        success_rate=success, test_ce=utility[0], test_accuracy=utility[1],
        ce_reference=config["reference_metrics"]["test_ce"],
        accuracy_reference=config["reference_metrics"]["test_accuracy"],
        trace=json.dumps(trace, sort_keys=True))
    return row


def _uncle_rows(config: dict, data: Dataset, points: pd.DataFrame,
                seeds: list[tuple[int, int]], requests: dict[int, Array]) -> Iterator[dict]:
    privacy = Privacy(**config["privacy"])
    n, d = data.X.shape
    for lam, lambda_points in points.groupby("lam", sort=True):
        theta = fit(data.X, data.y, lam, polish=True)
        if config["uncle_variant"] == RETAINED_VARIANT:
            model = retained.UncleModel.build(data.X, data.y, theta, lam)
        else:
            model = ab2.UncleModel.build(data.X, data.y, theta, lam, config["s_inf_values"])
        geometries = {c: AugmentedGeometry.build(data.X, c * n * lam)
                      for c in lambda_points.geometry_multiplier.unique()}
        for seed_index, seed in seeds:
            for k, group in lambda_points.groupby("k", sort=True):
                k = int(k)
                selected = randomness.selected_request(seed, k, config["removal_requests"])
                successes = {p: 0 for p in group.point_id}
                for index, prefix in enumerate(requests[seed_index]):
                    if config["uncle_variant"] == RETAINED_VARIANT:
                        request = model.prepare_request(prefix[:k], config["r_inf_values"])
                    else:
                        request = model.prepare_request(prefix[:k])
                    for c, c_points in group.groupby("geometry_multiplier", sort=True):
                        geometry = geometries[c].prepare(prefix[:k])
                        for point in c_points.itertuples():
                            radius = point.radius_factor * k * d / n ** 1.5
                            successes[point.point_id] += request.certify(radius, geometry).accepted
                    if index == selected:
                        selected_request = request
                # Only the predetermined request has a released model/utility.
                # Other rejected requests need no discarded fallback model.
                keep = np.ones(n, dtype=bool)
                keep[selected_request.indices] = False
                fallback = fit(data.X[keep], data.y[keep], lam, normalization_n=n)
                for c, c_points in group.groupby("geometry_multiplier", sort=True):
                    geometry = geometries[c].prepare(selected_request.indices)
                    release = PreparedRelease.build(selected_request, geometry,
                        randomness.mask_rng(seed, k, selected), fallback_theta=fallback)
                    for _, point in c_points.iterrows():
                        radius = point.radius_factor * k * d / n ** 1.5
                        output = release.release(radius, privacy)
                        branches = []
                        for branch in output.certificate.branches:
                            checks = [branch.c1, branch.c2, branch.c3, branch.c4]
                            recorded = dict(
                                checks=[dict(name=c.name, lhs=c.lhs if np.isfinite(c.lhs) else str(c.lhs),
                                             rhs=c.rhs, passed=c.passed) for c in checks])
                            if config["uncle_variant"] == RETAINED_VARIANT:
                                recorded["r_inf"] = branch.r_inf
                            else:
                                recorded["s_inf"] = branch.s_inf
                                recorded["r_inf"] = branch.r_inf
                            branches.append(recorded)
                        trace = dict(tau=geometry.tau, R2=radius, noise_std=output.noise_std,
                            precision="X_ret.T@X_ret+tau*I", covariance="sigma^2*inverse(precision)",
                            accepted=output.certificate.accepted,
                            witness_r_inf=output.certificate.witness_r_inf, branches=branches)
                        if config["uncle_variant"] == AB2_VARIANT:
                            trace.update(uncle_variant=AB2_VARIANT, eta=selected_request.eta,
                                         witness_s_inf=output.certificate.witness_s_inf)
                        yield _row(config, point, seed_index, seed, selected,
                            successes[point.point_id] / config["removal_requests"],
                            metrics(output.theta, data.X_test, data.y_test), trace)


def _guo_rows(config: dict, data: Dataset, points: pd.DataFrame,
              seeds: list[tuple[int, int]], requests: dict[int, Array]) -> Iterator[dict]:
    privacy = Privacy(**config["privacy"])
    geometry = GuoGeometry.build(data.X)
    # Data-only request bank: reuse in a sweep, charge it for an unseen query.
    spectra = {(i, int(k)): [geometry.spectral_norm(prefix[:int(k)]) for prefix in requests[i]]
               for i, _ in seeds for k in points.k.unique()}
    warm_lambdas = np.sort(np.asarray(config["guo_lambdas"]))
    warm_models = [fit(data.X, data.y, lam) for lam in warm_lambdas]
    for (lam, sigma), pair_points in points.groupby(["lam", "sigma"], sort=True):
        distances = np.abs(np.log(warm_lambdas) - np.log(lam))
        # Historical warm bank breaks log-distance ties toward stronger ridge.
        nearest = np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-14))[-1]
        warm = warm_models[int(nearest)]
        ks = sorted(map(int, pair_points.k.unique()))
        for seed_index, seed in seeds:
            model = GuoModel.build(data.X, data.y, lam, sigma, seed, warm_start=warm, geometry=geometry)
            bank = np.asarray([spectra[seed_index, k] for k in ks])
            prepared = dict(zip(ks, model.prepare_requests(requests[seed_index], ks, spectral_norms=bank)))
            for _, point in pair_points.iterrows():
                k = int(point.k)
                selected = randomness.selected_request(seed, k, config["removal_requests"])
                accepted = [r.certify(privacy.epsilon, privacy.delta)[0] for r in prepared[k]]
                output = prepared[k][selected].release(privacy.epsilon, privacy.delta)
                yield _row(config, point, seed_index, seed, selected, float(np.mean(accepted)),
                    metrics(output.theta, data.X_test, data.y_test),
                    dict(beta=output.beta, threshold=output.threshold, accepted=output.accepted,
                         noise="objective perturbation; no output masking", normalization_n=len(data.X)))


def _sekhari_rows(config: dict, data: Dataset, points: pd.DataFrame,
                  seeds: list[tuple[int, int]], requests: dict[int, Array]) -> Iterator[dict]:
    privacy = Privacy(**config["privacy"])
    geometry = SekhariGeometry.build(data.X)
    for lam, lambda_points in points.groupby("lam", sort=True):
        theta = fit(data.X, data.y, lam, polish=True)
        model = SekhariModel.build(data.X, data.y, theta, lam, geometry)
        if not np.isfinite(model.full_gradient_norm) or model.full_gradient_norm > 1e-6:
            raise RuntimeError("Sekhari full model failed the existing stationarity gate")
        for seed_index, seed in seeds:
            for _, point in lambda_points.iterrows():
                k = int(point.k)
                selected = randomness.selected_request(seed, k, config["removal_requests"])
                rng = randomness.sekhari_rng(config["master_seed"], config["rng_dataset"], seed, k)
                output = model.remove(requests[seed_index][selected, :k], privacy.epsilon, privacy.delta, rng)
                yield _row(config, point, seed_index, seed, selected, 1.0,
                    metrics(output.theta, data.X_test, data.y_test),
                    dict(sigma=output.sigma, sensitivity=output.sensitivity,
                         full_gradient_norm=model.full_gradient_norm,
                         success_kind="always-returning algorithm, NOT empirical certificate frequency",
                         covariance="sigma^2*I", normalization_n=len(data.X)-k))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_context(config: dict, data_sha256: str) -> dict:
    """Small provenance record carried into frozen discovery-frontier tables."""
    context = dict(dataset=config["dataset"], **config["privacy"],
        ce_reference=config["reference_metrics"]["test_ce"],
        accuracy_reference=config["reference_metrics"]["test_accuracy"],
        data_sha256=data_sha256, rng_dataset=config["rng_dataset"],
        request_max_k=max(config["ks"]), n_removal_requests=config["removal_requests"],
        uncle_variant=config["uncle_variant"])
    if config["uncle_variant"] == RETAINED_VARIANT:
        context["r_inf_values_json"] = json.dumps(config["r_inf_values"])
    else:
        context["s_inf_values_json"] = json.dumps(config["s_inf_values"])
    return context


def validate_summary_context(config: dict, frame: pd.DataFrame, data_sha256: str,
                             *, fresh: bool) -> None:
    for key, value in summary_context(config, data_sha256).items():
        if key not in frame or not frame[key].eq(value).all():
            raise ValueError(f"frozen summary differs from experiment: {key}")
    if "master_seed" not in frame or frame.master_seed.nunique() != 1:
        raise ValueError("summary needs one discovery master seed")
    same_seed = int(frame.master_seed.iloc[0]) == config["master_seed"]
    if same_seed == fresh:
        raise ValueError("fresh inference needs new seeds; refinement needs discovery seeds")


def run(config_path: Path, points_path: Path, output: Path,
        seed_start: int = 0, seed_stop: int | None = None) -> None:
    """One independent shard. Never overwrite results; merge disjoint seed shards."""
    config = read_config(config_path)
    points = pd.read_csv(points_path, float_precision="round_trip")
    if points.empty or points.point_id.duplicated().any() or not set(points.method) <= {"uncle", "guo", "sekhari"}:
        raise ValueError("nonempty, unique current-method point table required")
    if not points.point_id.eq(points.apply(design.point_id, axis=1)).all():
        raise ValueError("point identities differ from actual parameters")
    data = Dataset.load(Path(config["data_npz"]))
    data_sha256 = file_hash(Path(config["data_npz"]))
    if "selected_by_ce" in points or "selected_by_accuracy" in points:
        validate_summary_context(config, points, data_sha256, fresh=True)
    if max(config["ks"]) >= len(data.X) or not set(points.k) <= set(config["ks"]):
        raise ValueError("point deletion sizes differ from configured request prefixes")
    stop = config["num_seeds"] if seed_stop is None else seed_stop
    if not 0 <= seed_start < stop <= config["num_seeds"]:
        raise ValueError("invalid seed shard")
    all_seeds = randomness.trial_seeds(config["master_seed"], config["rng_dataset"], config["num_seeds"])
    seeds = [(i, int(all_seeds[i])) for i in range(seed_start, stop)]
    requests = {i: randomness.deletion_requests(seed, len(data.X), max(config["ks"]), config["removal_requests"])
                for i, seed in seeds}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config=config, config_sha256=file_hash(config_path), points_sha256=file_hash(points_path),
        data_sha256=data_sha256, seed_start=seed_start, seed_stop=stop,
        trial_seeds=[seed for _, seed in seeds],
        sources={p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    points.to_csv(output / "points.csv", index=False)
    manifest["saved_points_sha256"] = file_hash(output / "points.csv")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started, count = time.perf_counter(), 0
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = None
        for method in ("uncle", "guo", "sekhari"):
            selected_points = points[points.method.eq(method)]
            if selected_points.empty:
                continue
            if method == "uncle":
                rows = _uncle_rows(config, data, selected_points, seeds, requests)
            elif method == "guo":
                rows = _guo_rows(config, data, selected_points, seeds, requests)
            else:
                rows = _sekhari_rows(config, data, selected_points, seeds, requests)
            for row in rows:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                count += 1
    if count != len(points) * len(seeds):
        raise RuntimeError("missing point/seed outputs")
    (output / "complete.json").write_text(json.dumps(dict(status="pass", rows=count,
        runs_sha256=file_hash(output / "runs.csv"), wall_seconds=time.perf_counter()-started), indent=2))
