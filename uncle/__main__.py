"""Small, explicit CLI. All data and generated outputs use supplied paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import design, experiment, reporting


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def summarize(directories: list[Path], output: Path) -> None:
    """Merge completed seed/point shards; do not pool incompatible campaigns."""
    ledgers, selections, first = [], [], None
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        complete = json.loads((directory / "complete.json").read_text())
        if (complete["status"] != "pass"
                or complete["runs_sha256"] != experiment.file_hash(directory / "runs.csv")
                or manifest["saved_points_sha256"] != experiment.file_hash(directory / "points.csv")):
            raise ValueError(f"incomplete or changed shard: {directory}")
        if first is None:
            first = manifest
        elif any(first[name] != manifest[name] for name in ("config", "data_sha256", "sources")):
            raise ValueError("cannot mix data, configs/seeds/privacy, or implementations")
        rows = read_csv(directory / "runs.csv")
        config = manifest["config"]
        for key, value in config["privacy"].items():
            if not rows[key].eq(value).all():
                raise ValueError("recorded privacy budget differs from config")
        expected = set(range(manifest["seed_start"], manifest["seed_stop"]))
        points = read_csv(directory / "points.csv")
        if (len(rows) != complete["rows"] or set(rows.point_id) != set(points.point_id)
                or len(rows) != len(points) * len(expected)
                or any(set(group.seed_index) != expected for _, group in rows.groupby("point_id"))):
            raise ValueError("shard is missing expected operating points or seeds")
        ledgers.append(rows)
        columns = [c for c in ("point_id", "selected_by_ce", "selected_by_accuracy", "ce_order", "accuracy_order") if c in points]
        selections.append(points[columns])
    config = first["config"]
    summary = design.summarize_seeds(pd.concat(ledgers, ignore_index=True),
        expected_seeds=config["num_seeds"], requests_per_seed=config["removal_requests"])
    flags = pd.concat(selections).drop_duplicates()
    if flags.point_id.duplicated().any():
        raise ValueError("inconsistent frozen frontier selections")
    summary = summary.merge(flags, on="point_id", validate="one_to_one")
    for name, value in experiment.summary_context(config, first["data_sha256"]).items():
        summary[name] = value
    summary["master_seed"] = config["master_seed"]
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(dict(
        config=config, shards=[str(p) for p in directories],
        data_sha256=first["data_sha256"], sources=first["sources"]), indent=2))


def refine(config: dict, summary: pd.DataFrame, method: str, output: Path) -> None:
    if "selected_by_ce" in summary or "selected_by_accuracy" in summary:
        raise ValueError("do not retune on the fresh frozen-frontier stage")
    experiment.validate_summary_context(config, summary,
        experiment.file_hash(Path(config["data_npz"])), fresh=False)
    if method == "guo":
        pairs, metadata, _, _ = design.guo_refinement_design(summary,
            dataset=config["dataset"], count=config["guo_refinement_points"],
            master_seed=config["master_seed"])
        points = pairs.rename(columns={"lambda": "lam"})[["lam", "sigma"]]
    else:
        lambdas, metadata = design.refinement_lambdas(summary[summary.method.eq("sekhari")],
            config["sekhari_lambdas"], config["sekhari_refinement_points"])
        points = pd.DataFrame({"lam": lambdas})
    points = points.merge(pd.DataFrame({"k": config["ks"]}), how="cross")
    points["method"] = method
    points["point_id"] = points.apply(design.point_id, axis=1)
    write_table(points, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-config", help="translate one frozen matched native config")
    export.add_argument("source", type=Path)
    export.add_argument("--variant", choices=("retained", "ab2"), required=True)
    plan = commands.add_parser("plan", help="write explicit coarse operating points")
    plan.add_argument("config", type=Path)
    plan.add_argument("--methods", nargs="+", choices=("uncle", "guo", "sekhari"), required=True)
    run = commands.add_parser("run", help="run one disjoint seed shard")
    run.add_argument("config", type=Path)
    run.add_argument("points", type=Path)
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--seed-stop", type=int)
    aggregate = commands.add_parser("summarize", help="merge completed shards with identical provenance")
    aggregate.add_argument("runs", type=Path, nargs="+")
    refinement = commands.add_parser("refine", help="write the existing Guo v5 / Sekhari refinement")
    refinement.add_argument("config", type=Path)
    refinement.add_argument("summary", type=Path)
    refinement.add_argument("--method", choices=("guo", "sekhari"), required=True)
    select = commands.add_parser("select", help="freeze discovery-frontier parameters for independent seeds")
    select.add_argument("summary", type=Path)
    plot = commands.add_parser("plot", help="PNG/PDF Pareto and summary plots")
    plot.add_argument("summary", type=Path)
    plot.add_argument("--dataset", required=True)
    for command in (export, plan, run, aggregate, refinement, select, plot):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-config":
        config = experiment.export_config(args.source, args.variant)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(config, stream, indent=2)
    elif args.command == "plan":
        write_table(experiment.initial_points(experiment.read_config(args.config), args.methods), args.out)
    elif args.command == "run":
        experiment.run(args.config, args.points, args.out, args.seed_start, args.seed_stop)
    elif args.command == "summarize":
        summarize(args.runs, args.out)
    elif args.command == "refine":
        refine(experiment.read_config(args.config), read_csv(args.summary), args.method, args.out)
    elif args.command == "select":
        write_table(design.freeze_frontiers(read_csv(args.summary)), args.out)
    elif args.command == "plot":
        summary = read_csv(args.summary)
        if summary.dataset.nunique() != 1 or summary.dataset.iloc[0] != args.dataset:
            raise ValueError("plot one matching dataset at a time")
        if any(summary[name].nunique() != 1 for name in ("epsilon", "delta", "master_seed")):
            raise ValueError("plot cannot mix privacy budgets or seed campaigns")
        reporting.plot_pareto(summary, args.out, dataset=args.dataset)
        reporting.plot_summary(summary, args.out, dataset=args.dataset)
    print(args.out)


if __name__ == "__main__":
    main()
