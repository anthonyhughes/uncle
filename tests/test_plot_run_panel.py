"""A comparison CSV becomes a fresh-run ledger, and that ledger draws the panel."""
import csv
from pathlib import Path

import pytest

from scripts.plot_run_panel import plot_run_panel
from scripts.write_run_ledger import write_run_ledger


def _write_comparison(path: Path, algorithms=("ab2", "guo"), rows=None) -> None:
    records = rows
    if records is None:
        records = [
            (algorithm, removal_size, f"{0.001 * removal_size}", "0.9")
            for algorithm in algorithms
            for removal_size in range(1, 11)
        ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["algorithm", "k", "ce_increase", "success_rate"]
        )
        writer.writeheader()
        for algorithm, removal_size, increase, success in records:
            writer.writerow({
                "algorithm": algorithm,
                "k": removal_size,
                "ce_increase": increase,
                "success_rate": success,
            })


def test_comparison_becomes_a_twenty_row_ledger_and_a_panel(tmp_path):
    source = tmp_path / "comparison.csv"
    _write_comparison(source)
    ledger_path = write_run_ledger(source, tmp_path / "ce_increase_ledger.csv", dataset="mnist")
    with ledger_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert {row["method"] for row in rows} == {"uncle-augmented", "Guo-v4+v5"}
    assert {row["dataset"] for row in rows} == {"mnist"}
    assert {row["status"] for row in rows} == {"attained"}
    assert {int(row["k"]) for row in rows} == set(range(1, 11))
    paths = plot_run_panel(ledger_path, tmp_path / "plots" / "mnist_ce_increase")
    by_suffix = {path.suffix: path for path in paths}
    assert set(by_suffix) == {".png", ".pdf"}
    assert by_suffix[".pdf"].read_bytes().startswith(b"%PDF")
    assert by_suffix[".png"].read_bytes().startswith(b"\x89PNG")


def test_missing_algorithm_is_rejected(tmp_path):
    source = tmp_path / "comparison.csv"
    _write_comparison(source, algorithms=("ab2",))
    with pytest.raises(ValueError, match="guo"):
        write_run_ledger(source, tmp_path / "ce_increase_ledger.csv", dataset="mnist")


def test_nonfinite_ce_increase_is_marked_missing(tmp_path):
    source = tmp_path / "comparison.csv"
    _write_comparison(
        source,
        rows=[
            ("ab2", 1, "0.01", "0.9"),
            ("ab2", 10, "nan", "0.9"),
            ("guo", 1, "0.02", "0.8"),
            ("guo", 10, "0.03", "0.8"),
        ],
    )
    ledger_path = write_run_ledger(source, tmp_path / "ce_increase_ledger.csv")
    with ledger_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    missing = [row for row in rows if row["status"] == "missing"]
    assert len(rows) == 4
    assert len(missing) == 1
    assert missing[0]["method"] == "uncle-augmented"
    assert int(missing[0]["k"]) == 10
    assert missing[0]["ce_increase"] == ""
