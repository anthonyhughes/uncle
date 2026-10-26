"""Smoke test: MolHIV-Transformer and Fingerprints can be loaded from a referenced npz."""
from pathlib import Path

import numpy as np
import pytest

from data.directories import DATASETS_PATH
from data.referenced_splits import (
    FINGERPRINTS,
    MOLHIV_TRANSFORMER,
    load_fingerprints,
    load_molhiv_transformer,
    reference_path,
)


def _write_split(path: Path, *, labels=(0, 1, 0, 1)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(labels, dtype=np.float64)
    x = np.arange(len(y) * 3, dtype=np.float64).reshape(len(y), 3)
    with path.open("wb") as stream:
        np.savez(stream, X_train=x, y_train=y, X_test=x[:2], y_test=y[:2])


@pytest.mark.parametrize(
    ("name", "load"),
    (
        (MOLHIV_TRANSFORMER, load_molhiv_transformer),
        (FINGERPRINTS, load_fingerprints),
    ),
)
def test_referenced_split_loads_train_and_test(tmp_path, name, load):
    source = tmp_path / f"{name}.npz"
    _write_split(source)
    split = load(source)
    assert split.train.features.shape == (4, 3)
    assert split.test is not None
    assert split.test.features.shape == (2, 3)
    assert set(split.train.labels.tolist()) <= {0.0, 1.0}
    assert set(split.test.labels.tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("name", (MOLHIV_TRANSFORMER, FINGERPRINTS))
def test_load_dataset_returns_the_referenced_split(tmp_path, name):
    from data.load_datasets import load_dataset

    source = tmp_path / f"{name}.npz"
    _write_split(source)
    direct = {"molhiv_transformer": load_molhiv_transformer, "fingerprints": load_fingerprints}[name](source)
    loaded = load_dataset(name, path=source)
    assert loaded.test is not None and direct.test is not None
    np.testing.assert_array_equal(loaded.train.features, direct.train.features)
    np.testing.assert_array_equal(loaded.train.labels, direct.train.labels)
    np.testing.assert_array_equal(loaded.test.features, direct.test.features)
    np.testing.assert_array_equal(loaded.test.labels, direct.test.labels)


def test_molhiv_transformer_and_fingerprints_are_referenced_under_the_data_directory():
    assert reference_path(MOLHIV_TRANSFORMER) == (
        DATASETS_PATH / "molhiv_transformer" / "molhiv_transformer.npz"
    )
    assert reference_path(FINGERPRINTS) == DATASETS_PATH / "fingerprints" / "fingerprints.npz"


def test_missing_reference_is_rejected(tmp_path):
    missing = tmp_path / "absent.npz"
    with pytest.raises(FileNotFoundError, match="molhiv_transformer"):
        load_molhiv_transformer(missing)


def test_loader_rejects_labels_outside_binary_classes(tmp_path):
    source = tmp_path / "bad.npz"
    _write_split(source, labels=(0, 2, 1, 0))
    with pytest.raises(ValueError, match="labels"):
        load_fingerprints(source)
