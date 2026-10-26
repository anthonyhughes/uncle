"""Train and test matrices referenced under the data directory.

The MolHIV-Transformer and Fingerprints files are not stored in the repository.
Place each npz at the path from ``reference_path``. The arrays are ``X_train``,
``y_train``, ``X_test``, and ``y_test``, with labels in {0, 1}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data.common import Dataset, DatasetName, SplitDataset
from data.directories import DATASETS_PATH

MOLHIV_TRANSFORMER = DatasetName.MOLHIV_TRANSFORMER.value
FINGERPRINTS = DatasetName.FINGERPRINTS.value


def reference_path(name: str) -> Path:
    """Return the data-directory path for one referenced feature matrix."""
    if name not in (MOLHIV_TRANSFORMER, FINGERPRINTS):
        raise ValueError(f"unknown referenced dataset: {name}")
    return DATASETS_PATH / name / f"{name}.npz"


def load_referenced_split(name: str, path: Path | None = None) -> SplitDataset:
    """Load one referenced binary split. ``path`` overrides the data-directory file."""
    source = reference_path(name) if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Expected {name} features at {source}. "
            "Place the train and test npz there. It is not stored in the repository."
        )
    with np.load(source, allow_pickle=False) as saved:
        train_x = np.asarray(saved["X_train"], dtype=np.float64)
        train_y = np.asarray(saved["y_train"], dtype=np.float64)
        test_x = np.asarray(saved["X_test"], dtype=np.float64)
        test_y = np.asarray(saved["y_test"], dtype=np.float64)
    _check_split(train_x, train_y, "train")
    _check_split(test_x, test_y, "test")
    return SplitDataset(train=Dataset(train_x, train_y), test=Dataset(test_x, test_y))


def load_molhiv_transformer(path: Path | None = None) -> SplitDataset:
    """Load the MolHIV-Transformer feature matrix."""
    return load_referenced_split(MOLHIV_TRANSFORMER, path)


def load_fingerprints(path: Path | None = None) -> SplitDataset:
    """Load the Fingerprints feature matrix."""
    return load_referenced_split(FINGERPRINTS, path)


def _check_split(features: np.ndarray, labels: np.ndarray, part: str) -> None:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError(f"{part} features and labels must be a non-empty aligned pair")
    if not np.isfinite(features).all():
        raise ValueError(f"{part} features must be finite")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError(f"{part} labels must be in {{0, 1}}")
