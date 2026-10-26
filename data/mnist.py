"""MNIST 3 vs 8 binary classification dataset.

Matches the setup from Guo et al. (ICML 2020): binary classification on
digits 3 and 8, with pixel features normalized to [0, 1].
"""

import numpy as np
from sklearn.datasets import fetch_openml

from data.common import Dataset, SplitDataset
from data.directories import DATASETS_PATH


def load_mnist(force_reset: bool = False) -> SplitDataset:
    """Load MNIST 3 vs 8 binary classification dataset.

    Downloads MNIST via sklearn, filters to digits 3 and 8,
    normalizes features to [0, 1], and maps labels to {0, 1}.
    """
    processed_dir = DATASETS_PATH / "mnist"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "mnist_3v8_processed.npz"

    if not force_reset and processed_path.is_file():
        print(f"Loading processed MNIST 3v8 dataset from {processed_path}")
        return SplitDataset.load(processed_path)

    print("Downloading MNIST via sklearn...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    # Filter to digits 3 and 8
    mask = np.isin(y, [3, 8])
    X, y = X[mask], y[mask]

    # Normalize features to [0, 1]
    X = X / 255.0

    # Map labels: 3 -> 0, 8 -> 1
    y = (y == 8).astype(np.int8)

    # Use standard MNIST split: first 60k are train, rest are test
    # After filtering, approximate the 6:1 ratio
    n_total = len(X)
    n_train = int(n_total * 6 / 7)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)
    split_dataset = SplitDataset(train=train_dataset, test=test_dataset)

    split_dataset.save(processed_path)
    print(f"Saved processed MNIST 3v8 dataset to {processed_path}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    return split_dataset
