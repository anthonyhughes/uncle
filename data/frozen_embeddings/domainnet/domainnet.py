import numpy as np
from data.common import Dataset, SplitDataset
from data.directories import DATASETS_PATH

LABELS_AND_COUNTS = [
    ("garden", 815),
    ("helicopter", 804),
    ("bird", 803),
    ("cat", 796),
    ("sheep", 796),
    ("bread", 794),
    ("mountain", 791),
    ("elephant", 789),
    ("mushroom", 788),
    ("cake", 786)
]


def load_domainnet(
    positive_labels=["cake"],
    negative_labels=["bread"],
    train_size=0.8,
    seed=42,
) -> SplitDataset:
    """
    Load and split a binary classification subset of DomainNet 'real_top10'.

    Args:
        positive_labels (list of str): Labels to assign class 1.
        negative_labels (list of str): Labels to assign class 0.
        train_size (float): Proportion of data to use for training.
        seed (int): Random seed for reproducibility.

    Returns:
        SplitDataset: With train and test splits.
    """
    data_path = DATASETS_PATH / "domainnet" / "real_top10.npz"
    data = np.load(data_path, allow_pickle=True)

    X = data["features"]
    Y = data["labels"].astype(str)

    # Filter relevant samples
    mask_pos = np.isin(Y, positive_labels)
    mask_neg = np.isin(Y, negative_labels)
    mask = mask_pos | mask_neg

    X_filtered = X[mask]
    Y_filtered = np.where(mask_pos[mask], 1, 0).astype(np.int8)

    # Deterministic shuffling
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_filtered))

    split_idx = int(len(X_filtered) * train_size)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    train = Dataset(X_filtered[train_idx], Y_filtered[train_idx])
    test = Dataset(X_filtered[test_idx], Y_filtered[test_idx])

    return SplitDataset(train=train, test=test)
