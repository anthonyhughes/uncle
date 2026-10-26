import numpy as np
from data.common import Dataset, SplitDataset
from data.directories import DATASETS_PATH, RAW_DATA_PATH


def load_dogfish(force_reset: bool = False) -> SplitDataset:
    """
    Loads and processes the Dogfish dataset.
    Assumes raw .npz files are located in RAW_DATA_PATH / 'dogfish'.
    Saves and loads processed data from DATASETS_PATH / 'dogfish'.
    """
    # Directories
    raw_dir = RAW_DATA_PATH / "dogfish"
    processed_dir = DATASETS_PATH / "dogfish"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Filenames
    raw_train_file = raw_dir / "dogfish_train.npz"
    raw_test_file = raw_dir / "dogfish_test.npz"
    processed_path = processed_dir / "dogfish_processed.npz"

    # Prefer the processed cache if it exists and we aren't forcing a reset.
    # This lets us run on environments that have only the processed npz
    # (e.g. HPC nodes without the raw download).
    if not force_reset and processed_path.is_file():
        print(f"Loading processed Dogfish dataset from {processed_path}")
        return SplitDataset.load(processed_path)

    # Otherwise we need raw files to regenerate.
    if not raw_train_file.is_file() or not raw_test_file.is_file():
        raise FileNotFoundError(
            f"Expected Dogfish files not found in {raw_dir}. "
            f"Please run the setup script to download them."
        )

    # Load raw .npz data
    data_train = np.load(raw_train_file)
    data_test = np.load(raw_test_file)

    X_train = data_train["inception_features_val"]
    Y_train = data_train["labels"].astype(np.int8)
    X_test = data_test["inception_features_val"]
    Y_test = data_test["labels"].astype(np.int8)

    # Build and save processed dataset
    train_dataset = Dataset(X_train, Y_train)
    test_dataset = Dataset(X_test, Y_test)
    split_dataset = SplitDataset(train=train_dataset, test=test_dataset)

    split_dataset.save(processed_path)
    print(f"Saved processed Dogfish dataset to {processed_path}")
    return split_dataset
