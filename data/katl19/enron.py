import numpy as np
import tarfile
from pathlib import Path
from scipy.sparse import csr_matrix
from data.common import Dataset, SplitDataset
from data.directories import DATASETS_PATH, RAW_DATA_PATH
from data.katl19.nlprocessor import NLProcessor
import os


def load_enron(force_reset: bool = False, truncate: int = None,
               load_legacy: bool = False) -> SplitDataset:
    """
    Loads and processes the Enron Spam dataset from pre-downloaded raw data.

    Args:
        force_reset (bool): If True, re-processes the data even if saved data exists.
        truncate (int): If provided, limits the number of samples processed.
        load_legacy (bool): Whether to load a legacy-format processed dataset.

    Returns:
        SplitDataset: The processed dataset.
    """
    # Processed data path (where .npz lives)
    dataset_dir = DATASETS_PATH / 'enron'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    processed_filename = 'spam_truncate_legacy.npz' if load_legacy else f'spam_truncate-{truncate}.npz'
    processed_path = dataset_dir / processed_filename

    # Raw data path (only required when we actually need to regenerate)
    raw_dir = RAW_DATA_PATH / 'enron'
    raw_data_path = raw_dir / 'enron1.tar.gz'

    # Prefer the processed cache; only require raw data when we have no cache.
    if not force_reset and processed_path.is_file():
        print(f"Loading processed data from {processed_path}")
        data = np.load(processed_path)
        X_train, Y_train = np.asarray(data['X_train'], dtype=np.float64), data['Y_train']
        X_valid, Y_valid = np.asarray(data['X_valid'], dtype=np.float64), data['Y_valid']
        X_test, Y_test = np.asarray(data['X_test'], dtype=np.float64), data['Y_test']
    else:
        if not raw_data_path.is_file():
            raise FileNotFoundError(
                f"Raw Enron data not found at {raw_data_path}. "
                "Please make sure you have downloaded the dataset using the setup script."
            )
        # Process the dataset
        print("Processing spam dataset...")
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_spam(raw_dir, truncate)

        # Convert sparse matrices to dense float64 (CountVectorizer produces int)
        X_train = X_train.toarray().astype(np.float64) if isinstance(X_train, csr_matrix) else np.asarray(X_train, dtype=np.float64)
        X_valid = X_valid.toarray().astype(np.float64) if isinstance(X_valid, csr_matrix) else np.asarray(X_valid, dtype=np.float64)
        X_test = X_test.toarray().astype(np.float64) if isinstance(X_test, csr_matrix) else np.asarray(X_test, dtype=np.float64)

        # Save processed data
        np.savez(processed_path, X_train=X_train, Y_train=Y_train, X_valid=X_valid,
                 Y_valid=Y_valid, X_test=X_test, Y_test=Y_test)
        print(f"Saved processed spam dataset to {processed_path}")

    # Wrap in Dataset and SplitDataset
    train_dataset = Dataset(features=X_train, labels=Y_train)
    valid_dataset = Dataset(features=X_valid, labels=Y_valid)
    test_dataset = Dataset(features=X_test, labels=Y_test)
    return SplitDataset(train=train_dataset, test=test_dataset, validation=valid_dataset)


def process_spam(dataset_dir: Path, truncate: int = None):
    """
    Processes the Enron Spam dataset, creating splits and extracting features and labels.

    Args:
        dataset_dir (Path): Path to the dataset directory.
        truncate (int): Limit on the number of examples to process.

    Returns:
        Tuple containing training, validation, and test sets (features and labels).
    """
    rng = np.random.RandomState(0)
    nlprocessor = NLProcessor(rng)

    spam = init_lists(dataset_dir / 'enron1' / 'spam')
    ham = init_lists(dataset_dir / 'enron1' / 'ham')

    docs, Y = nlprocessor.process_spam(spam[:truncate], ham[:truncate])
    num_examples = len(Y)

    # Splits
    train_fraction = 0.8
    valid_fraction = 0.1
    num_train = int(train_fraction * num_examples)
    num_valid = int(valid_fraction * num_examples)
    num_test = num_examples - num_train - num_valid

    docs_train, Y_train = docs[:num_train], Y[:num_train]
    docs_valid, Y_valid = docs[num_train:num_train + num_valid], Y[num_train:num_train + num_valid]
    docs_test, Y_test = docs[-num_test:], Y[-num_test:]

    # Validate sizes
    assert len(docs_train) == len(Y_train)
    assert len(docs_valid) == len(Y_valid)
    assert len(docs_test) == len(Y_test)

    # Process features
    nlprocessor.learn_vocab(docs_train)
    X_train = nlprocessor.get_bag_of_words(docs_train)
    X_valid = nlprocessor.get_bag_of_words(docs_valid)
    X_test = nlprocessor.get_bag_of_words(docs_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def init_lists(folder: Path):
    """
    Initializes a list of documents from a folder.

    Args:
        folder (Path): Path to the folder containing text files.

    Returns:
        List of documents as strings.
    """
    file_list = os.listdir(folder)
    docs = []
    for file_name in file_list:
        with open(folder / file_name, 'r', encoding='latin-1') as f:
            docs.append(f.read())
    return docs
