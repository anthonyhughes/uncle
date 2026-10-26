import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix
from tqdm import tqdm

from data.common import Dataset, SplitDataset
from data.directories import DATASETS_PATH, RAW_DATA_PATH

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

CDR_DB_FILENAME = "cdr.db"
PARSED_CDR = "cdr.npz"

# Assuming RAW_DATA_PATH is defined globally as a Path object
def download_cdr_data(data_dir="cdr", force_reset: bool = False):
    """
    Deprecated: No longer downloads data. Just checks that the file exists.
    """
    cdr_db_path = Path(data_dir) / CDR_DB_FILENAME
    if not cdr_db_path.exists():
        raise FileNotFoundError(f"Expected CDR database file not found at {cdr_db_path}. "
                                "Make sure you've run the data setup script.")
    print(f"Using existing CDR database at {cdr_db_path}.")


def extract_cdr_dataset(force_refresh=False) -> SplitDataset:
    """
    Extracts features and labels for train, validation, and test sets from the CDR SQLite database.
    Assumes the raw database has already been downloaded.

    Returns:
        SplitDataset: An instance containing train, validation, and test Dataset objects.
    """
    cdr_path = RAW_DATA_PATH / "cdr"
    parsed_cdr_path = DATASETS_PATH / PARSED_CDR

    # Load pre-parsed data if available
    if not force_refresh and parsed_cdr_path.is_file():
        print(f"Loading pre-saved dataset from {parsed_cdr_path}")
        return SplitDataset.load(str(parsed_cdr_path))

    # Check that raw database file is present
    download_cdr_data(data_dir=cdr_path)

    raw_path = cdr_path / CDR_DB_FILENAME
    print(f'Extracting data from {raw_path}.')
    conn = sqlite3.connect(raw_path)

    # Load data from SQL tables
    LF_labels = pd.read_sql_query("SELECT * FROM label;", conn)
    features = pd.read_sql_query("SELECT * FROM feature;", conn)
    splits = pd.read_sql_query("SELECT id, split FROM candidate;", conn)
    start_test_ind = np.min(splits['id'][splits['split'] == 2])

    test_gold_labels = pd.read_sql_query(
        f"SELECT value, candidate_id FROM gold_label WHERE candidate_id>{start_test_ind - 1} ORDER BY candidate_id ASC;",
        conn
    )

    conn.close()

    # Define train/validation/test splits
    split_ids = [splits['id'][splits['split'] == i] for i in range(3)]
    ids_dups = np.array(LF_labels['candidate_id'])
    which_split = [np.isin(ids_dups, split_ids[i]) for i in range(3)]
    labels = [np.array(LF_labels['value'][which_split[i]]) for i in range(3)]
    print('Extracted labels.')

    # Set maximum feature ID to 328
    _num_features = 328
    # Candidate feature indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 125, 126, 127, 128, 129, 135, 136, 137, 138, 139, 140, 143, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 163, 164, 165, 166, 178, 179, 180, 181, 183, 185, 186, 187, 188, 192, 194, 195, 203, 204, 205, 211, 215, 216, 224, 225, 237, 238, 239, 240, 241, 242, 243, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 281, 283, 284, 294, 297, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422]
    _num_ex = splits.shape[0]
    _num_LF_ex = [labels[i].shape[0] for i in range(3)]

    # Map examples to sparse features
    last_seen = features['candidate_id'][0]
    count = 0
    ex_id_to_ind = {last_seen: count}
    ind_to_features = [dict() for _ in range(_num_ex + 1)]
    for val, ex_id, key_id in tqdm(np.array(features)):
        if ex_id > last_seen:
            count += 1
            last_seen = ex_id
            ex_id_to_ind[last_seen] = count
        if key_id <= _num_features:
            ind_to_features[count][int(key_id)] = val

    # Create sparse feature matrices for LF examples
    x = [dok_matrix((_num_LF_ex[i], _num_features)) for i in range(3)]
    counts = [0 for _ in range(3)]
    for _, ex_id, _ in tqdm(np.array(LF_labels)):
        split = splits['split'][ex_id_to_ind[ex_id]]
        for key_id, val in ind_to_features[ex_id_to_ind[ex_id]].items():
            x[split][counts[split], int(key_id) - 1] = val
        counts[split] += 1
    print('Extracted feature matrices.')

    # Reset test feature matrix to include only gold labels
    _num_test = sum(splits['split'] == 2)
    x[2] = dok_matrix((_num_test, _num_features))
    labels[2] = np.array(test_gold_labels['value'])
    count = 0
    for ex_id, split in tqdm(np.array(splits)):
        if split == 2:
            for key_id, val in ind_to_features[ex_id_to_ind[ex_id]].items():
                x[2][count, int(key_id) - 1] = val
            count += 1

    # Convert labels to binary (0, 1)
    labels = [((labels[i] + 1) / 2) for i in range(3)]

    # Convert sparse matrices to csr format
    train_features = csr_matrix(x[0])
    val_features = csr_matrix(x[1])
    test_features = csr_matrix(x[2])

    # Create and save SplitDataset
    train_dataset = Dataset(features=np.array(train_features.todense()), labels=labels[0])
    val_dataset = Dataset(features=np.array(val_features.todense()), labels=labels[1])
    test_dataset = Dataset(features=np.array(test_features.todense()), labels=labels[2])
    split_dataset = SplitDataset(train=train_dataset, validation=val_dataset, test=test_dataset)
    split_dataset.save(parsed_cdr_path)
    print(f"Saved parsed dataset to {parsed_cdr_path}")

    return split_dataset
