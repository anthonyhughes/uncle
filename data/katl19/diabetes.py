import zipfile

import numpy as np
import pandas as pd

from data.common import Dataset, SplitDataset
from data.directories import DATASETS_PATH, RAW_DATA_PATH

DIABETES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"


# RAW_DATA_PATH is assumed to be a global pathlib.Path object
def load_diabetes(force_reset: bool = False) -> SplitDataset:
    """
    Loads the diabetes dataset, processes features and labels, and saves or loads it as an npz file.
    Assumes raw data is already available at RAW_DATA_PATH / 'diabetes'.

    Returns:
        SplitDataset: Train and test split dataset with processed features and labels.
    """
    # Updated path to pre-downloaded raw data
    hospital_path = DATASETS_PATH / 'hospital.npz'
    csv_path = RAW_DATA_PATH / 'diabetic_data.csv'

    # Ensure raw data exists
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing expected CSV file: {csv_path}. "
            f"Make sure the dataset was downloaded using the setup script."
        )

    # Process data and save to .npz if it hasn't been done already
    if force_reset or (not hospital_path.is_file()):
        print("Processing diabetes data...")
        df = pd.read_csv(csv_path)

        # Process features and labels
        X, Y = process_diabetes_data(df)

        rng = np.random.RandomState(2)

        num_examples = len(Y)
        assert X.shape[0] == num_examples
        num_train_examples = 20000
        num_train_examples_per_class = int(num_train_examples / 2)
        num_test_examples = num_examples - num_train_examples
        assert num_test_examples > 0

        pos_idx = np.where(Y == 1)[0]
        neg_idx = np.where(Y == -1)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        assert len(pos_idx) + len(neg_idx) == num_examples

        train_idx = np.concatenate((pos_idx[:num_train_examples_per_class], neg_idx[:num_train_examples_per_class]))
        test_idx = np.concatenate((pos_idx[num_train_examples_per_class:], neg_idx[num_train_examples_per_class:]))
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        X_train = np.array(X.iloc[train_idx, :], dtype=np.float32)
        Y_train = Y[train_idx]

        X_test = np.array(X.iloc[test_idx, :], dtype=np.float32)
        Y_test = Y[test_idx]

        Y_train = np.array((Y_train + 1) / 2, dtype=int)
        Y_test = np.array((Y_test + 1) / 2, dtype=int)

        # Save processed data
        np.savez(hospital_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        print(f"Saved processed data to {hospital_path}")
    else:
        # Load processed data from the saved npz file
        data = np.load(hospital_path, allow_pickle=True)  # Set allow_pickle=True to avoid loading issues
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']

    # Create SplitDataset instance
    train_dataset = Dataset(features=X_train, labels=Y_train)
    test_dataset = Dataset(features=X_test, labels=Y_test)
    return SplitDataset(train=train_dataset, test=test_dataset)


def download_file(url, dest_path):
    """Downloads a file from a given URL to the specified path."""
    import urllib.request
    urllib.request.urlretrieve(url, dest_path)


def process_diabetes_data(df):
    """
    Processes the raw diabetes data, encoding categorical variables and setting binary labels.

    Parameters:
        df (pd.DataFrame): The raw diabetes dataset DataFrame.

    Returns:
        X (pd.DataFrame): Processed feature DataFrame.
        Y (np.ndarray): Binary labels for each sample.
    """
    # Extract numerical features
    continuous_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]

    categorical_feature = [
        'gender', 'race', 'age', 'discharge_disposition_id',
        'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
        'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'troglitazone', 'tolazamide', 'examide', 'citoglipton',
        'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change', 'diabetesMed'
    ]

    X = df.loc[:, continuous_features]

    for cat_var_name in categorical_feature:
        cat_var = pd.Categorical(df.loc[:, cat_var_name])

        # Just have one dummy variable if it's boolean
        drop_first = (len(cat_var.categories) == 2)
        # drop_first = True

        dummies = pd.get_dummies(cat_var, prefix=cat_var_name, drop_first=drop_first)

        X = pd.concat([X, dummies], axis=1)

    readmitted = pd.Categorical(df.readmitted)
    Y = np.copy(readmitted.codes)

    Y[Y >= 1] = -1
    Y[Y == 0] = 1
    return X, Y
