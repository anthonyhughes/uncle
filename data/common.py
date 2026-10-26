import os
from dataclasses import dataclass
from typing import Optional
import urllib
import urllib.request
import numpy as np
import sys

try:
    import ssl
    import certifi
except:
    pass

from enum import Enum


class DatasetName(str, Enum):
    # Embedded datasets
    SST2 = "sst2"
    IMDB = "imdb"
    CIFAR10_CAT_DOG = "cifar10_catdog"
    CIFAR10_AUTO_TRUCK = "cifar10_auto_truck"
    CIFAR10_DEER_HORSE = "cifar10_deer_horse"
    OXFORD_FLOWERS = "oxford_flowers"
    ADULT_INCOME = "adult_income"
    TAPE_FLUORESCENCE = "tape_fluorescence"

    # Traditional datasets
    DOGFISH = "DogFish"
    ENRON = "Enron"
    DIABETES = "Diabetes"
    CDR = "CDR"
    MNIST = "MNIST"
    MOLHIV_TRANSFORMER = "molhiv_transformer"
    FINGERPRINTS = "fingerprints"




@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    original_indices: Optional[np.ndarray] = None

    def subsample(self, max_samples: Optional[int], seed: int = 42) -> 'Dataset':
        if max_samples is None or max_samples >= len(self.features):
            return self
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.features), size=max_samples, replace=False)
        return Dataset(
            features=self.features[indices],
            labels=self.labels[indices],
            original_indices=(self.original_indices[indices] if self.original_indices is not None else indices)
        )



@dataclass
class SplitDataset:
    """Dataclass containing train, validation, and test datasets for logistic regression."""
    train: Dataset
    validation: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def subsample(self, max_train: Optional[int] = None, max_test: Optional[int] = None,
                  seed: int = 42) -> 'SplitDataset':
        return SplitDataset(
            train=self.train.subsample(max_train, seed=seed),
            validation=self.validation,  # Validation untouched
            test=(self.test.subsample(max_test, seed=seed + 1) if self.test else None)
        )

    def save(self, file_path: str):
        """Saves the SplitDataset to a single .npz file."""
        data_dict = {
            'train_features': self.train.features,
            'train_labels': self.train.labels,
        }
        if self.train.original_indices is not None:
            data_dict['train_original_indices'] = self.train.original_indices

        if self.validation is not None:
            data_dict['validation_features'] = self.validation.features
            data_dict['validation_labels'] = self.validation.labels
            if self.validation.original_indices is not None:
                data_dict['validation_original_indices'] = self.validation.original_indices

        if self.test is not None:
            data_dict['test_features'] = self.test.features
            data_dict['test_labels'] = self.test.labels
            if self.test.original_indices is not None:
                data_dict['test_original_indices'] = self.test.original_indices

        np.savez(file_path, **data_dict)

    @classmethod
    def load(cls, file_path: str) -> "SplitDataset":
        """Loads a SplitDataset from a single .npz file."""
        with np.load(file_path) as data:
            train = Dataset(
                features=data['train_features'],
                labels=data['train_labels'],
                original_indices=data['train_original_indices'] if 'train_original_indices' in data else None
            )

            validation = None
            if 'validation_features' in data and 'validation_labels' in data:
                validation = Dataset(
                    features=data['validation_features'],
                    labels=data['validation_labels'],
                    original_indices=data[
                        'validation_original_indices'] if 'validation_original_indices' in data else None
                )

            test = None
            if 'test_features' in data and 'test_labels' in data:
                test = Dataset(
                    features=data['test_features'],
                    labels=data['test_labels'],
                    original_indices=data['test_original_indices'] if 'test_original_indices' in data else None
                )

        return cls(train=train, validation=validation, test=test)


def maybe_download(url, filename, download_dir):
    """
    Downloads a file into the specified download directory if it does
    not already exist.

    :param url: the web URL to the file
    :param filename: the filename to save it as
    :param download_dir: the directory to download into
    :return: the absolute path to the downloaded file
    """

    save_path = os.path.abspath(os.path.join(download_dir, filename))

    if not os.path.exists(save_path):
        print("Downloading {} into {}".format(filename, download_dir))

        os.makedirs(download_dir, exist_ok=True)

        def _print_download_progress(count, blockSize, totalSize):
            percent = int(count * blockSize * 100.0 / totalSize)
            sys.stdout.write("\rDownloading {}: {}%".format(filename, percent))
            sys.stdout.flush()

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except:
            ssl_context = None

        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=save_path,
                                                  reporthook=_print_download_progress)

        print("\nDownload complete.")

    return save_path


