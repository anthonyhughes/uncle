import fnmatch
import re

from data.common import DatasetName
from data.frozen_embeddings.loader import EmbeddedDataset, VISION_MODELS


def get_all_dataset_names():
    classic_datasets = ["Diabetes", "CDR"]

    # Tabular dataset
    tabular_datasets = [
        EmbeddedDataset(DatasetName.ADULT_INCOME)
    ]

    # Vision-based embedding datasets
    flowers_datasets = [
        EmbeddedDataset(DatasetName.OXFORD_FLOWERS, model=model)
        for model in VISION_MODELS
    ]

    cifar_catdog_datasets = [
        EmbeddedDataset(DatasetName.CIFAR10_CAT_DOG, model=model)
        for model in VISION_MODELS
    ]

    # Remaining CIFAR variants (not cat/dog)
    cifar_other = [
        DatasetName.CIFAR10_AUTO_TRUCK,
        DatasetName.CIFAR10_DEER_HORSE
    ]

    cifar_other_datasets = [
        EmbeddedDataset(name, model=model)
        for name in cifar_other
        for model in VISION_MODELS
    ]

    nlp_datasets = [
        EmbeddedDataset(name=DatasetName.IMDB),
        EmbeddedDataset(name=DatasetName.SST2),
    ]

    return (
            ["DogFish", "Enron", "MNIST"] +
            cifar_catdog_datasets +
            flowers_datasets +
            classic_datasets +
            cifar_other_datasets +
            tabular_datasets +
            nlp_datasets
    )


def select_datasets(dataset_pattern: str, regex: bool = False):
    all_datasets = get_all_dataset_names()
    # Compile pattern
    if regex:
        pattern = dataset_pattern
        regex = re.compile(pattern, re.IGNORECASE)
    else:
        pattern = fnmatch.translate(dataset_pattern)
        regex = re.compile(pattern, re.IGNORECASE)

    selected_datasets = [ds for ds in all_datasets if regex.match(str(ds))]

    return selected_datasets
