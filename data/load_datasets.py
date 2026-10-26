from data.katl19.cdr_data import extract_cdr_dataset
from data.common import SplitDataset, DatasetName

from typing import Union

from data.katl19.diabetes import load_diabetes
from data.katl19.dogfish import load_dogfish


def load_dataset(
        dataset: Union[str, DatasetName, "EmbeddedDataset"],
        verbose: bool = False,
        **kwargs
) -> SplitDataset:
    """
    Load a dataset, supporting traditional and embedded formats.

    Parameters:
        dataset (Union[str, DatasetName, EmbeddedDataset]): Dataset identifier.
        verbose (bool): If True, print dataset details.
        **kwargs: Passed to loader (e.g., seed, train_size, etc.).

    Returns:
        SplitDataset: Train/test data.
    """

    # Embedding objects import their loader only when that branch runs.
    if not isinstance(dataset, (str, DatasetName)):
        from data.frozen_embeddings.loader import EmbeddedDataset, load_embedding_dataset
        if not isinstance(dataset, EmbeddedDataset):
            raise TypeError("dataset must be a str, DatasetName, or EmbeddedDataset")
        if verbose:
            print(f"Loading embedded dataset: {dataset.name.value} (model: {dataset.model})")
        return load_embedding_dataset(dataset, **kwargs)

    # If dataset is a DatasetName enum or str, normalize to string
    if isinstance(dataset, DatasetName):
        dataset_name = dataset.value
    else:
        dataset_name = dataset

    if verbose:
        print(f"Loading dataset: {dataset_name}")

    # Handle traditional datasets
    if dataset_name == "DogFish":
        experiment = load_dogfish()
    elif dataset_name == "Enron":
        from data.katl19.enron import load_enron
        experiment = load_enron(**kwargs)
    elif dataset_name == "MNIST":
        from data.mnist import load_mnist
        experiment = load_mnist(**kwargs)
    elif dataset_name == "Diabetes":
        experiment = load_diabetes(**kwargs)
    elif dataset_name == "CDR":
        experiment = extract_cdr_dataset(**kwargs)
    elif dataset_name in (DatasetName.MOLHIV_TRANSFORMER.value, "molhiv_transformer"):
        from data.referenced_splits import load_molhiv_transformer
        experiment = load_molhiv_transformer(**kwargs)
    elif dataset_name in (DatasetName.FINGERPRINTS.value, "fingerprints"):
        from data.referenced_splits import load_fingerprints
        experiment = load_fingerprints(**kwargs)
    else:
        # Assume it's an embedded dataset without a model specified
        from data.frozen_embeddings.loader import EmbeddedDataset, load_embedding_dataset
        try:
            name_enum = DatasetName(dataset_name.lower())
            experiment = load_embedding_dataset(EmbeddedDataset(name_enum), **kwargs)
        except ValueError:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    if verbose:
        print("Train set dimensions:")
        print(experiment.train.features.shape, experiment.train.labels.shape)

    return experiment
