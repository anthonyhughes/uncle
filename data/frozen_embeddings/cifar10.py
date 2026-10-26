from typing import List
from typing import Union, Tuple

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import datasets

from data.directories import RAW_DATA_PATH
from data.frozen_embeddings.embeddings.common import get_label_array
from data.frozen_embeddings.embeddings.vision import image_transform

CIFAR10_TRAIN = datasets.CIFAR10(root=RAW_DATA_PATH, train=True, download=True, transform=image_transform)
CIFAR10_TEST = datasets.CIFAR10(root=RAW_DATA_PATH, train=False, download=False, transform=image_transform)
CIFAR10_TRAIN_LABELS = get_label_array(CIFAR10_TRAIN)
CIFAR10_TEST_LABELS = get_label_array(CIFAR10_TEST)
CIFAR10_ALL = ConcatDataset([
    CIFAR10_TRAIN, CIFAR10_TEST
])
CIFAR10_ALL_LABELS = np.concatenate((CIFAR10_TRAIN_LABELS, CIFAR10_TEST_LABELS))
CIFAR10_CLASSES = CIFAR10_TRAIN.classes
CIFAR10_TRAIN_RAW = datasets.CIFAR10(root=RAW_DATA_PATH, train=True, download=False)
CIFAR10_TEST_RAW = datasets.CIFAR10(root=RAW_DATA_PATH, train=False, download=False)
CIFAR10_RAW = ConcatDataset([
    CIFAR10_TRAIN_RAW, CIFAR10_TEST_RAW
])


def get_random_cifar_image_by_class(
        include: Union[str, List[str]] = None,
        exclude: Union[str, List[str]] = None
) -> Tuple[np.ndarray, str, int]:
    """
    Returns a random CIFAR image matching class filters.

    Args:
        include: a single class name or list of class names to include.
        exclude: a single class name or list of class names to exclude.

    Returns:
        img: (32, 32, 3) uint8 NumPy array
        class_name: class string
        index: global index in CIFAR10 dataset
    """
    # Normalize inputs
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]

    valid_classes = CIFAR10_CLASSES.copy()

    if include:
        include_set = set(include)
        valid_classes = [cls for cls in valid_classes if cls in include_set]

    if exclude:
        exclude_set = set(exclude)
        valid_classes = [cls for cls in valid_classes if cls not in exclude_set]

    if not valid_classes:
        raise ValueError("No valid classes left after applying include/exclude filters.")

    # Get indices for valid classes
    label_indices = [
        i for i, label in enumerate(CIFAR10_ALL_LABELS)
        if label in [CIFAR10_CLASSES.index(cls) for cls in valid_classes]
    ]

    if not label_indices:
        raise ValueError("No samples found matching specified class filters.")

    selected_index = np.random.choice(label_indices)
    img, class_name = load_cifar_image(selected_index)

    return img, class_name, selected_index


def map_filtered_to_global_index(filtered_index: int, class_names: List[str]) -> int:
    """
    Map an index from the class-filtered CIFAR dataset back to its index in the full CIFAR10 dataset.

    Args:
        filtered_index: Index within the filtered dataset (e.g., cat/dog only)
        class_names: List of class names used to filter (e.g., ["cat", "dog"])

    Returns:
        Index in the full CIFAR10 dataset
    """
    # Convert class names to label indices
    target_class_indices = [CIFAR10_CLASSES.index(cls) for cls in class_names]

    # Convert all labels to NumPy array
    all_labels = np.array(CIFAR10_ALL_LABELS)

    # Boolean mask for labels in the class subset
    mask = np.isin(all_labels, target_class_indices)

    # Get all matching global indices
    matching_indices = np.nonzero(mask)[0]

    if filtered_index >= len(matching_indices):
        raise IndexError(f"Filtered index {filtered_index} is out of range for class subset {class_names}.")

    return int(matching_indices[filtered_index])


def load_cifar_image(index: int) -> Tuple[np.ndarray, str]:
    """
    Load a CIFAR image from the raw dataset.

    Converts it from PIL to a (32, 32, 3) uint8 NumPy array.
    """
    img_pil, label = CIFAR10_RAW[index]
    img = np.array(img_pil)

    if img.dtype != np.uint8:
        img = (255 * img).astype(np.uint8)  # Very rare, mostly a safeguard

    return img.copy(), CIFAR10_CLASSES[label]
