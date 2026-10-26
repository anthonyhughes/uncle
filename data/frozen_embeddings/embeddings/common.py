from tqdm import tqdm

import torch
import numpy as np


def get_label_array(dataset) -> np.ndarray:
    """Return all labels as a NumPy array from various dataset types."""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    else:
        # Fallback: expensive, but rarely needed
        return np.array([label for _, label in tqdm(dataset, desc="Extracting Labels")])


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
