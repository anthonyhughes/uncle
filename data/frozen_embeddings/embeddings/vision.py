import time
from typing import Optional, Iterable, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import resnet18, resnet50, efficientnet_b0, vit_b_16, swin_t, ResNet18_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from data.frozen_embeddings.embeddings.common import device, get_label_array
from torch.utils.data import Dataset
from PIL import Image
import torch

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ImageListDataset(Dataset):
    def __init__(self, images, labels=None, transform=image_transform):
        """
        Args:
            images (list of np.ndarray or PIL.Image): The raw images.
            labels (list or int or None): Labels for each image, or a single label to apply to all.
            transform (callable, optional): Transform to apply to each image.
        """
        self.images = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in images]
        self.labels = labels if labels is not None else [0] * len(self.images)
        if isinstance(self.labels, int):
            self.labels = [self.labels] * len(self.images)
        self.image_transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.image_transform:
            img = self.image_transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


def embed_images_vision_models(
        image_dataset,
        target_labels: List[int],
        model_name="resnet18",
        max_samples: Optional[int] = None,
        classes: Optional[Iterable] = None,
        embed_all_labels: bool = False,
        all_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vision embeddings using different pretrained models.

    Args:
        image_dataset: Torchvision dataset
        target_labels: List of class names to keep
        model_name: Model to use ("resnet18", "resnet50", "efficientnet_b0", "vit_b_16", "swin_t")
        max_samples: Limit on number of samples

    Returns:
        features: np.ndarray of embeddings
        labels: np.ndarray of binary labels (0/1)
    """
    image_dataset.image_transform = image_transform

    if classes is None:
        classes = image_dataset.classes

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    target_idx = [class_to_idx[c] for c in target_labels]

    # Select model
    model = None
    return_nodes = None

    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        return_nodes = {"avgpool": "feat"}
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        return_nodes = {"avgpool": "feat"}
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(pretrained=True)
        return_nodes = {"avgpool": "feat"}
    elif model_name == "vit_b_16":
        model = vit_b_16(pretrained=True)
        return_nodes = {"encoder.ln": "feat"}  # Final transformer output including CLS token
    elif model_name == "swin_t":
        from torchvision.models import Swin_T_Weights
        weights = Swin_T_Weights.DEFAULT  # This is equivalent to pretrained=True
        model = swin_t(weights=weights)
        return_nodes = {"flatten": "feat"}  # Best node for CLS-like features
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = model.to(device)
    model.eval()
    extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Step 1: Filter dataset
    t0 = time.time()
    if embed_all_labels:
        sample_indices = np.arange(len(image_dataset))
    else:
        # Convert labels to NumPy array (fast)
        if all_labels is None:
            all_labels = get_label_array(image_dataset)

        # Mask: True for samples matching one of the target classes
        mask = np.isin(all_labels, target_idx)

        # Get indices of matching samples
        sample_indices = np.nonzero(mask)[0]

    filtered_dataset = Subset(image_dataset, sample_indices[:max_samples])
    # filtered_dataset = [
    #     (img, target_idx.index(label))
    #     for img, label in tqdm(image_dataset, desc=f"Filtering Images for Classes {target_labels}")
    #     if label in target_idx
    # ]

    # Step 2: Create a DataLoader for batching
    batch_size = 32  # Tune based on your GPU memory
    dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

    features, labels = [], []

    # Step 3: Batch inference
    for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Vision Embeddings [{model_name}]"):
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            outputs = extractor(batch_imgs)["feat"]
            if len(outputs.shape) > 2:
                outputs = torch.flatten(outputs, start_dim=1)  # Flatten if not already
            feats = outputs.cpu().numpy()
        features.extend(feats)
        labels.extend(batch_labels.numpy())

    return np.array(features), np.array(labels)


def embed_images_resnet(image_dataset, target_labels, max_samples: Optional[int] = None):
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])
    # image_dataset.transform = transform
    class_to_idx = {cls: idx for idx, cls in enumerate(image_dataset.classes)}
    target_idx = [class_to_idx[c] for c in target_labels]

    resnet = resnet18(pretrained=True).to(device)
    resnet.eval()
    extractor = create_feature_extractor(resnet, return_nodes={"avgpool": "feat"})

    features, labels = [], []
    count = 0
    for img, label in tqdm(image_dataset, desc="Vision Embeddings"):
        if label in target_idx:
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = extractor(img)["feat"].squeeze().cpu().numpy()
            features.append(feat)
            # Correct label assignment
            labels.append(target_idx.index(label))
        if (max_samples is not None) and (len(labels) >= max_samples):
            break

    return np.array(features), np.array(labels)
