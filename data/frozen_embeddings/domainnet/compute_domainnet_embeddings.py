import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets.frozen_embeddings.embeddings.common


def compute_resnet18_embeddings(data_dir, output_path, batch_size=64, num_workers=4):
    """
    Compute ResNet-18 embeddings for all images in a dataset and save them as an NPZ file.

    Args:
        data_dir (str): Path to the dataset folder structured like ImageNet.
        output_path (str): Path to save the embeddings NPZ file.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Device configuration
    device = datasets.frozen_embeddings.embeddings.common.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ResNet-18 model pretrained on ImageNet
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    model = model.to(device)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a dataset and DataLoader
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Containers for embeddings and labels
    all_embeddings = []
    all_labels = []

    # Compute embeddings with a tqdm progress bar
    print("Computing embeddings...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Processing images"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()  # Extract embeddings
            all_embeddings.append(outputs)
            all_labels.append(labels)

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Save embeddings and labels to an NPZ file
    np.savez(output_path, features=all_embeddings, labels=all_labels)
    print(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    # Paths
    data_dir = "./real"  # Path to the DomainNet dataset (e.g., "real" domain)
    output_path = "./embeddings/real_embeddings.npz"  # Output file path

    # Run the computation
    compute_resnet18_embeddings(data_dir, output_path)
