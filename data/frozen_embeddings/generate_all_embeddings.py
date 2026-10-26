import os
import argparse
import numpy as np
import pandas as pd
import glob
from data.directories import RAW_DATA_PATH
from data.directories import DATASETS_PATH
from data.frozen_embeddings.embeddings.common import get_label_array

os.makedirs(DATASETS_PATH / "embeddings", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--datasets', nargs='+', default=[
        "sst2", "imdb", "cifar10_catdog", "cifar10_auto_truck", "cifar10_deer_horse",
        "tape", "tabular", "oxford_flowers"
    ],
    help="List of datasets to process: sst2 imdb cifar10_catdog cifar10_auto_truck cifar10_deer_horse tape tabular oxford_flowers"
)
parser.add_argument(
    '--vision_models', nargs='+',
    default=["resnet18", "resnet50", "efficientnet_b0", "vit_b_16", "swin_t"],
    help="List of vision models to use for image embeddings. Default: all"
)

parser.add_argument('--max_samples', type=int, default=None,
                    help="Maximum number of samples per dataset (if unset, use all)")
args = parser.parse_args()

print(f"{args=}")

print(f"{RAW_DATA_PATH=}")
print(f"{DATASETS_PATH=}")

# Do we have any vision embeddings to compute?
if "oxford_flowers" in args.datasets or np.any([x.startswith("cifar") for x in args.datasets]):
    from torchvision import datasets
    from torchvision import datasets
    from torch.utils.data import ConcatDataset
    from data.frozen_embeddings.embeddings.vision import (embed_images_vision_models, image_transform)
    if np.any([x.startswith("cifar") for x in args.datasets]):
        from data.frozen_embeddings.cifar10 import CIFAR10_TRAIN, CIFAR10_ALL, CIFAR10_ALL_LABELS

# Do we have any NLP embeddings to compute?
if "sst2" in args.datasets or "imdb" in args.datasets:
    from data.frozen_embeddings.embeddings.nlp import embed_text_bert

# Do we have any protein datasets to embed?
if "tape" in args.datasets:
    from data.frozen_embeddings.embeddings.misc import load_tape_lmdb, embed_tape_proteins

if "tabular" in args.datasets:
    from data.frozen_embeddings.embeddings.misc import embed_tabnet_features


def save_embeddings(name, features: np.ndarray, labels: np.ndarray):
    print(f"Saving: {name} => {features.shape}, {labels.shape}")
    np.savez(DATASETS_PATH / f"embeddings/{name}.npz", features=features, labels=labels)


if "sst2" in args.datasets:
    print("Processing SST-2...")
    sst2 = pd.read_csv(RAW_DATA_PATH / "SST-2/train.tsv", sep="\t")
    texts, labels = sst2["sentence"].tolist(), sst2["label"].tolist()
    if args.max_samples:
        texts, labels = texts[:args.max_samples], labels[:args.max_samples]
    features = embed_text_bert(texts)
    save_embeddings("sst2", features, np.array(labels))

if "imdb" in args.datasets:
    print("Processing IMDb...")
    imdb = pd.read_csv(RAW_DATA_PATH / "imdb.csv")
    texts = imdb["review"].tolist()
    labels = [1 if s == "positive" else 0 for s in imdb["sentiment"]]
    if args.max_samples:
        texts, labels = texts[:args.max_samples], labels[:args.max_samples]
    features = embed_text_bert(texts)
    save_embeddings("imdb", features, np.array(labels))

if "tabular" in args.datasets:
    print("Processing Tabular (Adult Income)...")
    cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
    df = pd.read_csv(RAW_DATA_PATH / "adult.data", names=cols, na_values=" ?", skipinitialspace=True).dropna()
    if args.max_samples:
        df = df.iloc[:args.max_samples]
    features, labels = embed_tabnet_features(df, label_col="income")
    save_embeddings("adult_income", features, labels)

if "tape" in args.datasets:
    print("Processing TAPE (Protein Sequences)...")
    sequences, labels = load_tape_lmdb(RAW_DATA_PATH / "tape/fluorescence/fluorescence_train.lmdb",
                                       max_samples=args.max_samples)
    if args.max_samples:
        sequences, labels = sequences[:args.max_samples], labels[:args.max_samples]
    features = embed_tape_proteins(sequences)
    save_embeddings("tape_fluorescence", features, np.array(labels))

if "oxford_flowers" in args.datasets:
    print("Processing Oxford Flowers-102...")

    flowers_train = datasets.Flowers102(root=RAW_DATA_PATH, split="train", download=True, transform=image_transform)
    flowers_val = datasets.Flowers102(root=RAW_DATA_PATH, split="val", download=True, transform=image_transform)
    flowers_test = datasets.Flowers102(root=RAW_DATA_PATH, split="test", download=True, transform=image_transform)

    flowers_train_labels = get_label_array(flowers_train)
    flowers_val_labels = get_label_array(flowers_val)
    flowers_test_labels = get_label_array(flowers_test)

    flowers_all = ConcatDataset([
        flowers_train, flowers_val, flowers_test
    ])
    flowers_all_labels = np.concatenate((flowers_train_labels, flowers_val_labels, flowers_test_labels))

    # Since Flowers-102 has 102 classes, maybe you want to just embed them all.
    for model_name in args.vision_models:
        print(f"Embedding Oxford Flowers using {model_name}...")
        target_labels = list(range(102))  # ✅ All classes are indexed 0 to 101
        features, labels = embed_images_vision_models(
            flowers_all,
            target_labels=target_labels,  # all classes
            model_name=model_name,
            max_samples=args.max_samples,
            classes=target_labels,
            embed_all_labels=True,
            all_labels=flowers_all_labels
        )
        save_embeddings(f"oxford_flowers_{model_name}", features, labels)

cifar_pairs = {
    "cifar10_catdog": ("cat", "dog"),
    "cifar10_auto_truck": ("automobile", "truck"),
    "cifar10_deer_horse": ("deer", "horse"),
}

# # Are we doing any CIFAR datasets?
# if np.any([key in args.datasets for key in cifar_pairs]):
#     cifar_train = datasets.CIFAR10(root=RAW_DATA_PATH, train=True, download=True, transform=image_transform)
#     cifar_test = datasets.CIFAR10(root=RAW_DATA_PATH, train=False, download=True, transform=image_transform)
#     cifar_train_labels = get_label_array(cifar_train)
#     cifar_test_labels = get_label_array(cifar_test)
#     cifar_all = ConcatDataset([
#         cifar_train, cifar_test
#     ])
#     cifar_all_labels = np.concatenate((cifar_train_labels, cifar_test_labels))

for key, (class1, class2) in cifar_pairs.items():
    if key in args.datasets:
        print(f"Processing CIFAR-10 ({class1} vs {class2})...")
        for model_name in args.vision_models:
            print(f"Embedding CIFAR-10 [{key}] using {model_name}...")
            features, labels = embed_images_vision_models(
                CIFAR10_ALL,
                [class1, class2],
                model_name=model_name,
                max_samples=args.max_samples,
                classes=CIFAR10_TRAIN.classes,
                all_labels=CIFAR10_ALL_LABELS
            )
            save_embeddings(f"{key}_{model_name}", features, labels)
