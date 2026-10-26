import time

import pandas as pd
# import openl3
# import soundfile as sf
try:
    from tape import ProteinBertModel, TAPETokenizer
except ImportError:
    ProteinBertModel = None
    TAPETokenizer = None
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import numpy as np

from data.frozen_embeddings.embeddings.common import device
from torchvision.models.feature_extraction import create_feature_extractor



def embed_tabnet_features(
    df: pd.DataFrame, label_col, cat_cols=None, num_cols=None,
    pretrained_model_path=None, train_embed_split=0.8,
    final_feature_dim=1024
):
    print(f"Running Tabnet Embedding with {label_col=}, {final_feature_dim=}")
    print(f"{df.describe()}")
    df = df.copy()
    y = LabelEncoder().fit_transform(df[label_col])
    X = df.drop(columns=[label_col])

    if num_cols is None:
        num_cols = X.select_dtypes(include=["float", "int"]).columns.tolist()
    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Encode categorical features
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Scale numeric features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_np = X[num_cols + cat_cols].values

    print(f"{np.shape(X_np)=}, {np.shape(y)=}")

    # Split the dataset
    X_train, X_embed, y_train, y_embed = train_test_split(
        X_np, y, train_size=train_embed_split, stratify=y, random_state=42
    )
    print(f"{X_train.shape=}")

    # Train TabNet or load pretrained model
    clf = TabNetClassifier(
        n_d=final_feature_dim,
        n_a=final_feature_dim,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-3,
        seed=42,
        verbose=0
    )
    print(clf)

    if pretrained_model_path:
        clf.load_model(pretrained_model_path)
    else:
        clf.fit(X_train, y_train, max_epochs=200, patience=20)

    clf.network.eval()

    # Generate embeddings on embed set
    with torch.no_grad():
        X_tensor = torch.tensor(X_embed, dtype=torch.float32).to(device)
        embedded_x = clf.network.embedder(X_tensor)  # passes through input embedder
        # Run the TabNet encoder only, skip final classification head
        encoder_out = clf.network.tabnet.encoder(embedded_x)[0]  # shape: (batch, n_d)
        embeddings = encoder_out[-1].detach().cpu().numpy()

    return embeddings, y_embed


def embed_tape_proteins(sequences, model_name="bert-base", batch_size=16):
    model = ProteinBertModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = TAPETokenizer(vocab="iupac")

    features = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="TAPE Embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        token_ids = [tokenizer.encode(seq) for seq in batch_seqs]
        # Filter out empty tokenized sequences
        filtered = [(s, t) for s, t in zip(batch_seqs, token_ids) if len(t) > 0]

        if any(len(t) == 0 for t in token_ids):
            print(f"Warning: {[t for t in token_ids if len(t) == 0]} empty tokenized sequence in batch")

        if not filtered:
            continue  # skip batch

        batch_seqs, token_ids = zip(*filtered)
        max_len = max(len(t) for t in token_ids)
        input_ids = torch.tensor([list(t) + [0] * (max_len - len(t)) for t in token_ids]).to(device)
        with torch.no_grad():
            output = model(input_ids)
            cls_embeds = output[0][:, 0, :].cpu().numpy()  # [CLS] token embedding
            features.append(cls_embeds)

    return np.concatenate(features, axis=0)


def load_tape_lmdb(path, max_samples=None):
    import lmdb
    import pickle

    env = lmdb.open(path, readonly=True, lock=False)
    sequences = []
    labels = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (_, val) in enumerate(cursor):
            try:
                example = pickle.loads(val)

                # Skip non-dict entries (e.g., int, str, etc.)
                if not isinstance(example, dict):
                    print(f"⚠️ Skipping entry #{i} (not a dict): {type(example)}")
                    continue

                sequences.append(example["primary"])
                labels.append(example["log_fluorescence"][0])

                if max_samples and len(sequences) >= max_samples:
                    break

            except Exception as e:
                print(f"❌ Failed to parse entry #{i}: {e}")
                continue

    return sequences, labels
