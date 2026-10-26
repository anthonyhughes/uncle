from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from data.frozen_embeddings.embeddings.common import device


def embed_text_bert(texts, model_name="bert-base-uncased", batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text Embeddings"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(features, axis=0)
