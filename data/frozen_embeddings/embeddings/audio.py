import numpy as np
import openl3
import soundfile as sf
from tqdm import tqdm
from data.frozen_embeddings.embeddings.common import device

def embed_audio_openl3(filepaths, input_repr="mel256", content_type="env", embedding_size=512):
    """
    Compute audio embeddings from a list of .wav filepaths using OpenL3.

    Args:
        filepaths (List[str]): List of paths to audio files.
        input_repr (str): 'mel128' or 'mel256'
        content_type (str): 'env' (environmental) or 'music'
        embedding_size (int): 512 or 6144

    Returns:
        np.ndarray: Audio embeddings (one per file)
    """

    features = []
    for path in tqdm(filepaths, desc="Computing audio embeddings"):
        audio, sr = sf.read(path)
        # Convert stereo to mono if necessary
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        emb, _ = openl3.get_audio_embedding(
            audio,
            sr,
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size
        )
        # Mean pooling over time
        emb = emb.mean(axis=0)
        features.append(emb)

    return np.array(features)
