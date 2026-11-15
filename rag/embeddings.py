"""Step 4: BGE-M3 dense embeddings (normalised, so inner product = cosine)."""
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size: int = 16) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return vecs.astype("float32")
