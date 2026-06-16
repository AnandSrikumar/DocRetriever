import numpy as np
from sentence_transformers import SentenceTransformer

from app.configs import SENTENCE_EMBEDDING_MODELS


class TransformerEmbeds:
    def __init__(self, model):
        if model not in SENTENCE_EMBEDDING_MODELS:
            raise ValueError("Invalid sentence transformer model")
        self.model = SentenceTransformer(SENTENCE_EMBEDDING_MODELS[model])

    def embed_sentence(self, chunks: list[str]):
        vectors = self.model.encode(
            chunks, normalize_embeddings=True, convert_to_numpy=True, batch_size=64
        ).astype(np.float32)
        return vectors
