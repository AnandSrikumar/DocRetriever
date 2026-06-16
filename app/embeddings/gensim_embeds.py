import gensim.downloader as api
import numpy as np

from app.configs import WORD_EMBEDDING_MODELS


class GensimEmbeds:
    def __init__(self, model: str):
        if model not in WORD_EMBEDDING_MODELS:
            raise ValueError("Invalid word embedding model")
        self.model = api.load(WORD_EMBEDDING_MODELS[model])

    def embed_chunk(self, chunk: str):
        vectors = [self.model[word] for word in chunk.split() if word in self.model]  # type: ignore
        if not vectors:
            return np.zeros(self.model.vector_size)  # type: ignore
        return np.mean(vectors, axis=0)  # type: ignore

    def embed_chunks(self, chunks: list[str]):
        vectors = np.array(
            [self.embed_chunk(text) for text in chunks], dtype=np.float32
        )
        return vectors
