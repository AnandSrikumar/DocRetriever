import numpy as np
from sentence_transformers import SentenceTransformer

model_map = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


class TransformerEmbeds:
    def __init__(self):
        self.model_map = {k: SentenceTransformer(v) for k, v in model_map.items()}

    def _get_model(self, model: str):
        if model not in self.model_map:
            self.model_map[model] = SentenceTransformer(model_map[model])
        return self.model_map[model]

    def embed_chunks(self, chunks: list[str], model: str) -> np.ndarray:
        if model not in model_map:
            raise AttributeError("Invalid model...")
        sentence_model = self._get_model(model)
        vectors = sentence_model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vectors
