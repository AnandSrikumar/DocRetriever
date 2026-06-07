import gensim.downloader as api
import numpy as np

model_map = {
    "word2vec": "word2vec-google-news-300",
    "fasttext": "fasttext-wiki-news-subwords-300",
}


class GensimEmbeds:
    def __init__(self):
        self.model_map = {}

    def _get_model(self, model: str):
        if model not in self.model_map:
            self.model_map[model] = api.load(model_map[model])
        return self.model_map[model]

    def _embed_text(self, text: str, model):
        vectors = [model[word] for word in text.split() if word in model]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    def embed_chunks(self, chunks: list[str], model: str) -> np.ndarray:
        if model not in model_map:
            raise AttributeError("Invalid embeddings model")
        model_gensim = self._get_model(model)
        vectors = np.array(
            [self._embed_text(text, model_gensim) for text in chunks], dtype=np.float32
        )
        return vectors
