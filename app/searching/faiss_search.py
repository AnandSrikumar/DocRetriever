import numpy as np

from app.embeddings.gensim_embeds import GensimEmbeds
from app.embeddings.transformer_embeds import TransformerEmbeds
from app.storage.storage_utils import load_obj
from app.text_preprocess.preprocess import preprocess_text

MODEL_MAP = {
    "word2vec": GensimEmbeds,
    "fasttext": GensimEmbeds,
    "all-minilm": TransformerEmbeds,
}


class FaissSearch:
    def __init__(self, index_path, model: str):
        self.model = MODEL_MAP[model](model)
        print(f"about to load {index_path}")
        self.index = load_obj(index_path)

    def retrieve_score(self, query: str, top_k):
        cleaned = preprocess_text([query])["embeds"]
        query_vector = self.model.embed_sentence(cleaned)[0]
        scores, ids = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32),
            top_k,
        )
        return ids[0], scores[0]
