from sklearn.metrics.pairwise import cosine_similarity

from app.configs import EMBEDDING_MODELS, SENTENCE_EMBEDDING_MODELS
from app.embeddings.gensim_embeds import GensimEmbeds
from app.embeddings.transformer_embeds import TransformerEmbeds
from app.storage.storage_utils import load_obj
from app.text_preprocess.preprocess import preprocess_text

import gensim.downloader as api


class WordEmbeddingSearch:
    def __init__(self, index_loc: str, search_metric: str, backend: str = "pickle"):
        self.model = GensimEmbeds(search_metric)
        self.embeddings = load_obj(
            f"{index_loc}/{EMBEDDING_MODELS[search_metric].get_index_name(backend)}"
        )

    def retrieve_score(self, query: str):
        cleaned = preprocess_text([query])["embeds"]
        embeds = self.model.embed_chunks(cleaned)[0]
        scores = cosine_similarity(embeds.reshape(1, -1), self.embeddings).ravel()
        return scores


class SentenceEmbeddingSearch:
    def __init__(self, index_loc: str, search_metric: str, backend: str = "pickle"):
        self.model = TransformerEmbeds(search_metric)
        self.embeddings = load_obj(
            f"{index_loc}/{SENTENCE_EMBEDDING_MODELS[search_metric].get_index_name(backend)}"
        )

    def retrieve_score(self, query: str):
        cleaned = preprocess_text([query])["embeds"]
        embeds = self.model.embed_sentence(cleaned)[0]
        scores = cosine_similarity(embeds.reshape(1, -1), self.embeddings).ravel()
        return scores
