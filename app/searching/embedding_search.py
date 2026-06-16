from sklearn.metrics.pairwise import cosine_similarity

from app.configs import WORD_EMBEDDING_INDEX_PATHS, SENTENCE_EMBEDDINGS_INDEX_PATH
from app.embeddings.gensim_embeds import GensimEmbeds
from app.embeddings.transformer_embeds import TransformerEmbeds
from app.pickle_util import load_pickle
from app.text_preprocess.preprocess import preprocess_text

import gensim.downloader as api


class WordEmbeddingSearch:
    def __init__(self, index_loc: str, search_metric: str):
        self.model = GensimEmbeds(search_metric)
        self.embeddings = load_pickle(f"{index_loc}/{WORD_EMBEDDING_INDEX_PATHS[search_metric]}")

    def retrieve_score(self, query: str):
        cleaned = preprocess_text([query])["embeds"]
        embeds = self.model.embed_chunks(cleaned)[0]
        scores = cosine_similarity(embeds.reshape(1, -1), self.embeddings).ravel()
        return scores


class SentenceEmbeddingSearch:
    def __init__(self, index_loc: str, search_metric: str):
        self.model = TransformerEmbeds(search_metric)
        self.embeddings = load_pickle(f"{index_loc}/{SENTENCE_EMBEDDINGS_INDEX_PATH[search_metric]}")

    def retrieve_score(self, query: str):
        cleaned = preprocess_text([query])["embeds"]
        embeds = self.model.embed_sentence(cleaned)[0]
        scores = cosine_similarity(embeds.reshape(1, -1), self.embeddings).ravel()
        return scores
        