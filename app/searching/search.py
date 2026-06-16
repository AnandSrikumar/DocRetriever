from typing import Protocol

import numpy as np

from app.configs import (SENTENCE_EMBEDDINGS_INDEX_PATH, VECTOR_INDEX_PATHS,
                         WORD_EMBEDDING_INDEX_PATHS)
from app.pickle_util import load_pickle
from app.searching.embedding_search import SentenceEmbeddingSearch, WordEmbeddingSearch
from app.searching.retrieve_docs import retrieve_doc
from app.searching.vector_search import VectorSearch


class Retriever(Protocol):
    def retrieve_score(self, query): ...


class Search:
    def __init__(self, args):
        self.search_metric = args.search_method
        self.top_k = args.top_k
        self.index_loc = args.index_loc
        self.retrieve: Retriever | None = None  # type: ignore
        self._load_retriever()
        self._load_docs()

    def _load_docs(self):
        self.chunks = load_pickle(f"{self.index_loc}/chunks.pkl")
        self.docs = load_pickle(f"{self.index_loc}/doc_id_map.pkl")

    def _load_retriever(self):
        if (
            self.search_metric not in VECTOR_INDEX_PATHS
            and self.search_metric not in WORD_EMBEDDING_INDEX_PATHS
            and self.search_metric not in SENTENCE_EMBEDDINGS_INDEX_PATH
        ):
            raise ValueError("Invalid search metric")
        if self.search_metric in VECTOR_INDEX_PATHS:
            self.retrieve: Retriever = VectorSearch(self.index_loc, self.search_metric)

        elif self.search_metric in WORD_EMBEDDING_INDEX_PATHS:
            self.retrieve: Retriever = WordEmbeddingSearch(self.index_loc, self.search_metric)

        elif self.search_metric in SENTENCE_EMBEDDINGS_INDEX_PATH:
            self.retrieve: Retriever = SentenceEmbeddingSearch(self.index_loc, self.search_metric)


    def _get_top_k(self, scores):
        k = min(self.top_k, len(scores))
        top_idxs = np.argpartition(scores, -k)[-k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]
        top_scores = scores[top_idxs]
        return top_idxs, top_scores

    def search(self, query: str):
        scores = self.retrieve.retrieve_score(query)
        top_idx, top_scores = self._get_top_k(scores)
        docs = retrieve_doc(top_idx, top_scores, self.chunks, self.docs)
        return docs
