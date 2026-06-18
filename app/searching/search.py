import numpy as np

from app.configs import (CHUNKS_PATHS, DOCUMENTS_PATH, EMBEDDING_MODELS,
                         SENTENCE_EMBEDDING_MODELS, VECTORIZERS)
from app.searching.faiss_search import FaissSearch
from app.searching.retrieve_docs import retrieve_doc
from app.searching.vector_search import VectorSearch
from app.storage.storage_utils import load_obj


class Search:
    def __init__(self, args):
        self.top_k = args.top_k
        self.retriever = args.retriever
        self.index_loc = args.index_loc
        self._load_searcher()
        self._load_docs()

    def _load_docs(self):
        self.chunks = load_obj(f"{self.index_loc}/{CHUNKS_PATHS}")
        self.documents = load_obj(f"{self.index_loc}/{DOCUMENTS_PATH}")
        print(f"docs and chunks loaded")

    def _load_searcher(self):
        if self.retriever in VECTORIZERS:
            self.searcher = VectorSearch(self.index_loc, self.retriever)
            print(f"Vectorizer {self.retriever} loaded")

        elif self.retriever in EMBEDDING_MODELS:
            index_path = f"{EMBEDDING_MODELS[self.retriever].get_index_name('faiss')}"
            self.searcher = FaissSearch(index_path, self.retriever)
            print(f"word embedding model {self.retriever} loaded")

        elif self.retriever in SENTENCE_EMBEDDING_MODELS:
            index_path = (
                f"{SENTENCE_EMBEDDING_MODELS[self.retriever].get_index_name('faiss')}"
            )
            self.searcher = FaissSearch(self.index_loc, self.retriever)
            print(f"Sentence embedding model {self.retriever} loaded")

        else:
            raise ValueError("Unsupported retriever....")

    def search(self, query: str):
        idx, scores = self.searcher.retrieve_score(query, self.top_k)
        docs = retrieve_doc(idx, scores, self.chunks, self.documents)
        return docs
