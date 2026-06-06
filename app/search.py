import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.models.chunk import Chunk
from app.models.document import Document
from app.pickle_util import load_pickle
from app.preprocess import preprocess_text
from app.profiling_utils import timeit


def pdf_search(document: Document):
    relevants = []


class Search:
    def __init__(self, args):
        self.indexed_loc = args.index_loc
        self.top_k = args.top_k
        self.search_method = args.search_method
        self.vectors = {"tfidf", "bow"}
        self.embeds = {"fasttext", "word2vec"}
        self.load_pickles()

    def load_pickles(self):
        self.doc_map = load_pickle(f"{self.indexed_loc}doc_id_map.pkl")
        self.chunks = load_pickle(f"{self.indexed_loc}chunks.pkl")
        self.cleaned_chunks = load_pickle(f"{self.indexed_loc}cleaned_chunks.pkl")
        self.tfidf = load_pickle(f"{self.indexed_loc}tfidf.pkl")
        self.tfidf_vectors = load_pickle(f"{self.indexed_loc}tfidf_vectors.pkl")
        self.bow = load_pickle(f"{self.indexed_loc}bow.pkl")
        self.bow_vectors = load_pickle(f"{self.indexed_loc}bow_vectors.pkl")
        self.w2vec = load_pickle(f"{self.indexed_loc}word2vec_embeddings.pkl")
        self.fasttext = load_pickle(f"{self.indexed_loc}fasttext_embeddings.pkl")
        self.vector_map = {"tfidf": self.tfidf, "bow": self.bow}
        self.embed_map = {"fasttext": self.fasttext, "word2vec": self.w2vec}

    def _get_relevant_docs(self, indices, scores):
        for idx in indices:
            print(scores[idx])
            chunk = self.chunks[idx]
            print(chunk.text)
            doc = self.doc_map.get(chunk.doc_id)
            if doc:
                print(doc.source)

    def search_vectors(self, query: str, vector_type: str):
        vectorizer = self.vector_map.get(vector_type)
        if not vectorizer:
            raise AttributeError(
                "Invalid vectorizer.... Only tfidf and bow are available"
            )
        cleaned_query = preprocess_text([query], True)
        q_vector = vectorizer.transform(cleaned_query)
        scores = cosine_similarity(q_vector, self.tfidf_vectors)
        scores = scores.flatten()
        indices = np.argpartition(scores, -self.top_k)[-self.top_k :]
        self._get_relevant_docs(indices, scores)

    def search(self, query: str):
        if self.search_method in self.vectors:
            self.search_vectors(query, self.search_method)
