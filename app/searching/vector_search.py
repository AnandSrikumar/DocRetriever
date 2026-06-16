from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.configs import VECTOR_INDEX_PATHS, VECTORIZER_PATHS
from app.pickle_util import load_pickle
from app.text_preprocess.preprocess import preprocess_text


class VectorSearch:
    def __init__(self, index_loc: str, search_metric: str):
        self.vectorizer: TfidfTransformer | CountVectorizer = load_pickle(
            f"{index_loc}/{VECTORIZER_PATHS[search_metric]}"
        )
        self.vectors = load_pickle(f"{index_loc}/{VECTOR_INDEX_PATHS[search_metric]}")

    def retrieve_score(self, query: str):
        cleaned = preprocess_text([query])["vectors"]
        q_vector = self.vectorizer.transform(cleaned)  # type: ignore
        scores = cosine_similarity(q_vector, self.vectors)[0]
        return scores
