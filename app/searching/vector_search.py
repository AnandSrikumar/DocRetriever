from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.configs import VECTORIZERS
from app.storage.storage_utils import load_obj
from app.text_preprocess.preprocess import preprocess_text


class VectorSearch:
    def __init__(self, index_loc: str, search_metric: str, backend: str = "pickle"):
        self.vectorizer: TfidfTransformer | CountVectorizer = load_obj(
            f"{index_loc}/{VECTORIZERS[search_metric].model}"
        )
        self.vectors = load_obj(
            f"{index_loc}/{VECTORIZERS[search_metric].get_index_name(backend)}"
        )

    def retrieve_score(self, query: str):
        cleaned = preprocess_text([query])["vectors"]
        q_vector = self.vectorizer.transform(cleaned)  # type: ignore
        scores = cosine_similarity(q_vector, self.vectors)[0]
        return scores
