from typing import Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VectorizerType = Union[
    TfidfVectorizer,
    CountVectorizer,
]

vector_map = {"tfidf": TfidfVectorizer, "bow": CountVectorizer}


def vectorize(chunks: list[str], vec_type: str):
    vec = vector_map.get(vec_type)
    if not vec:
        raise AttributeError("Invalid vector type: only tfidf and bow are available")
    vec_obj: VectorizerType = vec()
    vectors = vec_obj.fit_transform(chunks)
    return vec_obj, vectors


def vectorize_query(query: str, vectorizer: VectorizerType):
    return vectorizer.transform([query])
