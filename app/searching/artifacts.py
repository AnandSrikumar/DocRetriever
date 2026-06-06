from dataclasses import dataclass

from app.pickle_util import load_pickle


@dataclass
class SearchArtifacts:
    doc_map: dict

    chunks: list
    cleaned_chunks: list

    tfidf: object
    tfidf_vectors: object

    bow: object
    bow_vectors: object

    word2vec_embeddings: object
    fasttext_embeddings: object


class ArtifactLoader:

    @staticmethod
    def load(indexed_loc: str) -> SearchArtifacts:
        return SearchArtifacts(
            doc_map=load_pickle(f"{indexed_loc}doc_id_map.pkl"),
            chunks=load_pickle(f"{indexed_loc}chunks.pkl"),
            cleaned_chunks=load_pickle(f"{indexed_loc}cleaned_chunks.pkl"),
            tfidf=load_pickle(f"{indexed_loc}tfidf.pkl"),
            tfidf_vectors=load_pickle(f"{indexed_loc}tfidf_vectors.pkl"),
            bow=load_pickle(f"{indexed_loc}bow.pkl"),
            bow_vectors=load_pickle(f"{indexed_loc}bow_vectors.pkl"),
            word2vec_embeddings=load_pickle(f"{indexed_loc}word2vec_embeddings.pkl"),
            fasttext_embeddings=load_pickle(f"{indexed_loc}fasttext_embeddings.pkl"),
        )
