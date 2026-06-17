from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    index_name: str

    def get_index_name(self, backend: str) -> str:
        ext_map = {"pickle": ".pkl", "faiss": ".index"}
        if backend not in ext_map:
            raise ValueError("Unsupported backend.....")
        return self.index_name + ext_map[backend]


EMBEDDING_MODELS = {
    "word2vec": ModelConfig(
        model="word2vec-google-news-300", index_name="word2vec_embeddings"
    ),
    "fasttext": ModelConfig(
        model="fasttext-wiki-news-subwords-300", index_name="fasttext_embeddings"
    ),
}

SENTENCE_EMBEDDING_MODELS = {
    "all-minilm": ModelConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_name="all_minilm_embeddings",
    )
}

VECTORIZERS = {
    "tfidf": ModelConfig(model="tfidf.pkl", index_name="tfidf_vectors"),
    "bow": ModelConfig(model="bow.pkl", index_name="bow_vectors"),
}
