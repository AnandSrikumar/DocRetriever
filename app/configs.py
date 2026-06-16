WORD_EMBEDDING_MODELS = {
    "word2vec": "word2vec-google-news-300",
    "fasttext": "fasttext-wiki-news-subwords-300",
}

WORD_EMBEDDING_INDEX_PATHS = {
    "word2vec": "word2vec_embeddings.pkl",
    "fasttext": "fasttext_embeddings.pkl",
}

SENTENCE_EMBEDDING_MODELS = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

SENTENCE_EMBEDDINGS_INDEX_PATH = {"all-minilm": "all_minilm_embeddings.pkl"}

VECTORIZER_PATHS = {"tfidf": "tfidf.pkl", "bow": "bow.pkl"}

VECTOR_INDEX_PATHS = {"tfidf": "tfidf_vectors.pkl", "bow": "bow_vectors.pkl"}
