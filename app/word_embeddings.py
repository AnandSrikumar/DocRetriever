from concurrent.futures import ProcessPoolExecutor
from functools import partial

import gensim.downloader as api
import numpy as np

model_map = {
    "word2vec": "word2vec-google-news-300",
    "fasttext": "fasttext-wiki-news-subwords-300",
}


def embed_text(text: str, model):
    vectors = [model[word] for word in text.split() if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def embed_chunks(chunks: list[str], model: str):
    model_gensim = model_map.get(model)
    if not model_gensim:
        raise AttributeError("Invalid embeddings model")
    api_model = api.load(model_gensim)
    vectors = np.array(
        [embed_text(text, api_model) for text in chunks], dtype=np.float32
    )
    return vectors
