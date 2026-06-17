import pickle
from pathlib import Path

import faiss
import numpy as np


def __serialize_pickle(obj, path: str):  # type: ignore
    path: Path = Path(path)  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def __deserialize_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def __serialize_faiss(vectors, path: str):
    if hasattr(vectors, "toarray"):
        vectors = vectors.toarray()

    embeddings = np.asarray(
        vectors,
        dtype=np.float32,
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, path)


def __deserialize_faiss(path: str):
    return faiss.read_index(path)


LOAD_HANDLERS = {"pkl": __deserialize_pickle, "index": __deserialize_faiss}

SAVE_HANDLERS = {"pkl": __serialize_pickle, "index": __serialize_faiss}


def save_obj(vectors, path: str):
    suffix = path.split(".")[-1]
    if suffix not in SAVE_HANDLERS:
        raise ValueError("Invalid vector/embeddings....")
    SAVE_HANDLERS[suffix](vectors, path)


def load_obj(path: str):
    suffix = path.split(".")[-1]
    if suffix not in LOAD_HANDLERS:
        raise ValueError("Invalid vector/embeddings....")
    return LOAD_HANDLERS[suffix](path)
