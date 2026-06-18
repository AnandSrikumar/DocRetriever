import pickle
from pathlib import Path

import faiss
import numpy as np


def _serialize_pickle(obj, path: str):  # type: ignore
    path: Path = Path(path)  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _serialize_faiss(vectors, path: str):
    embeddings = np.asarray(
        vectors,
        dtype=np.float32,
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)


def _deserialize_faiss(path: str):
    return faiss.read_index(path)


LOAD_HANDLERS = {"pkl": _deserialize_pickle, "index": _deserialize_faiss}

SAVE_HANDLERS = {"pkl": _serialize_pickle, "index": _serialize_faiss}


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
