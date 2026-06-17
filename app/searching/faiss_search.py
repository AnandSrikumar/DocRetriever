from app.storage.storage_utils import load_obj

import numpy as np


class FaissSearch:
    def __init__(self, index_path):
        self.index = load_obj(index_path)

    def search(self, query_vector, top_k):
        scores, ids = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32),
            top_k,
        )
        return scores[0], ids[0]
