import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.embeddings.word_embedings import Embed, get_model_names
from app.searching.artifacts import ArtifactLoader
from app.searching.retrieve_docs import retrieve_doc
from app.text_preprocess.preprocess import preprocess_text


class Search:
    def __init__(self, args):
        self.search_metric = args.search_method
        self.index_loc = args.index_loc
        self.top_k = int(args.top_k)
        self.artifacts = ArtifactLoader.load(self.index_loc)
        self.vector_map = {"tfidf": self.artifacts.tfidf, "bow": self.artifacts.bow}
        self.vectors = {
            "tfidf": self.artifacts.tfidf_vectors,
            "bow": self.artifacts.bow_vectors,
        }
        self.embed_map = {
            "fasttext": self.artifacts.fasttext_embeddings,
            "word2vec": self.artifacts.word2vec_embeddings,
        }
        self.embed_model_map = None

    def load_gensim_models(self):
        self.embed = Embed()

    def search_vectors(self, query: str):
        vectorizer = self.vector_map.get(self.search_metric)
        pretrained_vectors = self.vectors.get(self.search_metric)
        cleaned_query = preprocess_text([query], "vector")
        q_vector = vectorizer.transform(cleaned_query)  # type: ignore
        scores = cosine_similarity(q_vector, pretrained_vectors)[0]  # type: ignore
        k = min(self.top_k, len(scores))
        top_idxs = np.argpartition(scores, -k)[-k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]
        top_scores = scores[top_idxs]
        return top_idxs, top_scores

    def search_embeds(self, query: str):
        cleaned_query = preprocess_text([query], "vector")[0]
        q_vector = self.embed.embed_chunks([cleaned_query], self.search_metric)[0]
        scores = cosine_similarity(
            q_vector.reshape(1, -1),
            embeds,  # type: ignore
        )[0]

        k = min(self.top_k, len(scores))

        top_idxs = np.argpartition(
            scores,
            -k,
        )[-k:]

        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        top_scores = scores[top_idxs]

        return top_idxs, top_scores

    def search(self, query: str):
        vectors = {"tfidf", "bow"}
        if self.search_metric in vectors:
            top_idx, top_scores = self.search_vectors(query)
            results = retrieve_doc(
                top_idx, top_scores, self.artifacts.chunks, self.artifacts.doc_map
            )
            print(results)
            return results
        if self.embed is None:
            self.load_gensim_models()
        top_idx, top_scores = self.search_embeds(query)
        return retrieve_doc(
            top_idx, top_scores, self.artifacts.chunks, self.artifacts.doc_map
        )
