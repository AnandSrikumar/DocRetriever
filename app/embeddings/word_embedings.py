from app.embeddings.gensim_embeds import GensimEmbeds
from app.embeddings.gensim_embeds import model_map as gensim_models
from app.embeddings.transformer_embeds import TransformerEmbeds
from app.embeddings.transformer_embeds import model_map as transformer_models


def get_model_names(model: str):
    if model in gensim_models:
        return gensim_models[model]
    if model in transformer_models:
        return transformer_models[model]
    raise AttributeError("Invalid embedding model")


class Embed:
    def __init__(self):
        self.gensim = GensimEmbeds()
        self.transformer = TransformerEmbeds()

    def embed_chunks(self, chunks: list[str], model: str):
        if model in gensim_models:
            return self.gensim.embed_chunks(chunks, model)
        if model in transformer_models:
            return self.transformer.embed_chunks(chunks, model)
        raise AttributeError("Invalid embedding model")
