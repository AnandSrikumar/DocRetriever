import time
from argparse import Namespace

from app.chunking.chunker import chunk_docs
from app.configs import (SENTENCE_EMBEDDINGS_INDEX_PATH, VECTOR_INDEX_PATHS,
                         WORD_EMBEDDING_INDEX_PATHS)
from app.embeddings.gensim_embeds import GensimEmbeds
from app.embeddings.transformer_embeds import TransformerEmbeds
from app.file_loaders.loader import LoaderFactory
from app.models.document import Document
from app.pickle_util import save_pickle
from app.profiling_utils import timeit
from app.text_preprocess.preprocess import preprocess_text
from app.vectorizer import vectorize


@timeit  # type: ignore
def create_docs(data_dir: str, index_loc: str) -> list[Document]:
    docs, doc_id_map = LoaderFactory.load(data_dir)
    save_pickle(doc_id_map, f"{index_loc}/doc_id_map.pkl")
    return docs


@timeit  # type: ignore
def create_chunks(
    docs: list[Document],
    chunk_type: str,
    chunk_size: int,
    chunk_overlap: int,
    index_loc: str,
) -> tuple[list[str], list[str]]:
    chunks = chunk_docs(docs, chunk_type, chunk_size, chunk_overlap)
    cleaned = preprocess_text(chunks)
    cleaned_chunks = cleaned["vectors"]
    cleaned_chunks_embeds = cleaned["embeds"]
    save_pickle(chunks, f"{index_loc}/chunks.pkl")
    save_pickle(cleaned_chunks, f"{index_loc}/cleaned_chunks.pkl")
    save_pickle(cleaned_chunks_embeds, f"{index_loc}/cleaned_chunks_embeds.pkl")
    return cleaned_chunks, cleaned_chunks_embeds


@timeit  # type: ignore
def create_vectors(cleaned_chunks: list[str], index_loc: str) -> None:
    for model, path in VECTOR_INDEX_PATHS.items():
        vectorizer, vectors = vectorize(cleaned_chunks, model)
        save_pickle(vectorizer, f"{index_loc}/{model}.pkl")
        save_pickle(vectors, f"{index_loc}/{path}")
        print(f"vectors: {model} done...")


@timeit  # type: ignore
def create_embeds(cleaned_chunks: list[str], index_loc: str) -> None:
    for model, path in WORD_EMBEDDING_INDEX_PATHS.items():
        gensim_model = GensimEmbeds(model)
        word_embeddings = gensim_model.embed_chunks(cleaned_chunks)
        save_pickle(word_embeddings, f"{index_loc}/{path}")
        print(f"Word embeddings: {model} done....")

    for model, path in SENTENCE_EMBEDDINGS_INDEX_PATH.items():
        sent_model = TransformerEmbeds(model)
        sent_embeddings = sent_model.embed_sentence(cleaned_chunks)
        save_pickle(sent_embeddings, f"{index_loc}/{path}")
        print(f"Sentence embeddings: {model} done....")


def build_index(args: Namespace):
    index_loc = args.index_loc
    docs = create_docs(args.data_dir, index_loc)
    print("docs created....")

    cleaned_chunks, cleaned_chunks_embeds = create_chunks(
        docs, args.chunking, args.chunk_size, args.chunk_overlap, index_loc
    )
    print("chunking done...")

    create_vectors(cleaned_chunks, index_loc)

    print("Vectors done...")

    create_embeds(cleaned_chunks_embeds, index_loc)

    print("embeds done...")
