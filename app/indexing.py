import time
from argparse import Namespace

from app.chunking.chunker import chunk_docs
from app.file_loaders.loader import LoaderFactory
from app.models.document import Document
from app.pickle_util import save_pickle
from app.profiling_utils import timeit
from app.text_preprocess.preprocess import preprocess_text
from app.vectorizer import vectorize
from app.word_embeddings import embed_chunks


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
    cleaned_chunks = preprocess_text(chunks, "vector")
    cleaned_chunks_embeds = preprocess_text(chunks, "embed")
    save_pickle(chunks, f"{index_loc}/chunks.pkl")
    save_pickle(cleaned_chunks, f"{index_loc}/cleaned_chunks.pkl")
    save_pickle(cleaned_chunks_embeds, f"{index_loc}/cleaned_chunks_embeds.pkl")
    return cleaned_chunks, cleaned_chunks_embeds


@timeit  # type: ignore
def create_vectors(cleaned_chunks: list[str], index_loc: str) -> None:
    tfidf, tfidf_vectors = vectorize(cleaned_chunks, "tfidf")
    bow, bow_vectors = vectorize(cleaned_chunks, "bow")
    save_pickle(tfidf, f"{index_loc}/tfidf.pkl")
    save_pickle(tfidf_vectors, f"{index_loc}/tfidf_vectors.pkl")
    save_pickle(bow, f"{index_loc}/bow.pkl")
    save_pickle(bow_vectors, f"{index_loc}/bow_vectors.pkl")


@timeit  # type: ignore
def create_embeds(cleaned_chunks: list[str], index_loc: str) -> None:
    w2vec_embeddings = embed_chunks(cleaned_chunks, "word2vec")
    print("w2vec done")
    ftext_embeddings = embed_chunks(cleaned_chunks, "fasttext")
    print("fasttext done...")
    save_pickle(w2vec_embeddings, f"{index_loc}/word2vec_embeddings.pkl")
    save_pickle(ftext_embeddings, f"{index_loc}/fasttext_embeddings.pkl")


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
