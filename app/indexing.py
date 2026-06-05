import time
from argparse import Namespace

from app.chunker import chunk_docs
from app.loader import create_doc_id_map, load_docs
from app.models.document import Document
from app.pickle_util import save_pickle
from app.preprocess import preprocess_text
from app.profiling_utils import timeit
from app.vectorizer import vectorize
from app.word_embeddings import embed_chunks

INDEX_LOC = "indexed/"


@timeit  # type: ignore
def create_docs(data_dir: str) -> list[Document]:
    docs = load_docs(data_dir)
    doc_id_map = create_doc_id_map(docs)
    save_pickle(doc_id_map, f"{INDEX_LOC}/doc_id_map.pkl")
    return docs


@timeit  # type: ignore
def create_chunks(
    docs: list[Document], chunk_type: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    chunks = chunk_docs(docs, chunk_type, chunk_size, chunk_overlap)
    cleaned_chunks = preprocess_text(chunks, True)
    save_pickle(chunks, f"{INDEX_LOC}/chunks.pkl")
    save_pickle(cleaned_chunks, f"{INDEX_LOC}/cleaned_chunks.pkl")
    return cleaned_chunks


@timeit  # type: ignore
def create_vectors(cleaned_chunks: list[str]) -> None:
    tfidf, tfidf_vectors = vectorize(cleaned_chunks, "tfidf")
    bow, bow_vectors = vectorize(cleaned_chunks, "bow")
    save_pickle(tfidf, f"{INDEX_LOC}/tfidf.pkl")
    save_pickle(tfidf_vectors, f"{INDEX_LOC}/tfidf_vectors.pkl")
    save_pickle(bow, f"{INDEX_LOC}/bow.pkl")
    save_pickle(bow_vectors, f"{INDEX_LOC}/bow_vectors.pkl")


@timeit  # type: ignore
def create_embeds(cleaned_chunks: list[str]) -> None:
    w2vec_embeddings = embed_chunks(cleaned_chunks, "word2vec")
    print("w2vec done")
    ftext_embeddings = embed_chunks(cleaned_chunks, "fasttext")
    print("fasttext done...")
    save_pickle(w2vec_embeddings, f"{INDEX_LOC}/word2vec_embeddings.pkl")
    save_pickle(ftext_embeddings, f"{INDEX_LOC}/fasttext_embeddings.pkl")


def build_index(args: Namespace):
    docs = create_docs(args.data_dir)
    print("docs created....")

    cleaned_chunks = create_chunks(
        docs, args.chunking, args.chunk_size, args.chunk_overlap
    )
    print("chunking done...")

    create_vectors(cleaned_chunks)

    print("Vectors done...")

    create_embeds(cleaned_chunks)

    print("embeds done...")
