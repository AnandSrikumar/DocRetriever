from uuid import UUID

from app.chunking.chunker import chunk_docs
from app.configs import (EMBEDDING_MODELS, SENTENCE_EMBEDDING_MODELS,
                         VECTORIZERS)
from app.embeddings.gensim_embeds import GensimEmbeds
from app.embeddings.transformer_embeds import TransformerEmbeds
from app.file_loaders.loader import LoaderFactory
from app.models.chunk import Chunk
from app.models.document import Document
from app.profiling_utils import timeit
from app.storage.storage_utils import save_obj
from app.text_preprocess.preprocess import preprocess_text
from app.vectorizer import vectorize


@timeit
def load_docs(data_path: str, index_loc: str) -> dict[UUID, Document]:
    doc_id_map = LoaderFactory.load(data_path)
    print("Documents loaded...")
    save_obj(doc_id_map, f"{index_loc}/documents.pkl")
    print("Documents .pkl saved....")
    return doc_id_map


@timeit
def chunking(
    doc_map: dict[UUID, Document],
    chunk_type: str,
    chunk_size: int,
    chunk_overlap: int,
    index_loc: str,
) -> list[Chunk]:
    documents = list(doc_map.values())
    chunks = chunk_docs(documents, chunk_type, chunk_size, chunk_overlap)
    print("chunking done.....")
    save_obj(chunks, f"{index_loc}/chunks.pkl")
    return chunks


@timeit
def clean_chunks(chunks: list[Chunk]):
    cleaned = preprocess_text(chunks)
    print("chunks cleaned.....")
    return cleaned


@timeit
def create_vectors(cleaned_chunks: list[str], vector_type: str, index_loc: str):
    if vector_type not in VECTORIZERS:
        raise ValueError("Unsupported vectorizer....")
    vectorizer, vectors = vectorize(cleaned_chunks, vector_type)
    print("vectors created....")
    vector_model_path = f"{index_loc}/{VECTORIZERS[vector_type].model}"
    vectors_path = f'{index_loc}/{VECTORIZERS[vector_type].get_index_name("pickle")}'
    save_obj(vectorizer, vector_model_path)
    save_obj(vectors, vectors_path)
    print("vectors saved.....")


@timeit
def create_word_embeddings(cleaned_chunks: list[str], embedding_type, index_loc: str):
    embed = GensimEmbeds(embedding_type)
    embeddings = embed.embed_chunks(cleaned_chunks)
    print(f"{embedding_type} embeddings created....")
    save_path = (
        f"{index_loc}/{EMBEDDING_MODELS[embedding_type].get_index_name('faiss')}"
    )
    save_obj(embeddings, save_path)
    print(f"{embedding_type} saved....")


@timeit
def create_sentence_embeddings(
    cleaned_chunks: list[str], sent_embed: str, index_loc: str
):
    embed = TransformerEmbeds(sent_embed)
    embeddings = embed.embed_sentence(cleaned_chunks)
    print(f"{sent_embed} embeddings created....")
    save_path = (
        f"{index_loc}/{SENTENCE_EMBEDDING_MODELS[sent_embed].get_index_name('faiss')}"
    )
    save_obj(embeddings, save_path)
    print(f"{sent_embed} saved....")


@timeit
def build_tokenizer(
    retriever: str, cleaned_chunks: dict[str, list[str]], index_loc: str
):
    if retriever in VECTORIZERS:
        create_vectors(cleaned_chunks["vectors"], retriever, index_loc)
    elif retriever in EMBEDDING_MODELS:
        create_word_embeddings(cleaned_chunks["embeds"], retriever, index_loc)
    elif retriever in SENTENCE_EMBEDDING_MODELS:
        create_sentence_embeddings(cleaned_chunks["embeds"], retriever, index_loc)
    else:
        raise ValueError("Unsupported retriever...")


@timeit
def build_index(args):
    data_path = args.data_dir
    chunk_type = args.chunking
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    retriever = args.retriever
    index_loc = args.index_loc

    docs_map = load_docs(data_path, index_loc)
    chunks = chunking(docs_map, chunk_type, chunk_size, chunk_overlap, index_loc)
    cleaned_chunks = clean_chunks(chunks)

    build_tokenizer(retriever, cleaned_chunks, index_loc)
