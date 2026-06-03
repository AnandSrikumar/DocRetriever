from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4

from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)

from app.models.chunk import Chunk
from app.models.document import Document


def chunk_text(
    splitter: CharacterTextSplitter | RecursiveCharacterTextSplitter,
    doc: Document,
    meta: dict,
):
    chunks = splitter.split_text(doc.text)
    chunk_objs = []
    for chunk in chunks:
        c = Chunk(id=uuid4(), text=chunk, metadata=meta, doc_id=doc.id)
        chunk_objs.append(c)
    return chunk_objs


def fixed_split(doc: Document, chunk_size: int, chunk_overlap: int):
    splitter = CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunk_text(splitter, doc, {"chunk-type": "fixed"})


def recurs_split(doc: Document, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        separators=[""], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunk_text(splitter, doc, meta={"chunk-type": "recursive"})


CHUNK_MAP = {
    "fixed": fixed_split,
    "recursive": recurs_split,
}


def chunk_docs(
    docs: list[Document], chunk_type: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunk_func = CHUNK_MAP.get(chunk_type)
    if not chunk_func:
        raise ValueError("Invalid chunk type.....")
    sizes = [chunk_size] * len(docs)  # workaround for parallelism
    overlaps = [chunk_overlap] * len(docs)
    chunks = []
    with ProcessPoolExecutor() as executor:
        chunks = executor.map(chunk_func, docs, sizes, overlaps)
    return [chunk for group in chunks for chunk in group]
