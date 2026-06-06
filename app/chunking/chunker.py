from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from app.chunking.langchain_chunks import fixed_split, recurs_split
from app.models.chunk import Chunk
from app.models.document import Document

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
    chunks = []
    with ProcessPoolExecutor() as executor:
        chunks = executor.map(
            chunk_func,
            docs,
            repeat(chunk_size),
            repeat(chunk_overlap),
        )
    return [chunk for group in chunks for chunk in group]
