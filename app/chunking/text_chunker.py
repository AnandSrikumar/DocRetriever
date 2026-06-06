from uuid import uuid4

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.models.chunk import Chunk
from app.models.document import Document


def chunk_text(
    splitter: CharacterTextSplitter | RecursiveCharacterTextSplitter,
    doc: Document,
    chunk_type: str,
):
    text = doc.text
    chunks = splitter.split_text(text)
    current_offset = 0
    chunk_objs = []
    for chunk in chunks:
        start_offset = text.find(chunk, current_offset)
        end_offset = start_offset + len(chunk)
        current_offset = end_offset
        c = Chunk(
            id=uuid4(),
            text=chunk,
            metadata={
                "type": chunk_type,
                "len": len(chunk),
                "token-size": len(chunk.split()),
            },
            doc_id=doc.id,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        chunk_objs.append(c)
    return chunk_objs
