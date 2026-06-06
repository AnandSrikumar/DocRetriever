from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.chunking.text_chunker import chunk_text
from app.models.document import Document


def fixed_split(doc: Document, chunk_size: int, chunk_overlap: int):
    splitter = CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunk_text(splitter, doc, "fixed")


def recurs_split(doc: Document, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        separators=[""], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunk_text(splitter, doc, "recursive")
