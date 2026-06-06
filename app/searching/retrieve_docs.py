from bisect import bisect_right
from uuid import UUID

import numpy as np

from app.models.chunk import Chunk
from app.models.document import Document


def _get_page(starts: list[int], offset: int) -> int:
    idx = bisect_right(starts, offset) - 1
    return idx + 1


def generate_pdf_report(doc: Document, chunk: Chunk, score: float) -> dict:
    chunk_start_offset = chunk.start_offset
    chunk_end_offset = chunk.end_offset
    page_offsets = list(doc.metadata["page-offsets"].values())
    ps = _get_page(page_offsets, chunk_start_offset)
    pe = _get_page(page_offsets, chunk_end_offset)
    return {
        "file_type": "pdf",
        "file_name": doc.source,
        "page_start": ps,
        "page_end": pe,
        "score": score,
        "text": chunk.text,
    }


def generate_text_report(doc: Document, chunk: Chunk, score: float):
    return {
        "file_type": "txt",
        "file_name": doc.source,
        "score": score,
        "text": chunk.text,
    }


def generate_docx_report(doc: Document, chunk: Chunk, score: float):
    return {
        "file_type": "docx",
        "file_name": doc.source,
        "score": score,
        "text": chunk.text,
    }


def retrieve_doc(
    indices: np.ndarray,
    scores: list[float],
    chunks: list[Chunk],
    doc_map: dict[UUID, Document],
):
    doc_func_map = {
        "pdf": generate_pdf_report,
        "txt": generate_text_report,
        "docx": generate_docx_report,
    }
    results = []
    for idx, score in zip(indices, scores):
        chunk = chunks[idx]
        doc = doc_map.get(chunk.doc_id)
        if not doc:
            continue
        ext = doc.source.split(".")[-1]
        doc_func = doc_func_map[ext]
        results.append(doc_func(doc, chunk, score))
    return results
