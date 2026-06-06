from uuid import uuid4

import pymupdf

from app.models.document import Document


class PdfLoader:
    def load(self, doc_path: str) -> Document:
        full_text = ""
        page_offset_map = {}
        offset = 0
        pages = 0
        with pymupdf.open(doc_path) as doc:
            for page_num, page in enumerate(doc, start=1):  # type: ignore
                page_text = page.get_text()
                full_text += page_text
                page_offset_map[page_num] = offset
                offset += len(page_text)
                pages += 1
        did = uuid4()
        metadata = {
            "type": "pdf",
            "num-pages": pages,
            "page-offsets": page_offset_map,
            "token-size": len(full_text.split()),
        }
        document = Document(id=did, metadata=metadata, text=full_text, source=doc_path)
        return document
