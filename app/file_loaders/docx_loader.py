from uuid import uuid4

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.models.document import Document


class DocxLoader:
    def _iter_blocks(self, doc):
        """Iterates the blocks of docx to ensure the paras and tables are in right order"""
        parent = doc.element.body

        for child in parent.iterchildren():

            if isinstance(child, CT_P):
                yield Paragraph(child, doc)

            elif isinstance(child, CT_Tbl):
                yield Table(child, doc)

    def load(self, doc_path: str) -> Document:
        docx = DocxDocument(doc_path)
        parts = []
        for block in self._iter_blocks(docx):
            if isinstance(block, Paragraph):
                parts.append(block.text)

            elif isinstance(block, Table):
                for row in block.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    parts.append(row_text)

        full_text = "\n".join(parts)
        metadata = {"type": "docx", "token-size": len(full_text.split())}
        document = Document(
            id=uuid4(), source=doc_path, text=full_text, metadata=metadata
        )
        return document
