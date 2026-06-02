from app.loader import load_docs, pdf_loader, text_loader, docx_loader
from app.models.document import Document
import pytest


def test_text_prepare_doc():
    doc = text_loader("data/alt.atheism.txt")
    assert isinstance(doc, Document)
    assert doc.source == "data/alt.atheism.txt"
    assert doc.metadata["type"] == "txt"


def test_pdf_loader():
    doc = pdf_loader("data/harry_potter.pdf")
    assert isinstance(doc, Document)
    assert "page-offsets" in doc.metadata
    assert doc.metadata["num-pages"] == 250


def test_docx_loader():
    doc = docx_loader("data/redis_caching.docx")
    assert isinstance(doc, Document)
    assert "token-size" in doc.metadata
    assert doc.source == "data/redis_caching.docx"


def test_load_docs():
    docs = load_docs("data/")
    assert isinstance(docs, list)
    assert isinstance(docs[0], Document)
