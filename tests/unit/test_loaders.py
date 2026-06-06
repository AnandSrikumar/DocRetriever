from app.file_loaders.text_loader import TextLoader
from app.file_loaders.docx_loader import DocxLoader
from app.file_loaders.pdf_loader import PdfLoader
from app.file_loaders.loader import LoaderFactory
from app.models.document import Document
import pytest


def test_text_prepare_doc():
    txt = TextLoader()
    doc = txt.load("data/alt.atheism.txt")
    assert isinstance(doc, Document)
    assert doc.source == "data/alt.atheism.txt"
    assert doc.metadata["type"] == "txt"


def test_pdf_loader():
    p = PdfLoader()
    doc = p.load("data/harry_potter.pdf")
    assert isinstance(doc, Document)
    assert "page-offsets" in doc.metadata
    assert doc.metadata["num-pages"] == 250


def test_docx_loader():
    d = DocxLoader()
    doc = d.load("data/redis_caching.docx")
    assert isinstance(doc, Document)
    assert "token-size" in doc.metadata
    assert doc.source == "data/redis_caching.docx"


def test_load_docs():
    docs, doc_map = LoaderFactory.load("data/")
    assert isinstance(docs, list)
    assert isinstance(docs[0], Document)
    assert isinstance(doc_map, dict)
