import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Protocol

from app.file_loaders.docx_loader import DocxLoader
from app.file_loaders.pdf_loader import PdfLoader
from app.file_loaders.text_loader import TextLoader
from app.models.document import Document

loader_map = {"txt": TextLoader(), "pdf": PdfLoader(), "docx": DocxLoader()}


class Loader(Protocol):
    def load(self) -> Document: ...


def _get_paths(docs_path: str):
    paths = []
    for file_path in Path(docs_path).iterdir():
        if not file_path.is_file():
            continue
        paths.append(file_path)
    return paths


def _parallel_load(file_path: Path) -> Document | None:
    extension = file_path.suffix.lower().lstrip(".")
    loader: Loader | None = loader_map.get(extension)
    if loader is None:
        print(f"WARNING: Unsupported file type: {file_path.name}")
        return
    try:
        document = loader.load(str(file_path))
        return document

    except Exception as exc:
        print(f"WARNING: Failed to load " f"{file_path.name}: {exc}")
        return


class LoaderFactory:
    @staticmethod
    def load(docs_path: str):
        documents = []
        paths = _get_paths(docs_path)
        s = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            documents = executor.map(_parallel_load, paths)
        documents = [doc for doc in documents if doc is not None]
        print(
            "docs loaded:",
            len(documents),
            "loading took",
            (time.perf_counter() - s),
            "seconds",
        )
        return documents, {doc.id: doc for doc in documents}
