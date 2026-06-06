from uuid import uuid4

from app.models.document import Document


class TextLoader:
    def load(self, doc_path: str) -> Document:
        with open(doc_path) as fp:
            data = fp.read()
            metadata = {"type": "txt", "token_size": len(data.split())}
            did = uuid4()
            return Document(id=did, text=data, source=doc_path, metadata=metadata)
