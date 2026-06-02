import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: uuid.UUID
    source: str
    text: str
    metadata: dict[str, Any]
