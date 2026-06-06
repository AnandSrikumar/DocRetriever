from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass
class Chunk:
    id: UUID
    doc_id: UUID
    text: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]
