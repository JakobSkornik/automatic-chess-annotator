from pydantic import BaseModel

from app.models.Move import Move

class MoveAnalysisNode(BaseModel):
    id: int
    parent: int
    move: Move
    depth: int
    piece: str | None
