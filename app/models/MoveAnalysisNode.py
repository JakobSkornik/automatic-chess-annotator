from pydantic import BaseModel
from typing import Dict


class MoveAnalysisNode(BaseModel):
    id: int
    depth: int
    parent: int
    move: str
    fen: str
    shallow_score: float
    deep_score: float
    trace: Dict
    piece: str | None
    context: str
    phase: str
    capturedByWhite: Dict[str, int]
    capturedByBlack: Dict[str, int]

    def __repr__(self):
        return (
            f"Node(id={self.id}, depth={self.depth}, parent={self.parent}, move='{self.move}', "
            f"shallow_score={self.shallow_score}, deep_score={self.deep_score}, trace={self.trace})"
        )
