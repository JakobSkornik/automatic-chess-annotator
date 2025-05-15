from pydantic import BaseModel
from typing import List, Dict
from app.models.Move import Move
from app.models.MoveAnalysisNode import MoveAnalysisNode
from app.models.PgnMetadata import PgnMetadata

class AnalysisResponse(BaseModel):
    metadata: PgnMetadata
    moves: List[Move]
    move_tree: Dict[int, MoveAnalysisNode]
