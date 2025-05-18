from pydantic import BaseModel
from typing import Dict, List

class Move(BaseModel):
    position: str
    move: str
    shallow_score: float
    deep_score: float
    bestContinuations: List[Dict[str, str | float]]
    trace: Dict
    phase: str
    capturedByWhite: Dict[str, int]
    capturedByBlack: Dict[str, int]
