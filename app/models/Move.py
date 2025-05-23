from pydantic import BaseModel
from typing import Dict, List, Optional

class PV(BaseModel):
    score: float
    moves: List[str]

class Move(BaseModel):
    position: str
    move: str
    context: str
    isAnalyzed: bool
    score: Optional[float] = None
    pvs: Optional[List[PV]] = None
    trace: Optional[Dict] = None
    phase: Optional[str] = None
    capturedByWhite: Optional[Dict[str, int]] = None
    capturedByBlack: Optional[Dict[str, int]] = None
