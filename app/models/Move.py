from pydantic import BaseModel
from typing import Dict, Optional

class AnalysisStage:
    SHALLOW = "shallow"
    DEEP = "deep"
    FINAL = "final"

class Move(BaseModel):
    id: int
    position: str
    move: str
    context: str
    depth: int
    isAnalyzed: bool
    piece: str
    score: Optional[float] = None
    trace: Optional[Dict] = None
    phase: Optional[str] = None
    capturedByWhite: Optional[Dict[str, int]] = None
    capturedByBlack: Optional[Dict[str, int]] = None
    analysisStage: Optional[str] = None  # AnalysisStage
    analysisVersion: Optional[int] = None
    hiddenFeatures: Optional[Dict] = None
