from pydantic import BaseModel


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
    score: float | None = None
    phase: str | None = None
    capturedByWhite: dict[str, int] | None = None
    capturedByBlack: dict[str, int] | None = None
    analysisStage: str | None = None  # AnalysisStage
    analysisVersion: int | None = None
    hiddenFeatures: dict | None = None
