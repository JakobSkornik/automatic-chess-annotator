from pydantic import BaseModel
from typing import List, Optional, Union

class MoveScore(BaseModel):
    cp: Optional[int] = None
    mate: Optional[int] = None

class Variation(BaseModel):
    rank: int
    move_san: str
    score: Optional[MoveScore] = None
    line: List[str]

class GameMove(BaseModel):
    mn: int
    color: str
    san: str
    uci: str
    fen: str
    score: Optional[MoveScore] = None
    variations: List[Variation] = []
    comment: Optional[str] = None
    classification: Optional[str] = None

class GameMetadata(BaseModel):
    id: str
    white: str
    black: str
    result: str
    date: Optional[str] = None
    eventId: Optional[str] = None
    whiteElo: Optional[int] = None
    blackElo: Optional[int] = None
    opening: Optional[str] = None

class AnalysisInfo(BaseModel):
    engine: str
    depth: int
    multipv: int
    timestamp: float

class GameJson(BaseModel):
    metadata: GameMetadata
    moves: List[GameMove]
    analysis_info: AnalysisInfo



