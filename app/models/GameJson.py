from pydantic import BaseModel
from typing import Any, Dict, List, Optional

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
    move_quality: Optional[str] = None
    event_type: Optional[str] = None
    tactical_motifs: List[str] = []
    strategic_motifs: List[str] = []
    move_category: Optional[str] = None
    plan_comparison: Optional[Dict[str, Any]] = None
    is_critical: bool = False
    episode_index: Optional[int] = None

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

class EpisodeSummary(BaseModel):
    episode_index: int
    title: str
    start_move: int
    end_move: int
    narrative: Optional[str] = None
    dominant_theme: str = ""

class GameJson(BaseModel):
    metadata: GameMetadata
    moves: List[GameMove]
    episodes: List[EpisodeSummary] = []
    game_narrative: Optional[str] = None
    game_summary: Optional[Dict[str, Any]] = None
    analysis_info: AnalysisInfo



