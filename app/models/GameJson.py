from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class RagRef(BaseModel):
    """Serialized RAG hit persisted on a move (mirrors WebSocket rag_refs payload)."""

    source: str = ""
    fen: str = ""
    text: str = ""
    score: float = 0.0
    san: str = ""
    phase: str = ""
    opening_eco: str = ""
    opening_name: str = ""
    material_signature: str = ""


class MoveScore(BaseModel):
    cp: Optional[int] = None
    mate: Optional[int] = None

class Variation(BaseModel):
    rank: int
    move_san: str
    score: Optional[MoveScore] = None
    line: List[str]
    # Position after each ply of `line` — lets the PV popup slide/autoplay
    # without replaying SAN on the frontend.
    fens: List[str] = []


class FeatureRef(BaseModel):
    """A positional feature this move's comment is grounded in (chart highlight)."""

    name: str
    delta_cp: int = 0


class GameMove(BaseModel):
    mn: int
    color: str
    san: str
    uci: str
    fen: str
    phase: str = "mid"  # "early" | "mid" | "end"
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
    named_motifs: List[str] = []
    primary_motif_label: Optional[str] = None
    rag_refs: List[RagRef] = []
    opponent_threats: List[str] = []
    pv_motif_summary: List[str] = []
    motif_trajectory: Optional[str] = None
    # Guid Expert Module outputs
    feature_refs: List[FeatureRef] = []
    feature_diff: Optional[Dict[str, Any]] = None  # {"positive": [...], "negative": [...]}
    resolved_tokens: List[Dict[str, Any]] = []
    # Per-audience-level renderings of the same facts; `comment` mirrors the
    # intermediate level for backward compatibility.
    comments: Dict[str, str] = {}  # {"expert": ..., "intermediate": ..., "beginner": ...}
    resolved_tokens_by_level: Dict[str, List[Dict[str, Any]]] = {}
    # Trimmed CommentFacts for the structured comment renderer (assessment /
    # reasons / better alternative, each with its own line).
    comment_facts: Optional[Dict[str, Any]] = None

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
    opening_eco: Optional[str] = None
    strategic_archetype: Optional[str] = None

class AnalysisInfo(BaseModel):
    engine: str
    depth: int
    multipv: int
    timestamp: float


class FeatureSeries(BaseModel):
    """Per-ply progression of every charted positional feature (White-POV cp)."""

    plies: List[int] = []
    features: Dict[str, List[Optional[int]]] = {}

class EpisodeSummary(BaseModel):
    episode_index: int
    title: str
    start_move: int
    end_move: int
    narrative: Optional[str] = None
    dominant_theme: str = ""
    motif_trajectory: Optional[str] = None

class GameJson(BaseModel):
    metadata: GameMetadata
    moves: List[GameMove]
    episodes: List[EpisodeSummary] = []
    game_narrative: Optional[str] = None
    feature_series: Optional[FeatureSeries] = None
    analysis_info: AnalysisInfo



