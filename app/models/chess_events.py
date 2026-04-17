"""Semantic chess-event models for the annotation pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TacticalMotif(str, Enum):
    FORK = "fork"
    PIN = "pin"
    SKEWER = "skewer"
    DISCOVERED_ATTACK = "discovered_attack"
    DISCOVERED_CHECK = "discovered_check"
    DOUBLE_CHECK = "double_check"
    SACRIFICE = "sacrifice"
    BACK_RANK_THREAT = "back_rank_threat"
    MATING_NET = "mating_net"
    REMOVAL_OF_GUARD = "removal_of_guard"
    DEFLECTION = "deflection"
    OVERLOADED_PIECE = "overloaded_piece"


class MoveEventType(str, Enum):
    EVAL_SWING = "eval_swing"
    MISSED_TACTIC = "missed_tactic"
    POSITIONAL_CONCESSION = "positional_concession"
    BEST_MOVE_PLAYED = "best_move_played"
    CRITICAL_DECISION = "critical_decision"
    PHASE_TRANSITION = "phase_transition"
    STRUCTURAL_CHANGE = "structural_change"
    QUIET = "quiet"


class MoveQuality(str, Enum):
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class MoveEvent(BaseModel):
    """Rich per-move event descriptor produced by the feature extraction layer."""

    move_index: int
    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    phase: str
    eval_before_cp: Optional[int] = None
    eval_after_cp: Optional[int] = None
    eval_swing_cp: Optional[int] = None
    move_quality: MoveQuality
    event_type: MoveEventType
    tactical_motifs: List[TacticalMotif] = Field(default_factory=list)
    is_critical: bool = False
    best_move_san: Optional[str] = None
    best_move_uci: Optional[str] = None
    best_move_eval_cp: Optional[int] = None
    pv_lines: List[Dict[str, Any]] = Field(default_factory=list)
    material_balance: Optional[Dict[str, Any]] = None
    king_safety: Optional[Dict[str, Any]] = None
    pawn_structure_type: Optional[str] = None
    score_trend: List[int] = Field(default_factory=list)
    opening_name: Optional[str] = None
    opening_eco: Optional[str] = None
    key_moment_type: Optional[str] = None
    # PV from fen_after (engine), first plies as SAN — used by Tantivy BM25 RAG
    pv_san: Optional[List[str]] = None


class Episode(BaseModel):
    """A contiguous group of moves forming a strategic/narrative unit."""

    episode_index: int
    title: str
    start_ply: int
    end_ply: int
    move_events: List[MoveEvent] = Field(default_factory=list)
    eval_start_cp: Optional[int] = None
    eval_end_cp: Optional[int] = None
    eval_trend: List[int] = Field(default_factory=list)
    dominant_theme: str = ""
    tactical_motifs_in_episode: List[TacticalMotif] = Field(default_factory=list)
    phase: str = ""
    narrative_summary: Optional[str] = None


class GameAnalysisContext(BaseModel):
    """Complete structured analysis context for LLM consumption."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    move_events: List[MoveEvent] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    critical_moments: List[MoveEvent] = Field(default_factory=list)
    opening_name: Optional[str] = None
    opening_eco: Optional[str] = None
    game_narrative: Optional[str] = None


class AnalyzedMoveData(BaseModel):
    """Intermediate result from the engine pass, before semantic extraction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    score_cp: Optional[int] = None
    phase_raw: str = ""
    pvs: List[List[Any]] = Field(default_factory=list)
    hidden_features: Dict[str, Any] = Field(default_factory=dict)
    trace: Optional[Dict[str, Any]] = None
    captured_by_white: Dict[str, int] = Field(default_factory=dict)
    captured_by_black: Dict[str, int] = Field(default_factory=dict)
    analyzed_move: Any = None
