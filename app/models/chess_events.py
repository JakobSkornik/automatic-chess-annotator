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
    DECOY = "decoy"
    INTERFERENCE = "interference"
    ZWISCHENZUG = "zwischenzug"
    QUIET_MOVE_THREAT = "quiet_move_threat"
    X_RAY = "x_ray"
    DOUBLE_ATTACK = "double_attack"
    WINDMILL = "windmill"
    CLEARANCE = "clearance"
    ATTRACTION = "attraction"
    DESPERADO = "desperado"
    POSITIONAL_PAWN_SAC = "positional_pawn_sac"
    TRADE_TO_DEFUSE_ATTACK = "trade_to_defuse_attack"


class StrategicMotif(str, Enum):
    """Named strategic / endgame patterns (heuristic tags for LLM)."""

    MINORITY_ATTACK = "minority_attack"
    BACKWARD_PAWN_TARGET = "backward_pawn_target"
    OUTPOST_OCCUPIED = "outpost_occupied"
    OUTPOST_AVAILABLE = "outpost_available"
    OUTPOST = "outpost"
    WEAK_SQUARE_CREATED = "weak_square_created"
    WEAK_SQUARE_EXPLOITED = "weak_square_exploited"
    WEAK_SQUARE_CREATION = "weak_square_creation"
    COLOR_COMPLEX_WEAKNESS = "color_complex_weakness"
    ISOLATED_QUEEN_PAWN = "isolated_queen_pawn"
    HANGING_PAWNS = "hanging_pawns"
    OPEN_FILE_OCCUPATION = "open_file_occupation"
    BAD_BISHOP = "bad_bishop"
    GOOD_BISHOP = "good_bishop"
    BISHOP_PAIR_ADVANTAGE = "bishop_pair_advantage"
    OPPOSITE_SIDE_CASTLING_RACE = "opposite_side_castling_race"
    CENTRAL_COUNTER_VS_WING_ATTACK = "central_counter_vs_wing_attack"
    WRONG_WING_PIECE_IN_RACE = "wrong_wing_piece_in_race"
    PROPHYLAXIS = "prophylaxis"
    LUFT = "luft"
    DOMINATION = "domination"
    PIECE_REROUTING = "piece_rerouting"
    PAWN_LEVER = "pawn_lever"
    SPACE_ADVANTAGE = "space_advantage"
    PAWN_MAJORITY_ATTACK = "pawn_majority_attack"
    PASSED_PAWN_MIDDLEGAME = "passed_pawn_middlegame"
    CONNECTED_PASSERS = "connected_passers"
    BLOCKADE = "blockade"
    OVERPROTECTION = "overprotection"
    RESTRICTION = "restriction"
    ROOK_ON_SEVENTH = "rook_on_seventh"
    ROOK_LIFT = "rook_lift"
    SIMPLIFICATION_WHEN_AHEAD = "simplification_when_ahead"
    OPPOSITION = "opposition"
    TRIANGULATION = "triangulation"
    OUTSIDE_PASSER = "outside_passer"
    LUCENA = "lucena"
    PHILIDOR = "philidor"
    FORTRESS = "fortress"
    ZUGZWANG = "zugzwang"


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


class MoveCategory(str, Enum):
    """Coach-style move bucket for prompt selection."""

    BOOK = "book"
    FORCING = "forcing"
    PROPHYLACTIC = "prophylactic"
    POSITIONAL = "positional"
    TACTICAL = "tactical"
    DEFENSIVE = "defensive"
    INACCURACY = "inaccuracy"
    CRITICAL = "critical"


class PlanComparison(BaseModel):
    """Coarse plan hints from PV destination clustering."""

    played_target_squares: List[str] = Field(default_factory=list)
    best_target_squares: List[str] = Field(default_factory=list)
    played_plan_seed: str = ""
    best_plan_seed: str = ""
    played_recurring_destinations: List[str] = Field(default_factory=list)
    best_recurring_destinations: List[str] = Field(default_factory=list)


class MoveRationale(BaseModel):
    """Symbolic intermediate for LLM: why the move matters (no raw feature dump)."""

    eval_change_cp: int = 0
    immediate_effect: str = ""
    future_effect: str = ""
    motif: Optional[str] = None
    counterfactual: Optional[str] = None
    played_plan: Optional[str] = None
    best_plan: Optional[str] = None
    risk: Optional[str] = None
    stakes: Optional[str] = None
    glossary_phrasings: Dict[str, str] = Field(default_factory=dict)
    primary_motif_label: str = ""
    narrative_template: str = ""
    coach_scratchpad: Dict[str, str] = Field(default_factory=dict)


class FutureLineDelta(BaseModel):
    """Played PV vs best PV after N plies (engine); populated for critical moves at LLM time."""

    played_leaf_eval_cp: Optional[int] = None
    best_leaf_eval_cp: Optional[int] = None
    eval_gap_cp: Optional[int] = None  # best_leaf - played_leaf (White POV)
    feature_deltas: Dict[str, float] = Field(default_factory=dict)
    played_targets: List[str] = Field(default_factory=list)
    best_targets: List[str] = Field(default_factory=list)
    played_line_san: List[str] = Field(default_factory=list)
    best_line_san: List[str] = Field(default_factory=list)


class PvHorizonDiff(BaseModel):
    """Root vs leaf hidden features along engine PV1 (current position horizon)."""

    plies: int
    pv_san: List[str] = Field(default_factory=list)
    leaf_eval_cp: Optional[int] = None
    scalar_deltas: Dict[str, float] = Field(default_factory=dict)
    list_deltas: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)


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
    strategic_motifs: List[StrategicMotif] = Field(default_factory=list)
    plan_comparison: Optional[PlanComparison] = None
    move_category: Optional[MoveCategory] = None
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
    brief_commentary: bool = False
    teaching_moment: bool = False
    # When back-to-back key moments are suppressed, points to prior ply for stub text
    commentary_stub_ref_ply: Optional[int] = None
    # PV from fen_after (engine), first plies as SAN — used by Tantivy BM25 RAG
    pv_san: Optional[List[str]] = None
    # Future-line comparison (played vs best continuation); set in GameAnnotationPipeline for critical moves
    future_line: Optional[FutureLineDelta] = None
    # Root vs PV-leaf feature deltas along engine PV1 (engine pass, out-of-book positions)
    pv_horizon_diff: Optional[PvHorizonDiff] = None
    # max(eval@depth) - min(eval@depth) across depths 8/12/16 on after-move position
    eval_instability_cp: Optional[int] = None


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
    # Whole-game LLM digest (JSON); injected into per-move prompts before commentary runs
    game_digest: Dict[str, Any] = Field(default_factory=dict)
    # Last few one-line hints from prior LLM comments (motifs + archetype) for continuity
    prior_context_snippets: List[str] = Field(default_factory=list)


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
    # Search instability: eval at multiple depths on the after-move position (multipv=1)
    eval_at_depth: Dict[int, int] = Field(default_factory=dict)
    pv1_change_count: int = 0
    # Cached from ChessEventExtractor / KeyMomentDetector (single source of truth)
    key_moment_type: Optional[str] = None
    pv_horizon_diff: Optional[PvHorizonDiff] = None
