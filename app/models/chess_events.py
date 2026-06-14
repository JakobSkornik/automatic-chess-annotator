"""Semantic chess-event models for the annotation pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.comment_facts import CommentFacts


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

    played_target_squares: list[str] = Field(default_factory=list)
    best_target_squares: list[str] = Field(default_factory=list)
    played_plan_seed: str = ""
    best_plan_seed: str = ""
    played_recurring_destinations: list[str] = Field(default_factory=list)
    best_recurring_destinations: list[str] = Field(default_factory=list)
    played_plan_tags: list[str] = Field(default_factory=list)
    best_plan_tags: list[str] = Field(default_factory=list)


class PlyMotifScan(BaseModel):
    """Motifs detected at one ply along an engine PV line."""

    ply: int
    san: str = ""
    uci: str = ""
    mover: str = ""
    tactical_motifs: list[TacticalMotif] = Field(default_factory=list)
    strategic_motifs: list[StrategicMotif] = Field(default_factory=list)


class MoveRationale(BaseModel):
    """Symbolic intermediate for LLM: why the move matters (no raw feature dump)."""

    eval_change_cp: int = 0
    immediate_effect: str = ""
    future_effect: str = ""
    motif: str | None = None
    counterfactual: str | None = None
    played_plan: str | None = None
    best_plan: str | None = None
    risk: str | None = None
    stakes: str | None = None
    glossary_phrasings: dict[str, str] = Field(default_factory=dict)
    primary_motif_label: str = ""
    narrative_template: str = ""
    coach_scratchpad: dict[str, str] = Field(default_factory=dict)


class FutureLineDelta(BaseModel):
    """Played PV vs best PV after N plies (engine); populated for critical moves at LLM time."""

    played_leaf_eval_cp: int | None = None
    best_leaf_eval_cp: int | None = None
    eval_gap_cp: int | None = None  # best_leaf - played_leaf (White POV)
    feature_deltas: dict[str, float] = Field(default_factory=dict)
    played_targets: list[str] = Field(default_factory=list)
    best_targets: list[str] = Field(default_factory=list)
    played_line_san: list[str] = Field(default_factory=list)
    best_line_san: list[str] = Field(default_factory=list)


class PvHorizonDiff(BaseModel):
    """Root vs leaf hidden features along engine PV1 (current position horizon)."""

    plies: int
    pv_san: list[str] = Field(default_factory=list)
    leaf_eval_cp: int | None = None
    scalar_deltas: dict[str, float] = Field(default_factory=dict)
    list_deltas: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class MoveEvent(BaseModel):
    """Rich per-move event descriptor produced by the feature extraction layer."""

    move_index: int
    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    phase: str
    eval_before_cp: int | None = None
    eval_after_cp: int | None = None
    eval_swing_cp: int | None = None
    move_quality: MoveQuality
    event_type: MoveEventType
    tactical_motifs: list[TacticalMotif] = Field(default_factory=list)
    strategic_motifs: list[StrategicMotif] = Field(default_factory=list)
    plan_comparison: PlanComparison | None = None
    move_category: MoveCategory | None = None
    is_critical: bool = False
    best_move_san: str | None = None
    best_move_uci: str | None = None
    best_move_eval_cp: int | None = None
    pv_lines: list[dict[str, Any]] = Field(default_factory=list)
    material_balance: dict[str, Any] | None = None
    king_safety: dict[str, Any] | None = None
    pawn_structure_type: str | None = None
    score_trend: list[int] = Field(default_factory=list)
    opening_name: str | None = None
    opening_eco: str | None = None
    key_moment_type: str | None = None
    brief_commentary: bool = False
    teaching_moment: bool = False
    # When back-to-back key moments are suppressed, points to prior ply for stub text
    commentary_stub_ref_ply: int | None = None
    # PV from fen_after (engine), first plies as SAN — used by Tantivy BM25 RAG
    pv_san: list[str] | None = None
    # Future-line comparison (played vs best continuation); set in GameAnnotationPipeline for critical moves
    future_line: FutureLineDelta | None = None
    # Root vs PV-leaf feature deltas along engine PV1 (engine pass, out-of-book positions)
    pv_horizon_diff: PvHorizonDiff | None = None
    # max(eval@depth) - min(eval@depth) across depths 8/12/16 on after-move position
    eval_instability_cp: int | None = None
    # Motifs detected along engine PV1 (future plies)
    pv_motifs: list[PlyMotifScan] = Field(default_factory=list)
    # Tactical threats available to the opponent from fen_after
    opponent_threats: list[TacticalMotif] = Field(default_factory=list)
    # Sustained motif pattern label when a motif persists >=3 consecutive plies
    motif_trajectory: str | None = None
    # Guid Expert Module output: the move's inviolable comment facts
    comment_facts: CommentFacts | None = None


class Episode(BaseModel):
    """A contiguous group of moves forming a strategic/narrative unit."""

    episode_index: int
    title: str
    start_ply: int
    end_ply: int
    move_events: list[MoveEvent] = Field(default_factory=list)
    eval_start_cp: int | None = None
    eval_end_cp: int | None = None
    eval_trend: list[int] = Field(default_factory=list)
    dominant_theme: str = ""
    tactical_motifs_in_episode: list[TacticalMotif] = Field(default_factory=list)
    phase: str = ""
    narrative_summary: str | None = None
    motif_trajectory: str | None = None


class GameAnalysisContext(BaseModel):
    """Complete structured analysis context for LLM consumption."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    move_events: list[MoveEvent] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    critical_moments: list[MoveEvent] = Field(default_factory=list)
    opening_name: str | None = None
    opening_eco: str | None = None
    game_narrative: str | None = None
    # Whole-game LLM digest (JSON); injected into per-move prompts before commentary runs
    game_digest: dict[str, Any] = Field(default_factory=dict)
    # Last few one-line hints from prior LLM comments (motifs + archetype) for continuity
    prior_context_snippets: list[str] = Field(default_factory=list)


class AnalyzedMoveData(BaseModel):
    """Intermediate result from the engine pass, before semantic extraction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    score_cp: int | None = None
    phase_raw: str = ""
    pvs: list[list[Any]] = Field(default_factory=list)
    hidden_features: dict[str, Any] = Field(default_factory=dict)
    captured_by_white: dict[str, int] = Field(default_factory=dict)
    captured_by_black: dict[str, int] = Field(default_factory=dict)
    analyzed_move: Any = None
    # Search instability: eval at multiple depths on the after-move position (multipv=1)
    eval_at_depth: dict[int, int] = Field(default_factory=dict)
    pv1_change_count: int = 0
    # Cached from ChessEventExtractor / KeyMomentDetector (single source of truth)
    key_moment_type: str | None = None
    pv_horizon_diff: PvHorizonDiff | None = None
