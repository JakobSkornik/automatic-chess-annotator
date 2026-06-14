from typing import Any

from pydantic import BaseModel


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
    cp: int | None = None
    mate: int | None = None


class Variation(BaseModel):
    rank: int
    move_san: str
    score: MoveScore | None = None
    line: list[str]
    # Position after each ply of `line` — lets the PV popup slide/autoplay
    # without replaying SAN on the frontend.
    fens: list[str] = []
    # Search depth behind `score` (shown after the eval at the end of the line).
    depth: int | None = None
    # Key factors of the final position vs the base position: claims fired on
    # the root->leaf feature diff. [{text, text_state, features, delta_cp, flag_note}]
    key_factors: list[dict[str, Any]] = []


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
    score: MoveScore | None = None
    variations: list[Variation] = []
    comment: str | None = None
    classification: str | None = None
    move_quality: str | None = None
    event_type: str | None = None
    tactical_motifs: list[str] = []
    strategic_motifs: list[str] = []
    move_category: str | None = None
    plan_comparison: dict[str, Any] | None = None
    is_critical: bool = False
    # True only for moves that received a real key-moment/teaching commentary
    # pass (drives the move-list dot) — not the template-floor facts that every
    # analyzed move carries.
    is_key_moment: bool = False
    episode_index: int | None = None
    named_motifs: list[str] = []
    primary_motif_label: str | None = None
    rag_refs: list[RagRef] = []
    opponent_threats: list[str] = []
    pv_motif_summary: list[str] = []
    motif_trajectory: str | None = None
    # Guid Expert Module outputs
    feature_refs: list[FeatureRef] = []
    feature_diff: dict[str, Any] | None = None  # {"positive": [...], "negative": [...]}
    resolved_tokens: list[dict[str, Any]] = []
    # Per-audience-level renderings of the same facts; `comment` mirrors the
    # intermediate level for backward compatibility.
    comments: dict[
        str, str
    ] = {}  # {"expert": ..., "intermediate": ..., "beginner": ...}
    resolved_tokens_by_level: dict[str, list[dict[str, Any]]] = {}
    # Trimmed CommentFacts for the structured comment renderer (assessment /
    # reasons / better alternative, each with its own line).
    comment_facts: dict[str, Any] | None = None
    # Academic reasoning trace: how this move's conclusions were reached
    # (engine numbers, key moment, envisioned-line stats, fired/muted rules,
    # per-level rendering outcomes).
    debug: dict[str, Any] | None = None


class GameMetadata(BaseModel):
    id: str
    white: str
    black: str
    result: str
    date: str | None = None
    eventId: str | None = None
    whiteElo: int | None = None
    blackElo: int | None = None
    opening: str | None = None
    opening_eco: str | None = None
    strategic_archetype: str | None = None
    # Pre-analysis options the commentary was generated with
    commentary_level: str | None = None  # beginner | intermediate | expert
    comment_side: str | None = None  # white | black | both


class AnalysisInfo(BaseModel):
    engine: str
    depth: int
    multipv: int
    timestamp: float


class FeatureSeries(BaseModel):
    """Per-ply progression of every charted positional feature (White-POV cp)."""

    plies: list[int] = []
    features: dict[str, list[int | None]] = {}


class EpisodeSummary(BaseModel):
    episode_index: int
    title: str
    start_move: int
    end_move: int
    narrative: str | None = None
    dominant_theme: str = ""
    motif_trajectory: str | None = None


class GameJson(BaseModel):
    metadata: GameMetadata
    moves: list[GameMove]
    episodes: list[EpisodeSummary] = []
    game_narrative: str | None = None
    feature_series: FeatureSeries | None = None
    # Pipeline parameters behind the per-move debug traces (rule thresholds etc.)
    debug_info: dict[str, Any] | None = None
    # True once the LLM commentary sweep finished (drives the FE "done" state).
    commentary_complete: bool = False
    analysis_info: AnalysisInfo
