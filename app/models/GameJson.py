from typing import Any

from pydantic import BaseModel


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
    # NAG-style symbol derived from the key-moment classification
    # (!! brilliant, ! best/great, ?! inaccuracy, ? mistake, ?? blunder).
    annotation: str | None = None
    # True only for moves that received a real key-moment commentary pass
    # (drives the move-list dot) — not the template-floor facts that every
    # analyzed move carries.
    is_key_moment: bool = False
    resolved_tokens: list[dict[str, Any]] = []
    # Trimmed CommentFacts for the structured comment renderer (assessment /
    # reasons / better alternative, each with its own line).
    comment_facts: dict[str, Any] | None = None
    # Academic reasoning trace: how this move's conclusions were reached
    # (engine numbers, key moment, envisioned-line stats, fired/muted rules).
    debug: dict[str, Any] | None = None


class GameMetadata(BaseModel):
    id: str
    white: str
    black: str
    result: str
    eventId: str | None = None
    whiteElo: int | None = None
    blackElo: int | None = None
    opening: str | None = None
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


class GameJson(BaseModel):
    metadata: GameMetadata
    moves: list[GameMove]
    feature_series: FeatureSeries | None = None
    # True once the LLM commentary sweep finished (drives the FE "done" state).
    commentary_complete: bool = False
    analysis_info: AnalysisInfo
