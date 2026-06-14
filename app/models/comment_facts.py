"""Data models for the Guid-style Expert Module.

``CommentFacts`` is the single source of positional truth for a commented
move: the verdict, the displayed (shortened) line, the eval, and the claims
fired by the rule engine. The LLM enrichment layer may rephrase and decorate
these facts but may not add to them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureDelta(BaseModel):
    """One feature's change between the starting and envisioned position."""

    name: str
    delta_cp: int
    before_cp: int
    after_cp: int
    flag_before: int | None = None
    flag_after: int | None = None


class FeatureDiff(BaseModel):
    """White-POV diff vector start -> envisioned, split per Guid Tables 5.1/5.2."""

    positive: list[FeatureDelta] = Field(default_factory=list)  # favorable for White
    negative: list[FeatureDelta] = Field(default_factory=list)  # favorable for Black


class EnvisionedLine(BaseModel):
    """A shortened, quiescence-trimmed PV with its envisioned (leaf) position."""

    start_fen: str
    line_uci: list[str] = Field(default_factory=list)
    line_san: list[str] = Field(default_factory=list)
    fens: list[str] = Field(default_factory=list)  # position after each kept ply
    leaf_fen: str = ""
    root_eval_cp: int | None = None  # engine's backed-up eval of the full line
    depth: int | None = None
    trimmed_plies: int = 0
    start_quiescent: bool = True
    leaf_quiescent: bool = True
    # Per-point progression of the comment's features along this line
    # (point 0 = start position, then one per kept ply). White-POV cp.
    # Lets the UI chart how the fired-rule features evolve through the line.
    feature_series: dict[str, list[int]] = Field(default_factory=dict)


class Claim(BaseModel):
    """One fired rule: a declarative sentence backed by named feature changes."""

    rule_id: str
    text: str
    # Envisioned-state phrasing of the same fact (Guid subproblem 5):
    # change-form "Black has improved the pawn structure." vs
    # state-form  "Black's pawn structure is now improved."
    text_state: str | None = None
    features_involved: list[str] = Field(default_factory=list)
    delta_cp: int = 0  # combined White-POV magnitude behind the claim
    flag_note: str | None = None  # e.g. "2 -> 1"
    # Which side this claim favors ("white" | "black"). Drives mover-perspective
    # ordering: mover-beneficial claims are the move's merits, the rest are
    # concessions.
    beneficiary: str | None = None
    # True when this claim favors the opponent of the mover and is kept as an
    # explicitly framed trade-off / consequence.
    is_concession: bool = False


class BestAlternative(BaseModel):
    """The engine-preferred move, with its own envisioned line and claims."""

    san: str
    uci: str
    eval_cp: int | None = None
    verdict: str = ""
    display_line: EnvisionedLine | None = None
    claims: list[Claim] = Field(default_factory=list)


class CommentFacts(BaseModel):
    """Everything a comment is allowed to assert about one move."""

    ply: int
    san: str
    uci: str
    mover: str  # "White" | "Black"
    phase: str  # "early" | "mid" | "end"
    verdict: str  # e.g. "leads to equality", "wins a decisive advantage for White"
    eval_cp: int | None = None  # White-POV cp after the move
    eval_before_cp: int | None = (
        None  # White-POV cp before the move (transition verdicts)
    )
    eval_mate: int | None = None
    # For dubious moves: the opponent's punishing reply (SAN), straight from
    # the engine continuation — the "why it is bad" at board level.
    refutation_san: str | None = None
    depth: int | None = None
    engine: str = "Stockfish"
    display_line: EnvisionedLine | None = None
    claims: list[Claim] = Field(default_factory=list)
    feature_diff: FeatureDiff | None = None
    better_alternative: BestAlternative | None = None
    # Claims that fired but were muted by the adjacent-move dedup window
    # (kept for the academic debug/reasoning view).
    muted_claims: list[str] = Field(default_factory=list)
    # How concessions should be framed: "tradeoff" for sound moves
    # ("In return, ..."), "consequence" for inaccuracies/mistakes/blunders
    # (the concessions ARE the explanation of the eval swing).
    concession_mode: str = "tradeoff"

    def feature_refs(self) -> list[str]:
        seen: list[str] = []
        for c in self.claims:
            for f in c.features_involved:
                if f not in seen:
                    seen.append(f)
        return seen
