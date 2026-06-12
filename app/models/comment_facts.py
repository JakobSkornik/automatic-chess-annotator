"""Data models for the Guid-style Expert Module.

``CommentFacts`` is the single source of positional truth for a commented
move: the verdict, the displayed (shortened) line, the eval, and the claims
fired by the rule engine. The LLM enrichment layer may rephrase and decorate
these facts but may not add to them.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FeatureDelta(BaseModel):
    """One feature's change between the starting and envisioned position."""

    name: str
    delta_cp: int
    before_cp: int
    after_cp: int
    flag_before: Optional[int] = None
    flag_after: Optional[int] = None


class FeatureDiff(BaseModel):
    """White-POV diff vector start -> envisioned, split per Guid Tables 5.1/5.2."""

    positive: List[FeatureDelta] = Field(default_factory=list)  # favorable for White
    negative: List[FeatureDelta] = Field(default_factory=list)  # favorable for Black


class EnvisionedLine(BaseModel):
    """A shortened, quiescence-trimmed PV with its envisioned (leaf) position."""

    start_fen: str
    line_uci: List[str] = Field(default_factory=list)
    line_san: List[str] = Field(default_factory=list)
    fens: List[str] = Field(default_factory=list)  # position after each kept ply
    leaf_fen: str = ""
    root_eval_cp: Optional[int] = None  # engine's backed-up eval of the full line
    depth: Optional[int] = None
    trimmed_plies: int = 0
    start_quiescent: bool = True
    leaf_quiescent: bool = True


class Claim(BaseModel):
    """One fired rule: a declarative sentence backed by named feature changes."""

    rule_id: str
    text: str
    features_involved: List[str] = Field(default_factory=list)
    delta_cp: int = 0  # combined White-POV magnitude behind the claim
    flag_note: Optional[str] = None  # e.g. "2 -> 1"


class BestAlternative(BaseModel):
    """The engine-preferred move, with its own envisioned line and claims."""

    san: str
    uci: str
    eval_cp: Optional[int] = None
    verdict: str = ""
    display_line: Optional[EnvisionedLine] = None
    claims: List[Claim] = Field(default_factory=list)


class CommentFacts(BaseModel):
    """Everything a comment is allowed to assert about one move."""

    ply: int
    san: str
    uci: str
    mover: str  # "White" | "Black"
    phase: str  # "early" | "mid" | "end"
    verdict: str  # e.g. "leads to equality", "wins a decisive advantage for White"
    eval_cp: Optional[int] = None  # White-POV cp after the move
    eval_mate: Optional[int] = None
    depth: Optional[int] = None
    engine: str = "Stockfish"
    display_line: Optional[EnvisionedLine] = None
    claims: List[Claim] = Field(default_factory=list)
    feature_diff: Optional[FeatureDiff] = None
    better_alternative: Optional[BestAlternative] = None

    def feature_refs(self) -> List[str]:
        seen: List[str] = []
        for c in self.claims:
            for f in c.features_involved:
                if f not in seen:
                    seen.append(f)
        return seen
