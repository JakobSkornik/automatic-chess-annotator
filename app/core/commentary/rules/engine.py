"""Rule-based Expert Module (Guid §5.4) — public facade.

Rules are threshold conditions over the feature-difference vector between the
starting and envisioned positions; each fired rule yields a ``Claim``. The
implementation is split across cohesive modules:

  constants     tunable thresholds and limits
  context       the rule context (``_Ctx``) and side helpers
  material      concrete material-imbalance phrasing
  pawn_rules    pawn / passer / material rules
  piece_rules   piece, rook, king and space rules
  catalog       rule registry + ``run_rules`` + claim ordering
  verdicts      eval -> verdict phrasing
  realization   per-ply feature series + immediate/envisioned tagging
  facts_builder ``build_comment_facts`` (the module entry point)
"""

from __future__ import annotations

from .catalog import order_claims_for_mover, run_rules
from .constants import MAX_CONCESSIONS, THRESHOLDS
from .facts_builder import build_comment_facts
from .realization import _line_feature_series
from .verdicts import verdict_for_eval, verdict_for_transition

__all__ = [
    "MAX_CONCESSIONS",
    "THRESHOLDS",
    "_line_feature_series",
    "build_comment_facts",
    "order_claims_for_mover",
    "run_rules",
    "verdict_for_eval",
    "verdict_for_transition",
]
