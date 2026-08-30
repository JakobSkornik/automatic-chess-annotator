"""Rule registry, claim ordering, and the ``run_rules`` driver."""

from __future__ import annotations

import logging

import chess

from app.models.comment_facts import Claim, FeatureDiff

from .constants import MAX_CLAIMS_PER_MOVE, MAX_CONCESSIONS, THRESHOLDS
from .context import Rule, _Ctx
from .pawn_rules import (
    rule_doubled_pawns,
    rule_material,
    rule_outside_passer,
    rule_passed_pawn,
    rule_passer_escort,
    rule_pawn_structure,
    rule_rook_behind_passer,
)
from .piece_rules import (
    rule_back_rank,
    rule_bad_bishop,
    rule_bishop_pair,
    rule_center_and_space,
    rule_connected_rooks,
    rule_intent,
    rule_king_activity,
    rule_king_safety,
    rule_piece_activity,
    rule_rook_activity,
    rule_strong_knight,
)
from .threat_rules import rule_pin, rule_threats

logger = logging.getLogger(__name__)


def order_claims_for_mover(claims: list[Claim], mover: str) -> list[Claim]:
    """Mover-perspective ordering: the mover's merits first; claims favoring
    the opponent become explicitly tagged concessions, capped at
    ``MAX_CONCESSIONS`` (strongest kept)."""
    mover_key = mover.lower()
    merits = [c for c in claims if c.beneficiary in (mover_key, None)]
    concessions = [c for c in claims if c.beneficiary not in (mover_key, None)]
    if len(concessions) > MAX_CONCESSIONS:
        logger.debug(
            "order_claims_for_mover: MAX_CONCESSIONS cap dropped %d concession(s) "
            "(mover=%s, cap=%d): %s",
            len(concessions) - MAX_CONCESSIONS,
            mover,
            MAX_CONCESSIONS,
            [c.rule_id for c in concessions[MAX_CONCESSIONS:]],
        )
    concessions = [
        c.model_copy(update={"is_concession": True})
        for c in concessions[:MAX_CONCESSIONS]
    ]
    return merits + concessions


ALL_RULES: list[Rule] = [
    rule_material,
    rule_threats,
    rule_pin,
    rule_intent,
    rule_pawn_structure,
    rule_doubled_pawns,
    rule_bishop_pair,
    rule_strong_knight,
    rule_bad_bishop,
    rule_rook_activity,
    rule_rook_behind_passer,
    rule_passed_pawn,
    rule_king_safety,
    rule_back_rank,
    rule_piece_activity,
    rule_center_and_space,
    rule_connected_rooks,
    rule_king_activity,
    rule_outside_passer,
    rule_passer_escort,
]


def run_rules(
    diff: FeatureDiff,
    *,
    phase: str,
    mover: str,
    eval_cp: int | None = None,
    start_board: chess.Board | None = None,
    leaf_board: chess.Board | None = None,
) -> list[Claim]:
    """Fire all rules; keep the strongest few claims.

    A concrete material claim (``material_won`` / ``material_standing`` /
    ``tactical_material_loss``) is exempt from the per-move cap — winning or
    losing material is the single most important fact about a move and must
    never be crowded out by positional claims (it always leads)."""
    ctx = _Ctx(
        diff, phase, mover, eval_cp, start_board=start_board, leaf_board=leaf_board
    )
    claims: list[Claim] = []
    for rule in ALL_RULES:
        try:
            claims.extend(rule(ctx))
        except Exception as e:
            logger.warning("rule %s failed: %s", getattr(rule, "__name__", rule), e)
    # Every claim must clear the magnitude floor — a flag flip alone (e.g. a
    # routine bishop trade) is no longer enough to earn a sentence.
    below_floor = [c for c in claims if c.delta_cp < THRESHOLDS["min_claim_cp"]]
    if below_floor:
        logger.debug(
            "run_rules: min_claim_cp gate dropped %d claim(s) (mover=%s, threshold=%d): %s",
            len(below_floor),
            mover,
            THRESHOLDS["min_claim_cp"],
            [(c.rule_id, c.delta_cp) for c in below_floor],
        )
    claims = [c for c in claims if c.delta_cp >= THRESHOLDS["min_claim_cp"]]
    claims.sort(key=lambda c: -c.delta_cp)
    material_ids = {"material_won", "material_standing", "tactical_material_loss"}
    material = [c for c in claims if c.rule_id in material_ids]
    rest = [c for c in claims if c.rule_id not in material_ids]
    if len(rest) > MAX_CLAIMS_PER_MOVE:
        logger.debug(
            "run_rules: MAX_CLAIMS_PER_MOVE cap dropped %d claim(s) (mover=%s, cap=%d): %s",
            len(rest) - MAX_CLAIMS_PER_MOVE,
            mover,
            MAX_CLAIMS_PER_MOVE,
            [c.rule_id for c in rest[MAX_CLAIMS_PER_MOVE:]],
        )
    return material + rest[:MAX_CLAIMS_PER_MOVE]


# ---------------------------------------------------------------------------
# Verdict (qualitative assessment of the eval)
# ---------------------------------------------------------------------------
