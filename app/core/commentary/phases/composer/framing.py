"""Move framing: the commentary archetype (engine choice / brilliancy /
inaccuracy) and the better-alternative gap helpers that decide whether the
engine's move is "better" or merely "comparable"."""

from __future__ import annotations

from app.models.comment_facts import Claim, CommentFacts

# Below this mover-POV gap the engine's move is a comparable option, not a miss.
ALT_MATERIAL_GAP_CP = 50


def alt_gap_cp(facts: CommentFacts) -> int | None:
    """Mover-POV centipawns the alternative gains over the played move."""
    alt = facts.better_alternative
    if (alt is None or alt.eval_mate is not None or facts.eval_mate is not None
            or alt.eval_cp is None or facts.eval_cp is None):
        return None
    gap = alt.eval_cp - facts.eval_cp  # White-POV
    return gap if facts.mover == "White" else -gap


def alternative_comparison(facts: CommentFacts, *, threshold_cp: int = ALT_MATERIAL_GAP_CP) -> str:
    """Compare known engine scores in mover POV, keeping mate and cp distinct."""
    alt = facts.better_alternative
    if alt is None:
        return "unknown"
    direction = 1 if facts.mover == "White" else -1
    played_mate = facts.eval_mate
    alt_mate = alt.eval_mate
    if played_mate == 0 or alt_mate == 0:
        return "unknown"
    if played_mate is not None or alt_mate is not None:
        # A finite evaluation reports no forced mate at this search horizon;
        # it does not establish that the position is free of a mating line.
        if played_mate is None and facts.eval_cp is None:
            return "unknown"
        if alt_mate is None and alt.eval_cp is None:
            return "unknown"
        def rank(mate):
            return 0 if mate is None else (1 if mate * direction > 0 else -1)
        played_rank, alt_rank = rank(played_mate), rank(alt_mate)
        if alt_rank != played_rank:
            return "better" if alt_rank > played_rank else "weaker"
        if played_mate == alt_mate:
            return "comparable"
        # For a forced win prefer a shorter mate; for a forced loss delay it.
        shorter = abs(alt_mate) < abs(played_mate)
        better = shorter if alt_rank > 0 else not shorter
        return "better" if better else "weaker"
    gap = alt_gap_cp(facts)
    if gap is None:
        return "unknown"
    if gap >= threshold_cp:
        return "better"
    if gap <= -threshold_cp:
        return "weaker"
    return "comparable"


def _is_alt_materially_better(facts: CommentFacts) -> bool:
    return alternative_comparison(facts) == "better"


def _missed_noun(claims: list[Claim]) -> str:
    """'chance' if the alternative's gain lands at once, 'potential' if it only
    develops in the line (judged on the ordered-strongest claim)."""
    return "potential" if claims and claims[0].realization == "envisioned" else "chance"


def comment_archetype(key_moment_type: str | None) -> str:
    """Map the key-moment classification to a commentary framing.

    engine_choice      the played move IS the engine's pick — confirm, don't fault
    brilliant_sacrifice a material sacrifice that holds/improves the eval
    inaccuracy_missed  an imprecise move — lead with the opportunity passed up
    neutral            the default verdict + claims shape
    """
    kmt = key_moment_type or ""
    if kmt == "brilliant":
        return "brilliant_sacrifice"
    if kmt == "kingside_attack":
        return "kingside_attack"
    if kmt in ("best_move", "great_move"):
        return "engine_choice"
    if kmt in ("inaccuracy", "missed_opportunity"):
        return "inaccuracy_missed"
    return "neutral"


# NB: openers must not contain any phrase in forbidden_phrases.FORBIDDEN_REGEX,
# or the scrub pass strips it mid-sentence and leaves debris ("is the ,").
_ARCHETYPE_OPENER: dict[str, str] = {
    "brilliant_sacrifice": "is a brilliant sacrifice,",
    "kingside_attack": "turns the pieces toward the enemy king,",
}
