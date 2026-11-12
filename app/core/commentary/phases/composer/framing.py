"""Move framing: the commentary archetype (engine choice / brilliancy /
inaccuracy) and the better-alternative gap helpers that decide whether the
engine's move is "better" or merely "comparable"."""

from __future__ import annotations

import chess

from app.core.commentary import pipeline_flags
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
    if (pipeline_flags.enabled("scaled_class_limits") and gap > 0 and not alt.is_inferior
            and facts.move_quality in ("inaccuracy", "mistake", "blunder")):
        # With scaled limits an inaccuracy can lose less than the fixed gap; the class
        # of the move decides, so an error always has a better move beside it.
        return "better"
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


# --- v19 other_moves_concede ---
# Jakob's decision of 12 Nov 2026: a much worse second choice is not presented as "Weaker
# was ..." and its line is not explained; the comment says what other moves concede.
_PIECE_WORDS = {chess.QUEEN: "queen", chess.ROOK: "rook", chess.BISHOP: "bishop",
                chess.KNIGHT: "knight", chess.PAWN: "pawn"}
_COUNT_WORDS = {1: "a", 2: "two", 3: "three", 4: "four"}
_CONCEDED_WORDS = {"bishop pair": "the bishop pair"}


def clearly_worse_runner_up(facts: CommentFacts, *, threshold_cp: int = ALT_MATERIAL_GAP_CP) -> bool:
    """The played move is the engine's choice and the next-best move is clearly worse."""
    alt = facts.better_alternative
    return bool(pipeline_flags.enabled("other_moves_concede") and alt is not None and alt.is_inferior
                and alternative_comparison(facts, threshold_cp=threshold_cp) == "weaker")


def _material_conceded(facts: CommentFacts) -> str | None:
    """What the mover is short of at the end of the runner-up's line compared with the end of
    the played line, net of the opponent ('a knight', 'two pawns'). None when the difference
    is not a plain loss (an exchange of unequal pieces, or no difference)."""
    try:
        played = chess.Board(facts.display_line.leaf_fen or facts.display_line.fens[-1])
        other_line = facts.better_alternative.display_line
        other = chess.Board(other_line.leaf_fen or other_line.fens[-1])
    except (AttributeError, IndexError, ValueError):
        return None
    mover = chess.WHITE if facts.mover == "White" else chess.BLACK

    def net(board, piece):
        return len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover))

    change = {piece: net(other, piece) - net(played, piece) for piece in _PIECE_WORDS}
    if any(v > 0 for v in change.values()) or not any(v < 0 for v in change.values()):
        return None
    parts = [f"{_COUNT_WORDS.get(-v, str(-v))} {_PIECE_WORDS[piece]}{'s' if v < -1 else ''}"
             for piece, v in change.items() if v < 0]
    return " and ".join(parts)


def other_moves_concede(facts: CommentFacts) -> list[str]:
    """What moves other than the played one concede, by the runner-up: material first, then
    the feature groups in which the played line comes out ahead. At most three."""
    from .narrative import _feature_comparison  # narrative imports this module

    material = _material_conceded(facts)
    groups = (_feature_comparison(facts) or {}).get("favours_played_move") or []
    features = [_CONCEDED_WORDS.get(g, g) for g in groups if not (material and g == "material")]
    return (([material] if material else []) + features)[:3]
