"""Move-intent predicates (Guid: what the move is *for*).

Feature deltas describe what changed; these predicates describe the plan a
quiet move serves. All are cheap board comparisons between the position right
after the move and the starting position, computed on demand inside the rule
context (no new vector entries — intent is per-move, not a position metric):

  - newly attacked enemy piece (h4-h5 ideas, rook lifts, knight outposts
    aiming at a defender) -> "targets the bishop on g6"
  - pawn phalanx facing an enemy pawn with own support behind -> "prepares
    the break f5"
  - pawn advance opening a file for an own rook behind it -> "opens the
    h-file for the rook"

Each predicate yields at most one claim; ``rule_intent`` keeps the strongest.
"""

from __future__ import annotations

import chess

from app.models.comment_facts import Claim

# Intent claims rank above routine positional swings but below material and
# concrete defects; tuned to clear min_claim_cp (8) reliably.
INTENT_CLAIM_CP = 20


def _square_name(sq: int) -> str:
    return chess.square_name(sq)


def _piece_label(piece: chess.Piece) -> str:
    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
    }
    return names.get(piece.piece_type, "piece")


def newly_attacked_enemy_piece(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> tuple[int, str] | None:
    """(square, piece label) of the most valuable enemy piece the move now
    attacks that was not attacked before. Skips pawns (attacking pawns is
    rarely an idea in itself)."""
    enemy = not mover
    best: tuple[int, int, str] | None = None  # (-value, square, label)
    for sq, piece in leaf.piece_map().items():
        if piece.color != enemy or piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if not leaf.is_attacked_by(mover, sq):
            continue
        if start.is_attacked_by(mover, sq):
            continue  # already attacked before the move — nothing new
        value = -piece.piece_type  # QUEEN=5 sorts first
        if best is None or value < best[0]:
            best = (value, sq, _piece_label(piece))
    if best is None:
        return None
    return best[1], best[2]


def pawn_break_prepared(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> str | None:
    """File letter of a break the moved pawn now threatens: the moved pawn's
    file has exactly one enemy pawn directly ahead (a capturable target), the
    pawn is supported by a friendly pawn, and the advance was quiet."""
    enemy = not mover
    # Which pawn moved? Diff the pawn sets.
    start_pawns = set(start.pieces(chess.PAWN, mover))
    leaf_pawns = set(leaf.pieces(chess.PAWN, mover))
    arrived = leaf_pawns - {sq for sq in start_pawns}
    if len(arrived) != 1 or len(start_pawns - leaf_pawns) != 1:
        return None
    dest = arrived.pop()
    if chess.square_rank(dest) == 0 or chess.square_rank(dest) == 7:
        return None
    f = chess.square_file(dest)
    ahead_sq = dest + (8 if mover == chess.WHITE else -8)
    if not 0 <= ahead_sq < 64:
        return None
    if leaf.piece_type_at(ahead_sq) != chess.PAWN:
        return None
    if leaf.piece_at(ahead_sq).color != enemy:
        return None
    # Supported by a friendly pawn from behind-left/right on the origin side.
    support_from = []
    for df in (-1, 1):
        sf = f + df
        back = dest - (8 if mover == chess.WHITE else -8)
        if 0 <= sf <= 7 and 0 <= back < 64:
            s = chess.square(sf, chess.square_rank(back))
            p = leaf.piece_at(s)
            if p and p.piece_type == chess.PAWN and p.color == mover:
                support_from.append(s)
    if not support_from:
        return None
    return chess.FILE_NAMES[f]


def file_opened_for_rook(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> int | None:
    """File index whose only pawn left it this move while an own rook now
    stands on that file behind the pawn's former position."""
    start_pawns = set(start.pieces(chess.PAWN, mover))
    leaf_pawns = set(leaf.pieces(chess.PAWN, mover))
    departed = start_pawns - leaf_pawns
    if len(departed) != 1:
        return None
    f = chess.square_file(departed.pop())
    for sq in leaf.pieces(chess.ROOK, mover):
        if chess.square_file(sq) == f:
            return f
    return None


def build_intent_claim(
    start: chess.Board, leaf: chess.Board, mover: str
) -> Claim | None:
    """The single strongest intent claim for a quiet move, or None."""
    color = chess.WHITE if mover == "White" else chess.BLACK
    side = "White" if color == chess.WHITE else "Black"

    target = newly_attacked_enemy_piece(start, leaf, color)
    if target is not None:
        sq, label = target
        return Claim(
            rule_id="intent_targets_piece",
            beneficiary=mover.lower(),
            text=f"{side} targets the {label} on {_square_name(sq)}.",
            text_state=f"{side} is targeting the {label} on {_square_name(sq)}.",
            features_involved=[],
            delta_cp=INTENT_CLAIM_CP,
        )

    break_file = pawn_break_prepared(start, leaf, color)
    if break_file is not None:
        return Claim(
            rule_id="intent_pawn_break",
            beneficiary=mover.lower(),
            text=f"{side} prepares the break {break_file}-file pawn advance.",
            text_state=(
                f"{side} is preparing a pawn break on the {break_file}-file."
            ),
            features_involved=[],
            delta_cp=INTENT_CLAIM_CP,
        )

    opened = file_opened_for_rook(start, leaf, color)
    if opened is not None:
        fname = chess.FILE_NAMES[opened]
        return Claim(
            rule_id="intent_opens_file",
            beneficiary=mover.lower(),
            text=f"{side} opens the {fname}-file for the rook.",
            text_state=f"The {fname}-file is open for {side.lower()}'s rook.",
            features_involved=[],
            delta_cp=INTENT_CLAIM_CP,
        )
    return None
