"""Conservative board-supported relationships at a selected evidence node.

These predicates compare the baseline with the displayed support position.
They do not infer a player's intention or promise a future engine outcome.
Each predicate yields at most one claim; ``rule_intent`` keeps the first.
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


def _for_side(board: chess.Board, color: chess.Color) -> chess.Board:
    """Inspect legal opportunities for a side, without fabricating en passant."""
    result = board.copy(stack=False)
    if result.turn != color:
        result.ep_square = None
    result.turn = color
    return result


def _can_capture(board: chess.Board, color: chess.Color, square: int) -> bool:
    view = _for_side(board, color)
    return any(move.to_square == square for move in view.generate_legal_captures())


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
        if not _can_capture(leaf, mover, sq):
            continue
        if start.piece_at(sq) == piece and _can_capture(start, mover, sq):
            continue  # already attacked before the move — nothing new
        from app.core.commentary.rules.tactical_evidence import takes_attacker_safely
        if takes_attacker_safely(leaf, sq, leaf.attackers(mover, sq)):
            continue  # it can simply take the attacker: an offer, not a target
        value = -piece.piece_type  # QUEEN=5 sorts first
        if best is None or value < best[0]:
            best = (value, sq, _piece_label(piece))
    if best is None:
        return None
    return best[1], best[2]


def _prepared_break_move(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> chess.Move | None:
    """A supported, quietly advanced pawn can advance again to contact a pawn.

    A pawn blocked by an enemy pawn directly ahead cannot make this break.
    Require a legal next advance and a new diagonal enemy-pawn contact; this
    describes an available idea, not a forced continuation.
    """
    before = set(start.pieces(chess.PAWN, mover))
    after = set(leaf.pieces(chess.PAWN, mover))
    arrived, departed = after - before, before - after
    if len(arrived) != 1 or len(departed) != 1:
        return None
    dest, origin = next(iter(arrived)), next(iter(departed))
    step = 8 if mover else -8
    if chess.square_file(origin) != chess.square_file(dest) or dest-origin not in (step, 2*step):
        return None
    if not (leaf.attackers(mover, dest) & leaf.pieces(chess.PAWN, mover)):
        return None
    target = dest + step
    if not 0 <= target < 64 or chess.square_rank(target) in (0, 7):
        return None
    move = chess.Move(dest, target)
    view = _for_side(leaf, mover)
    if move not in view.legal_moves:
        return None
    enemy_pawns = leaf.pieces(chess.PAWN, not mover)
    if leaf.attacks(dest) & enemy_pawns:
        return None  # already in contact; no newly prepared advance
    view.push(move)
    if not (view.attacks(target) & enemy_pawns):
        return None
    return move


def pawn_break_prepared(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> str | None:
    """File of a supported pawn with a legal, pawn-challenging next advance."""
    move = _prepared_break_move(start, leaf, mover)
    return chess.FILE_NAMES[chess.square_file(move.from_square)] if move else None


def _new_rook_file(
    start: chess.Board, leaf: chess.Board, mover: chess.Color, *, semi_open: bool
) -> int | None:
    """A newly open/semi-open file, with a rook that can use its file ray."""
    for sq in sorted(leaf.pieces(chess.ROOK, mover)):
        f = chess.square_file(sq)
        mask = chess.BB_FILES[f]
        own_before = start.pieces_mask(chess.PAWN, mover) & mask
        own_after = leaf.pieces_mask(chess.PAWN, mover) & mask
        enemy_after = leaf.pieces_mask(chess.PAWN, not mover) & mask
        if own_after or bool(enemy_after) != semi_open:
            continue
        if semi_open:
            if not own_before:
                continue
        elif not (start.pawns & mask):
            continue  # already open before the line
        # A rook boxed in by friendly pieces has not gained usable access.
        if not (leaf.attacks_mask(sq) & mask & ~leaf.occupied_co[mover]):
            continue
        return f
    return None


def file_opened_for_rook(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> int | None:
    """Newly pawn-free file for an own rook; advances on the file do not count."""
    return _new_rook_file(start, leaf, mover, semi_open=False)


def file_semi_opened_for_rook(
    start: chess.Board, leaf: chess.Board, mover: chess.Color
) -> int | None:
    """Own pawns left the rook's file, but at least one enemy pawn remains."""
    return _new_rook_file(start, leaf, mover, semi_open=True)


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

    break_move = _prepared_break_move(start, leaf, color)
    if break_move is not None:
        advance = chess.square_name(break_move.to_square)
        return Claim(
            rule_id="intent_pawn_break",
            beneficiary=mover.lower(),
            text=f"{side} can consider {advance} to challenge an enemy pawn.",
            text_state=f"{side} can consider {advance} to challenge an enemy pawn.",
            features_involved=[],
            delta_cp=INTENT_CLAIM_CP,
        )

    opened = file_opened_for_rook(start, leaf, color)
    if opened is not None:
        fname = chess.FILE_NAMES[opened]
        return Claim(
            rule_id="intent_opens_file",
            beneficiary=mover.lower(),
            text=f"The {fname}-file becomes open for {side.lower()}'s rook.",
            text_state=f"The {fname}-file is open for {side.lower()}'s rook.",
            features_involved=[],
            delta_cp=INTENT_CLAIM_CP,
        )
    semi_opened = file_semi_opened_for_rook(start, leaf, color)
    if semi_opened is not None:
        fname = chess.FILE_NAMES[semi_opened]
        return Claim(
            rule_id="intent_semi_opens_file",
            beneficiary=mover.lower(),
            text=f"The {fname}-file becomes semi-open for {side.lower()}'s rook.",
            text_state=f"The {fname}-file is semi-open for {side.lower()}'s rook.",
            features_involved=[],
            delta_cp=INTENT_CLAIM_CP,
        )
    return None
