"""Named tactics of a single move, certified on the board after it.

- fork: the moved piece attacks two or more enemy pieces it could win
  (the king, a more valuable piece, or an undefended one), and is not simply lost;
- discovered attack / check: moving the piece uncovers another piece of the
  mover against the king or a piece it could win;
- mate threat: if the opponent passed, the mover would have a mate in one;
- early queen: the queen leaves its square in the first ten moves while two or
  more of the mover's minor pieces are still on their starting squares.

These are the patterns of the tactical detectors in ``features.tactical_motifs``,
restated with the exchange check so that each one is a fact, not a guess.
"""
from __future__ import annotations

import chess

from app.models.comment_facts import Claim

from .exchange import VALUE, see

MINOR_HOMES = {chess.WHITE: (chess.B1, chess.C1, chess.F1, chess.G1), chess.BLACK: (chess.B8, chess.C8, chess.F8, chess.G8)}


def _label(color: chess.Color) -> str:
    return "White" if color else "Black"


def _winnable(board: chess.Board, attacker: int, target: int) -> bool:
    """The attacker could win the target: it is the king, worth more, or undefended."""
    piece = board.piece_at(target)
    if piece.piece_type == chess.KING:
        return True
    if VALUE[piece.piece_type] > VALUE[board.piece_type_at(attacker)]:
        return True
    return not board.is_attacked_by(piece.color, target)


def _targets(board: chess.Board, square: int) -> list[int]:
    color = board.color_at(square)
    return [t for t in board.attacks(square)
            if board.color_at(t) == (not color) and board.piece_type_at(t) != chess.PAWN and _winnable(board, square, t)]


def _names(board: chess.Board, squares) -> str:
    parts = [f"the {chess.piece_name(board.piece_type_at(s))} on {chess.square_name(s)}" for s in squares]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def move_tactic_claims(before: chess.Board, move: chess.Move, after: chess.Board, common: dict) -> list[Claim]:
    mover = before.turn
    side, opponent = _label(mover), _label(not mover)
    claims = []

    def add(rule_id: str, text: str, feature: str, beneficiary: chess.Color = mover) -> None:
        claims.append(Claim(rule_id=rule_id, text=text, text_state=text, features_involved=[feature],
                            **{**common, "beneficiary": _label(beneficiary).lower()}))

    piece_name = chess.piece_name(before.piece_type_at(move.from_square))
    safe = see(before, move) >= 0
    # Fork: the moved piece attacks two or more winnable pieces at once.
    forked = _targets(after, move.to_square)
    if safe and len(forked) >= 2:
        add("fork", f"{side}'s {piece_name} on {chess.square_name(move.to_square)} attacks "
                    f"{_names(after, forked)} at once (a fork).", "TACTIC_FORK")
    # Discovered attack or check: another piece of the mover is uncovered.
    for square in chess.SquareSet(after.occupied_co[mover]):
        if square == move.to_square or after.piece_type_at(square) not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        new = [t for t in _targets(after, square) if t not in before.attacks(square)]
        if not new:
            continue
        slider = f"{side}'s {chess.piece_name(after.piece_type_at(square))} on {chess.square_name(square)}"
        if any(after.piece_type_at(t) == chess.KING for t in new):
            add("discovered_check", f"Moving the {piece_name} uncovers a check by {slider} (a discovered check).",
                "TACTIC_DISCOVERED")
        else:
            add("discovered_attack", f"Moving the {piece_name} uncovers {slider} against {_names(after, new)} "
                                     f"(a discovered attack).", "TACTIC_DISCOVERED")
    if len(after.checkers()) >= 2:
        add("double_check", f"{side} gives a double check: the {opponent} king must move.", "TACTIC_DOUBLE_CHECK")
    # Mate threat: if the opponent passed, the mover would mate in one.
    if not after.is_check() and not after.is_game_over():
        threat = after.copy(stack=False)
        threat.push(chess.Move.null())
        for reply in threat.legal_moves:
            threat.push(reply)
            mate = threat.is_checkmate()
            threat.pop()
            if mate:
                add("mate_threat", f"{side} threatens mate with {threat.san(reply)}.", "TACTIC_MATE_THREAT")
                break
    # Early queen: out before the minor pieces are developed.
    if (before.piece_type_at(move.from_square) == chess.QUEEN and before.fullmove_number <= 10
            and move.from_square == (chess.D1 if mover else chess.D8)):
        home = sum(1 for sq in MINOR_HOMES[mover] if before.piece_at(sq) in
                   (chess.Piece(chess.KNIGHT, mover), chess.Piece(chess.BISHOP, mover)))
        if home >= 2:
            add("early_queen", f"{side}'s queen comes out while {home} of {side}'s minor pieces are still "
                               f"on their starting squares.", "DEVELOPMENT", beneficiary=not mover)
    return claims
