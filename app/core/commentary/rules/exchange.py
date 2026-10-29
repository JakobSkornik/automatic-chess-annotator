"""Static exchange evaluation (SEE) and the facts it certifies about a move.

SEE plays out the capture sequence on one square, each side capturing with its
least valuable piece and free to stop when continuing would lose material.
Only legal captures are used, so pinned pieces do not take part. The result is
exact for that square; it does not see tactics elsewhere on the board.
"""
from __future__ import annotations

import chess

from app.models.comment_facts import Claim

VALUE = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
# Smallest exchange loss reported as a fact, and smallest gain that wins material: one pawn.
MIN_LOSS_CP = 100
MATERIAL_GAIN_CP = 100


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return VALUE[chess.PAWN]
    piece = board.piece_at(move.to_square)
    return VALUE[piece.piece_type] if piece else 0


def _cheapest_capture(board: chess.Board, square: int) -> chess.Move | None:
    captures = [m for m in board.generate_legal_captures() if m.to_square == square]
    return min(captures, key=lambda m: VALUE[board.piece_type_at(m.from_square)], default=None)


def see(board: chess.Board, move: chess.Move) -> int:
    """Material the mover gains (+) or loses (-) on ``move``'s target square, in centipawns."""
    gains = [_captured_value(board, move)]
    piece = board.piece_at(move.from_square)
    on_square = VALUE[move.promotion] if move.promotion else VALUE[piece.piece_type]
    if move.promotion:
        gains[0] += VALUE[move.promotion] - VALUE[chess.PAWN]
    probe = board.copy(stack=False)
    probe.push(move)
    while (reply := _cheapest_capture(probe, move.to_square)) is not None:
        gains.append(on_square - gains[-1])
        on_square = VALUE[probe.piece_type_at(reply.from_square)]
        probe.push(reply)
    # Each side may stop capturing when going on would cost it material.
    for depth in range(len(gains) - 1, 0, -1):
        gains[depth - 1] = -max(-gains[depth - 1], gains[depth])
    return gains[0]


def best_capture_gain(board: chess.Board, square: int) -> tuple[int, chess.Move | None]:
    """The side to move's best SEE gain from capturing on ``square`` (0 when no capture pays)."""
    best, best_move = 0, None
    for move in board.generate_legal_captures():
        if move.to_square == square:
            gain = see(board, move)
            if gain > best:
                best, best_move = gain, move
    return best, best_move


def _as_side_to_move(board: chess.Board, color: chess.Color) -> chess.Board:
    view = board.copy(stack=False)
    if view.turn != color:
        view.ep_square = None
    view.turn = color
    return view


def _name(board: chess.Board, square: int) -> str:
    piece = board.piece_at(square)
    side = "White" if piece.color else "Black"
    return f"{side}'s {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"


def material_loss_claims(before: chess.Board, move: chess.Move, after: chess.Board,
                         common: dict) -> list[Claim]:
    """Facts that explain why a move loses material on the board, at its first ply.

    - the move captures or lands on a square where the exchange loses material;
    - another piece of the mover becomes capturable with gain (it lost a defender
      or a line opened), when it was safe before the move.
    """
    claims = []
    mover = before.turn
    side, opponent = ("White", "Black") if mover else ("Black", "White")
    piece_name = chess.piece_name(before.piece_type_at(move.from_square))
    target = chess.square_name(move.to_square)
    loss = -see(before, move)
    if loss >= MIN_LOSS_CP:
        reply = _cheapest_capture(after, move.to_square)
        if reply is not None:
            taker = _name(after, reply.from_square)
            if before.is_capture(move):
                victim = chess.piece_name(before.piece_type_at(move.to_square)) if before.piece_at(move.to_square) else "pawn"
                text = (f"{side}'s {piece_name} takes the {victim} on {target}, but {taker} defends it, "
                        f"so the exchange on {target} loses material for {side}.")
            else:
                text = (f"{side}'s {piece_name} moves to {target}, where {taker} can take it, "
                        f"and the exchange on {target} loses material for {side}.")
            claims.append(Claim(rule_id="exchange_loses_material", text=text, text_state=text,
                                features_involved=["TACTICAL_EXCHANGE"], **{**common, "beneficiary": opponent.lower()}))
    # Other pieces of the mover that the move leaves open to a winning capture.
    opponent_color = not mover
    for square in chess.SquareSet(after.occupied_co[mover]):
        if square == move.to_square or after.piece_type_at(square) == chess.KING:
            continue
        if before.piece_at(square) != after.piece_at(square):
            continue
        gain_after, capture = best_capture_gain(_as_side_to_move(after, opponent_color), square)
        if gain_after < MIN_LOSS_CP or capture is None:
            continue
        gain_before, _ = best_capture_gain(_as_side_to_move(before, opponent_color), square)
        if gain_before >= MIN_LOSS_CP:
            continue  # already capturable before the move: not this move's doing
        attacker = _name(after, capture.from_square)
        text = f"After {side}'s move, {_name(after, square)} is no longer safe: {attacker} can win it."
        claims.append(Claim(rule_id="piece_left_en_prise", text=text, text_state=text,
                            features_involved=["TACTICAL_EXCHANGE"], **{**common, "beneficiary": opponent.lower()}))
    return claims
