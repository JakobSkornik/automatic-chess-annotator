"""Static exchange evaluation (SEE), shared by quiescence and the exchange facts.

SEE plays out the capture sequence on one square, each side capturing with its
least valuable piece and free to stop when continuing would lose material.
Only legal captures are used, so pinned pieces do not take part. The result is
exact for that square; it does not see tactics elsewhere on the board.
"""
from __future__ import annotations

import chess

VALUE = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}


def captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return VALUE[chess.PAWN]
    piece = board.piece_at(move.to_square)
    return VALUE[piece.piece_type] if piece else 0


def cheapest_capture(board: chess.Board, square: int) -> chess.Move | None:
    captures = [m for m in board.generate_legal_captures() if m.to_square == square]
    return min(captures, key=lambda m: VALUE[board.piece_type_at(m.from_square)], default=None)


def see(board: chess.Board, move: chess.Move) -> int:
    """Material the mover gains (+) or loses (-) on ``move``'s target square, in centipawns."""
    gains = [captured_value(board, move)]
    piece = board.piece_at(move.from_square)
    on_square = VALUE[move.promotion] if move.promotion else VALUE[piece.piece_type]
    if move.promotion:
        gains[0] += VALUE[move.promotion] - VALUE[chess.PAWN]
    probe = board.copy(stack=False)
    probe.push(move)
    while (reply := cheapest_capture(probe, move.to_square)) is not None:
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


def as_side_to_move(board: chess.Board, color: chess.Color) -> chess.Board:
    """A copy with ``color`` to move, without fabricating an en-passant right."""
    view = board.copy(stack=False)
    if view.turn != color:
        view.ep_square = None
    view.turn = color
    return view


def en_prise_squares(board: chess.Board, victim: chess.Color, min_gain: int = VALUE[chess.PAWN]) -> list[int]:
    """Squares of ``victim``'s pieces and pawns the opponent can win material on (SEE)."""
    view = as_side_to_move(board, not victim)
    out = []
    for square in chess.SquareSet(board.occupied_co[victim]):
        if board.piece_type_at(square) == chess.KING:
            continue
        gain, _ = best_capture_gain(view, square)
        if gain >= min_gain:
            out.append(square)
    return out
