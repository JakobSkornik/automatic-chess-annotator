"""Board reconstruction and score-conversion helpers used across the assembly."""

from __future__ import annotations

import chess
from chess.pgn import Game

from app.models.GameJson import MoveScore

from .constants import MATE_SCORE


def _board_before_mainline_move(game: Game, move_index: int) -> chess.Board:
    board = game.board()
    for i, gm in enumerate(game.mainline_moves()):
        if i >= move_index:
            break
        board.push(gm)
    return board


def _score_to_move_score(score_val: float | None) -> MoveScore:
    if score_val is None:
        return MoveScore(cp=None, mate=None)
    if abs(score_val) > MATE_SCORE - 1000:
        moves_to_mate = MATE_SCORE - abs(score_val)
        if score_val < 0:
            mate = -moves_to_mate
        else:
            mate = moves_to_mate
        return MoveScore(cp=None, mate=mate)
    return MoveScore(cp=round(score_val), mate=None)
