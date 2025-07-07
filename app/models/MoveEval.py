import chess
from typing import Optional, List, Dict

from app.models.AltMove import AltMove


class MoveEval:
    def __init__(
        self,
        id: int,
        depth: int,
        move: str,
        prev_shallow_score,
        shallow_score,
        shallow_diff,
        shallow_pv: List[str],
        prev_deep_score,
        deep_score,
        deep_diff,
        deep_pv: List[str],
        trace: Dict,
        trace_diff: Dict,
        board: Optional[chess.Board] = None,
        alt_moves: Optional[List[AltMove]] = [],
        extra_info: Optional[dict] = {},
    ):
        self.move = move
        self.id = id
        self.depth = depth
        self.prev_shallow_score = prev_shallow_score
        self.shallow_score = shallow_score
        self.shallow_diff = shallow_diff
        self.shallow_pv = shallow_pv
        self.prev_deep_score = prev_deep_score
        self.deep_score = deep_score
        self.deep_diff = deep_diff
        self.deep_pv = deep_pv
        self.trace = trace
        self.trace_diff = trace_diff
        self.alt_moves = alt_moves
        self.extra_info = extra_info
        if board:
            self.board = board.copy()

