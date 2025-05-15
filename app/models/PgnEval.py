from typing import List
from app.models.MoveEval import MoveEval


class PgnEval:
    def __init__(self, move_evals: List[MoveEval]):
        self.move_evals = move_evals
