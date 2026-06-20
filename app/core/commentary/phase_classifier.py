"""Game-phase classification, computable before any engine call.

Phase definitions (Guid-style pipeline):
  early — the full UCI prefix of the game so far is still matched by the ECO book;
          these plies are never sent to the engine.
  end   — fewer than ``ENDGAME_PIECE_THRESHOLD`` minor+major pieces remain in total.
  mid   — everything else.
"""

from __future__ import annotations

import os

import chess

from app.core.commentary.openings.eco_book import ECOBook


def endgame_piece_threshold() -> int:
    """Total knights+bishops+rooks+queens (both sides) below which the game is an endgame."""
    try:
        return int(os.environ.get("ENDGAME_PIECE_THRESHOLD", "7"))
    except ValueError:
        return 7


def minor_major_piece_count(board: chess.Board) -> int:
    count = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count += len(board.pieces(piece_type, chess.WHITE))
        count += len(board.pieces(piece_type, chess.BLACK))
    return count


class PhaseClassifier:
    """Classifies plies into early/mid/end using only the board and the ECO book."""

    def __init__(self, eco_book: ECOBook | None = None) -> None:
        self._eco = eco_book or ECOBook()

    def in_book(self, uci_prefix: list[str]) -> bool:
        """True while the ECO book covers every ply played so far."""
        if not uci_prefix:
            return False
        info, matched = self._eco.match(uci_prefix)
        return info is not None and matched >= len(uci_prefix)

    def classify(self, board_after: chess.Board, uci_prefix: list[str]) -> str:
        if self.in_book(uci_prefix):
            return "early"
        if minor_major_piece_count(board_after) < endgame_piece_threshold():
            return "end"
        return "mid"
