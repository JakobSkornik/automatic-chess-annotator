"""Move category classifier."""

import unittest

import chess

from app.core.commentary.features.move_category import classify_move_event
from app.models.chess_events import (
    FutureLineDelta,
    MoveCategory,
    MoveEvent,
    MoveEventType,
    MoveQuality,
    TacticalMotif,
)


def _minimal_event(**kwargs: object) -> MoveEvent:
    b0 = chess.Board()
    b1 = b0.copy()
    b1.push_san("Nf3")
    defaults = dict(
        move_index=0,
        ply=5,
        san="Nf3",
        uci="g1f3",
        fen_before=b0.fen(),
        fen_after=b1.fen(),
        phase="middlegame",
        eval_before_cp=0,
        eval_after_cp=20,
        move_quality=MoveQuality.GOOD,
        event_type=MoveEventType.QUIET,
        tactical_motifs=[],
    )
    defaults.update(kwargs)
    return MoveEvent(**defaults)  # type: ignore[arg-type]


class TestMoveCategory(unittest.TestCase):
    def test_tactical_overrides(self) -> None:
        me = _minimal_event(tactical_motifs=[TacticalMotif.FORK])
        self.assertEqual(classify_move_event(me), MoveCategory.TACTICAL)

    def test_future_line_forcing(self) -> None:
        me = _minimal_event()
        fl = FutureLineDelta(eval_gap_cp=120)
        self.assertEqual(classify_move_event(me, fl), MoveCategory.FORCING)


if __name__ == "__main__":
    unittest.main()
