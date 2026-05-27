"""Tests for motif trajectory tracking."""

import unittest

from app.core.commentary.features.motif_trajectory import (
    compute_episode_trajectories,
    compute_move_trajectories,
)
from app.models.chess_events import (
    Episode,
    MoveEvent,
    MoveEventType,
    MoveQuality,
    StrategicMotif,
)


def _ev(ply: int, motifs: list[StrategicMotif]) -> MoveEvent:
    return MoveEvent(
        move_index=ply - 1,
        ply=ply,
        san="Nf3",
        uci="g1f3",
        fen_before="",
        fen_after="",
        phase="middlegame",
        move_quality=MoveQuality.GOOD,
        event_type=MoveEventType.QUIET,
        strategic_motifs=motifs,
    )


class TestMotifTrajectory(unittest.TestCase):
    def test_sustained_outpost(self) -> None:
        events = [
            _ev(1, [StrategicMotif.OUTPOST]),
            _ev(2, [StrategicMotif.OUTPOST]),
            _ev(3, [StrategicMotif.OUTPOST]),
        ]
        compute_move_trajectories(events)
        self.assertEqual(events[2].motif_trajectory, "sustained_outpost")

    def test_streak_breaks_on_gap(self) -> None:
        events = [
            _ev(1, [StrategicMotif.OUTPOST]),
            _ev(2, []),
            _ev(3, [StrategicMotif.OUTPOST]),
        ]
        compute_move_trajectories(events)
        self.assertIsNone(events[0].motif_trajectory)

    def test_episode_trajectory(self) -> None:
        ev = _ev(3, [StrategicMotif.OUTPOST])
        ev.motif_trajectory = "sustained_outpost"
        ep = Episode(
            episode_index=0,
            title="test",
            start_ply=1,
            end_ply=3,
            move_events=[ev],
        )
        compute_episode_trajectories([ep])
        self.assertEqual(ep.motif_trajectory, "episode_sustained_outpost")


if __name__ == "__main__":
    unittest.main()
