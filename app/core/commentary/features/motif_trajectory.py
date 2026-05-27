"""Track sustained motif patterns across consecutive moves."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from app.models.chess_events import Episode, MoveEvent, StrategicMotif, TacticalMotif

MIN_STREAK = 3


def _motif_keys(ev: MoveEvent) -> List[str]:
    return [m.value for m in ev.tactical_motifs] + [m.value for m in ev.strategic_motifs]


def compute_move_trajectories(events: List[MoveEvent]) -> None:
    """
    Mutates ``events`` in place: sets ``motif_trajectory`` when a motif persists
    for ``MIN_STREAK`` consecutive plies.
    """
    if not events:
        return

    streaks: Dict[str, int] = defaultdict(int)
    prev_ply = events[0].ply - 1

    for ev in events:
        if ev.ply != prev_ply + 1:
            streaks.clear()
        prev_ply = ev.ply

        current: Set[str] = set(_motif_keys(ev))
        for key in list(streaks.keys()):
            if key in current:
                streaks[key] += 1
            else:
                del streaks[key]
        for key in current:
            if key not in streaks:
                streaks[key] = 1

        sustained = [k for k, n in streaks.items() if n >= MIN_STREAK]
        if sustained:
            primary = sustained[0]
            ev.motif_trajectory = f"sustained_{primary}"


def compute_episode_trajectories(episodes: List[Episode]) -> None:
    """Set ``Episode.motif_trajectory`` from dominant sustained patterns in the chunk."""
    for ep in episodes:
        counts: Dict[str, int] = defaultdict(int)
        for ev in ep.move_events:
            if ev.motif_trajectory:
                key = ev.motif_trajectory.replace("sustained_", "")
                counts[key] += 1
        if counts:
            best = max(counts.items(), key=lambda x: x[1])[0]
            ep.motif_trajectory = f"episode_sustained_{best}"


def trajectory_summary(events: List[MoveEvent]) -> List[str]:
    """Collect unique trajectory labels for prompts."""
    out: List[str] = []
    seen: Set[str] = set()
    for ev in events:
        if ev.motif_trajectory and ev.motif_trajectory not in seen:
            seen.add(ev.motif_trajectory)
            out.append(ev.motif_trajectory)
    return out
