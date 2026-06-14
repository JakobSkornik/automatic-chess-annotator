"""Group move events into narrative episodes."""

from __future__ import annotations

from app.models.chess_events import Episode, MoveEvent, TacticalMotif


def _theme_from_events(events: list[MoveEvent]) -> str:
    if not events:
        return "maneuvering"
    types = [e.event_type.value for e in events]
    if any("tactic" in t or "swing" in t for t in types):
        return "tactical_complications"
    phases = {e.phase for e in events}
    if "endgame" in phases:
        return "endgame_technique"
    return "positional_play"


class EpisodeSegmenter:
    MAX_EPISODE_LENGTH = 10
    ADVANTAGE_FLIP_CP = 100

    def segment(self, events: list[MoveEvent]) -> list[Episode]:
        if not events:
            return []

        raw_chunks: list[list[MoveEvent]] = []
        current: list[MoveEvent] = []
        prev_phase: str | None = None
        prev_eval: int | None = None

        for ev in events:
            should_split = False
            if current and prev_phase and ev.phase != prev_phase:
                should_split = True
            if (
                (current and prev_eval is not None and ev.eval_after_cp is not None)
                and (prev_eval > self.ADVANTAGE_FLIP_CP)
                != (ev.eval_after_cp > self.ADVANTAGE_FLIP_CP)
                and abs(ev.eval_after_cp - prev_eval) > 150
            ):
                should_split = True

            if current and should_split:
                raw_chunks.append(current)
                current = []

            current.append(ev)
            prev_phase = ev.phase
            prev_eval = ev.eval_after_cp

        if current:
            raw_chunks.append(current)

        chunks: list[list[MoveEvent]] = []
        for chunk in raw_chunks:
            start = 0
            while start < len(chunk):
                end = min(start + self.MAX_EPISODE_LENGTH, len(chunk))
                chunks.append(chunk[start:end])
                start = end

        episodes: list[Episode] = []
        for j, chunk in enumerate(chunks):
            evals = [e.eval_after_cp for e in chunk if e.eval_after_cp is not None]
            motifs: list[TacticalMotif] = []
            for e in chunk:
                for m in e.tactical_motifs:
                    if m not in motifs:
                        motifs.append(m)
            theme = _theme_from_events(chunk)
            ep = Episode(
                episode_index=j,
                title=f"Moves {chunk[0].ply}-{chunk[-1].ply} ({theme})",
                start_ply=chunk[0].ply,
                end_ply=chunk[-1].ply,
                move_events=list(chunk),
                eval_start_cp=evals[0] if evals else None,
                eval_end_cp=evals[-1] if evals else None,
                eval_trend=evals,
                dominant_theme=theme,
                tactical_motifs_in_episode=motifs,
                phase=chunk[0].phase,
            )
            episodes.append(ep)

        return episodes
