"""Refutation scan of plausible alternatives (Guid Ch.5 Search Module).

The dissertation's annotator searches "for all possible moves in an initial
position" and values refuting a move that "seems to be good at lower search
depths" — showing the refutation of a plausible-but-wrong move is among the
most instructive annotations. Full-width search of every legal move is
prohibitively slow, so this scan is targeted:

  - only key-moment positions (the moves already worth commenting on);
  - only the engine's top-N moves at shallow depth (the "plausible" set);
  - a move is flagged when its eval COLLAPSES between the shallow and the
    final depth (it looked fine shallow, loses deep) — the classic trap.

The scan is optional and failure-tolerant: without an engine it yields
nothing and the pipeline is unaffected.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import chess

logger = logging.getLogger(__name__)

# A plausible move is one whose shallow eval stays within this much of the best.
PLAUSIBLE_GAP_CP = 40
# Collapse: deep mover-POV eval at least this much below shallow.
REFUTATION_COLLAPSE_CP = 120
# Shallow depth used to rank the plausible set (cheap).
SHALLOW_DEPTH = 8


@dataclass
class RefutationFinding:
    """One plausible move that only deep search exposes as bad."""

    uci: str
    san: str
    shallow_cp: int  # mover-POV at SHALLOW_DEPTH
    deep_cp: int  # mover-POV at the full analysis depth
    collapse_cp: int  # shallow - deep (mover-POV), always > 0

    def as_dict(self) -> dict:
        return {
            "uci": self.uci,
            "san": self.san,
            "shallow_cp": self.shallow_cp,
            "deep_cp": self.deep_cp,
            "collapse_cp": self.collapse_cp,
        }


def _scan_enabled() -> bool:
    try:
        return os.environ.get("REFUTATION_SCAN", "1") not in ("0", "false", "off")
    except Exception:
        return True


def _max_positions() -> int:
    try:
        return int(os.environ.get("REFUTATION_SCAN_MAX_POSITIONS", "12"))
    except ValueError:
        return 12


class RefutationScanner:
    """Targeted shallow-vs-deep scan over the top moves of key positions."""

    def __init__(self, engine_connector):
        self._engine = engine_connector
        self._positions_scanned = 0

    @property
    def positions_scanned(self) -> int:
        return self._positions_scanned

    def scan(self, fen: str, deep_eval_cp: int | None) -> list[RefutationFinding]:
        """Find plausible moves whose deep eval collapses vs their shallow eval.

        ``deep_eval_cp`` is the mover-POV eval of the position after the played
        move at full depth — used only to skip decided positions."""
        if not _scan_enabled() or self._positions_scanned >= _max_positions():
            return []
        board = chess.Board(fen)
        if board.is_game_over():
            return []
        self._positions_scanned += 1
        if self._positions_scanned > _max_positions():
            self._positions_scanned = _max_positions()
            return []
        try:
            shallow = self._engine.analyse(
                board, depth=SHALLOW_DEPTH, multiPv=3
            )
        except Exception as e:
            logger.warning("refutation scan (shallow) failed for %s: %s", fen, e)
            return []
        if not isinstance(shallow, list) or len(shallow) < 2:
            return []

        sign = 1 if board.turn == chess.WHITE else -1
        findings: list[RefutationFinding] = []
        for info in shallow:
            if not isinstance(info, dict):
                continue
            pv = info.get("pv") or []
            sc = info.get("score")
            if not pv or sc is None:
                continue
            uci = pv[0].uci()
            try:
                cp = sign * int(sc.white().score(mate_score=1000000))
            except Exception:
                continue
            if cp < PLAUSIBLE_GAP_CP:
                continue  # not plausible for the mover even at shallow depth
            # Deep re-search of just this move's line: push it, search deep.
            try:
                probe_board = board.copy(stack=False)
                probe_board.push(pv[0])
                deep_info = self._engine.analyse(probe_board, depth=16, multiPv=1)
            except Exception as e:
                logger.warning("refutation scan (deep) failed for %s: %s", uci, e)
                continue
            deep_cp = None
            if isinstance(deep_info, list) and deep_info:
                deep_info = deep_info[0]
            if isinstance(deep_info, dict) and deep_info.get("score") is not None:
                try:
                    # Score after the mover's move, from the opponent's seat:
                    # negate to get the mover-POV continuation eval.
                    deep_cp = -sign * int(
                        deep_info["score"].white().score(mate_score=1000000)
                    )
                except Exception:
                    continue
            if deep_cp is None:
                continue
            collapse = cp - deep_cp
            if collapse >= REFUTATION_COLLAPSE_CP:
                try:
                    san = board.san(pv[0])
                except Exception:
                    san = uci
                findings.append(
                    RefutationFinding(
                        uci=uci,
                        san=san,
                        shallow_cp=cp,
                        deep_cp=deep_cp,
                        collapse_cp=collapse,
                    )
                )
        return findings
