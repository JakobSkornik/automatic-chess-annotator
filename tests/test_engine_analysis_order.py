"""Reverse-order engine analysis (REVERSE_ORDER_ENGINE_ANALYSIS).

Covers:
  (a) reverse-order analysis still produces one evaluation/PV per move,
      correctly indexed to the right move (fen/uci/index untouched by order).
  (b) positions are actually sent to the engine last-move-first, and nothing
      in the new path clears the transposition hash between positions.
  (c) the old forward-order behavior still works when the flag is disabled.
"""

from __future__ import annotations

import asyncio
import io

import chess.pgn

from app.core.engine.analysis.engine_pass import AnalysisRetriever
from app.core.engine.analysis_retriever import (
    _run_engine_pass,
    reverse_order_engine_analysis_enabled,
)

PGN = "1. Nf3 Nf6 2. c4 c5 3. Nc3 Nc6 *"


class _StubEcoBook:
    """Never matches: every ply is treated as out-of-book (goes to the engine)."""

    def match(self, prefix):
        return None, 0


class _FakeWhitePov:
    def __init__(self, cp: int) -> None:
        self._cp = cp

    def score(self, mate_score: int) -> int:
        return self._cp


class _FakeScore:
    def __init__(self, cp: int = 0) -> None:
        self._cp = cp

    def white(self) -> _FakeWhitePov:
        return _FakeWhitePov(self._cp)


class _RecordingEngine:
    """Duck-typed stand-in for EngineConnector: records every FEN analyzed and
    counts any hash-clearing calls (there should never be any)."""

    def __init__(self) -> None:
        self.analyse_calls: list[str] = []
        self.ucinewgame_calls = 0
        self.clear_hash_calls = 0

    def analyse(self, board, depth=None, time_limit=None, multiPv=1):
        self.analyse_calls.append(board.fen())
        return [{"score": _FakeScore(0), "pv": []}]

    def analyse_with_depth_snapshots(self, board, depth, snapshot_depths=(), multiPv=1):
        self.analyse_calls.append(board.fen())
        return [{"score": _FakeScore(0), "pv": []}], {}

    def evaluate_position(self, fen, *, depth):
        self.analyse_calls.append(fen)
        return 0

    # If any future code path tried to reset the engine between positions it
    # would call one of these; they must stay at zero for the new path.
    def ucinewgame(self, *a, **k):
        self.ucinewgame_calls += 1

    def send_clear_hash(self, *a, **k):
        self.clear_hash_calls += 1


def _build_game() -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(PGN))
    assert game is not None
    return game


async def _noop_progress(_pct: float, _msg: str) -> None:
    return None


def _run_pass(monkeypatch, reverse: bool):
    monkeypatch.setenv("REVERSE_ORDER_ENGINE_ANALYSIS", "1" if reverse else "0")
    game = _build_game()
    engine = _RecordingEngine()
    retriever = AnalysisRetriever(engine, game, eco_book=_StubEcoBook())
    moves_list = retriever.get_move_list()
    rows = asyncio.run(_run_engine_pass(game, retriever, moves_list, _noop_progress))
    return rows, engine, moves_list


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("REVERSE_ORDER_ENGINE_ANALYSIS", "1")
    assert reverse_order_engine_analysis_enabled() is True
    monkeypatch.setenv("REVERSE_ORDER_ENGINE_ANALYSIS", "0")
    assert reverse_order_engine_analysis_enabled() is False
    monkeypatch.delenv("REVERSE_ORDER_ENGINE_ANALYSIS", raising=False)
    assert reverse_order_engine_analysis_enabled() is True  # default: enabled


def _first_occurrence_order_of_after_fens(engine, moves_list) -> list[str]:
    """First-occurrence order, restricted to each move's *after* FEN (ignores
    the incidental "before" FENs also sent to the engine for PV computation,
    which alias the previous move's after-FEN, plus the game's start FEN)."""
    after_fens = {m.position for m in moves_list}
    order: list[str] = []
    for fen in engine.analyse_calls:
        if fen in after_fens and fen not in order:
            order.append(fen)
    return order


def test_reverse_order_sends_last_move_first(monkeypatch):
    rows, engine, moves_list = _run_pass(monkeypatch, reverse=True)

    first_analysed_fens = _first_occurrence_order_of_after_fens(engine, moves_list)
    expected_reverse_order = [m.position for m in reversed(moves_list)]
    assert first_analysed_fens == expected_reverse_order

    # No hash-clearing ever happens between (or around) positions.
    assert engine.ucinewgame_calls == 0
    assert engine.clear_hash_calls == 0


def test_forward_order_still_works_when_disabled(monkeypatch):
    rows, engine, moves_list = _run_pass(monkeypatch, reverse=False)

    first_analysed_fens = _first_occurrence_order_of_after_fens(engine, moves_list)
    expected_forward_order = [m.position for m in moves_list]
    assert first_analysed_fens == expected_forward_order
    assert engine.ucinewgame_calls == 0
    assert engine.clear_hash_calls == 0


def test_reverse_order_results_still_indexed_to_correct_move(monkeypatch):
    rows, engine, moves_list = _run_pass(monkeypatch, reverse=True)

    assert len(rows) == len(moves_list)
    for idx, (row, move_obj) in enumerate(zip(rows, moves_list)):
        assert row.index == idx
        assert row.uci == move_obj.move
        assert row.fen_after == move_obj.position
        assert row.score_cp is not None
        assert isinstance(row.pvs, list)


def test_forward_and_reverse_produce_same_per_move_data(monkeypatch):
    """Only the engine call ORDER changes; the stored per-move data must match."""
    forward_rows, _, _ = _run_pass(monkeypatch, reverse=False)
    reverse_rows, _, _ = _run_pass(monkeypatch, reverse=True)

    assert len(forward_rows) == len(reverse_rows)
    for f_row, r_row in zip(forward_rows, reverse_rows):
        assert f_row.index == r_row.index
        assert f_row.uci == r_row.uci
        assert f_row.fen_before == r_row.fen_before
        assert f_row.fen_after == r_row.fen_after
        assert f_row.score_cp == r_row.score_cp
