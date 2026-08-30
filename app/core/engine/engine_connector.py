"""
engine_connector.py

This module provides the EngineConnector class, a thin wrapper around a UCI chess engine
(e.g., Stockfish). It abstracts engine initialization, analysis calls, and shutdown procedures.
"""

import atexit
import logging
import os
import platform

import chess.engine

logger = logging.getLogger(__name__)

MATE_SCORE = 1000000
MULTIPV = 3


class EngineConnector:
    """
    A connector class to manage interactions with a UCI chess engine.
    """

    def __init__(self, engine_path: str):
        """
        Initializes the EngineConnector by launching the UCI engine.

        :param engine_path: The filesystem path to the engine executable.
        """
        self.engine_path = engine_path
        self.hash_size = 32
        self.threads = 4
        self.use_nnue = False
        self.engine = self._start_engine()

    def _start_engine(self) -> chess.engine.SimpleEngine:
        """
        Launches the chess engine using the UCI protocol.

        :return: An instance of SimpleEngine.
        """
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            engine.configure(
                {
                    "Hash": self.hash_size,
                    "Threads": self.threads,
                    "Use NNUE": self.use_nnue,
                }
            )
            return engine
        except Exception as e:
            raise RuntimeError(
                f"Failed to start engine at {self.engine_path}: {e}"
            ) from e

    def analyse(
        self,
        board: chess.Board,
        depth: int | None = None,
        time_limit: float | None = None,
        multiPv: int | None = 1,
    ) -> dict:
        """
        Analyzes a given board position using the chess engine.

        Either `depth` or `time_limit` (or both) must be provided. If time_limit is not provided,
        the default_time_limit set during initialization will be used.

        :param board: The chess.Board object representing the position to analyze.
        :param depth: (Optional) The search depth for the analysis.
        :param time_limit: (Optional) The time limit (in seconds) for the analysis.
        :return: A dictionary with the engine's analysis result.
        """
        if depth is None and time_limit is None:
            # Sensible minimal default time limit
            time_limit = 0.1

        if not multiPv:
            multiPv = MULTIPV

        limit = chess.engine.Limit(depth=depth, time=time_limit)
        result = self.engine.analyse(board, limit, multipv=multiPv)
        return result

    def analyse_with_depth_snapshots(
        self,
        board: chess.Board,
        depth: int,
        snapshot_depths: tuple[int, ...] = (),
        multiPv: int = 1,
    ) -> tuple[list, dict[int, tuple[int | None, str | None]]]:
        """One streamed search to ``depth`` that records (cp, pv1-uci) at each
        depth in ``snapshot_depths`` along the way — instead of re-searching
        once per depth. Returns (final info list, {depth: (cp, pv1_uci)}).

        Falls back to a plain blocking search (snapshots empty) if streaming
        fails, so callers always get a valid final result."""
        snapshots: dict[int, tuple[int | None, str | None]] = {}
        wanted = {d for d in snapshot_depths}

        def _snapshot(inf: dict) -> None:
            d = inf.get("depth")
            if d not in wanted or d in snapshots:
                return
            sc = inf.get("score")
            pv = inf.get("pv") or []
            cp = None
            if sc is not None:
                try:
                    cp = int(sc.white().score(mate_score=MATE_SCORE))
                except Exception:
                    cp = None
            snapshots[d] = (cp, pv[0].uci() if pv else None)

        try:
            result = self.engine.analysis(board, chess.engine.Limit(depth=depth), multipv=multiPv)
            while True:
                try:
                    info = result.next()
                except chess.engine.AnalysisComplete:
                    break
                if info is None:
                    break
                if isinstance(info, dict):
                    _snapshot(info)
            # The authoritative final state of the search is the aggregated
            # multipv view on the result object.
            agg = result.multipv
            return (agg if isinstance(agg, list) else [agg] if agg else []), snapshots
        except Exception:
            return self.analyse(board, depth=depth, multiPv=multiPv), snapshots

    def evaluate_position(self, fen: str, *, depth: int) -> int | None:
        """One-off White-POV cp eval of a position at ``depth``."""
        try:
            board = chess.Board(fen)
            info = self.analyse(board, depth=depth, multiPv=1)
            if isinstance(info, list) and info:
                info = info[0]
            if not isinstance(info, dict):
                return None
            sc = info.get("score")
            if sc is None:
                return None
            return int(sc.white().score(mate_score=MATE_SCORE))
        except Exception as e:
            logger.warning("Seed eval failed for FEN %s: %s", fen, e)
            return None

    def close(self) -> None:
        """
        Shuts down the chess engine.
        """
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception as e:
                # Log the error if using a logging framework
                print(f"Error closing the engine: {e}")
            finally:
                self.engine = None

    def __enter__(self):
        """
        Support for context management.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensures the engine is properly closed on exiting the context.
        """
        self.close()


stockfish_executable = (
    "stockfish.exe" if platform.system() == "Windows" else "stockfish"
)
# Navigate from app/core/engine/ to repo root
stockfish_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", stockfish_executable)
)

_global_engine_connector: EngineConnector | None = None


def _shutdown_global_engine() -> None:
    global _global_engine_connector
    if _global_engine_connector is not None:
        _global_engine_connector.close()
        _global_engine_connector = None


atexit.register(_shutdown_global_engine)


def get_global_engine_connector() -> EngineConnector:
    """Lazily start Stockfish so importing this module (e.g. in tests) does not spawn a process."""
    global _global_engine_connector
    if _global_engine_connector is None:
        _global_engine_connector = EngineConnector(stockfish_path)
    return _global_engine_connector
