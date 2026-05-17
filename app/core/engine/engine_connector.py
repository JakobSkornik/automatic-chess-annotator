"""
engine_connector.py

This module provides the EngineConnector class, a thin wrapper around a UCI chess engine
(e.g., Stockfish). It abstracts engine initialization, analysis calls, and shutdown procedures.
"""

import atexit
import chess.engine
import os
import platform
from typing import Optional

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
            engine.configure({"Hash": self.hash_size, "Threads": self.threads, "Use NNUE": self.use_nnue })
            return engine
        except Exception as e:
            raise RuntimeError(f"Failed to start engine at {self.engine_path}: {e}")

    def analyse(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        time_limit: Optional[float] = None,
        multiPv: Optional[int] = 1,
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

    def trace(self) -> dict:
        """
        Retrieves a trace of the engine's search process for the given board position.

        :param board: The chess.Board object representing the position to trace.
        :return: A dictionary with the engine's search trace.
        """
        try:
            trace = self.engine.trace()
            return trace
        except Exception as e:
            raise RuntimeError(f"Engine trace failed: {e}")

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

_global_engine_connector: Optional[EngineConnector] = None


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
