"""
score_converter.py

This module provides utility functions for converting raw engine evaluations
(chess.engine.Score) into internal representations such as centipawn scores and
win/draw/loss probabilities.
"""

from typing import Optional, Dict
import chess
import chess.engine


class ScoreConverter:
    @staticmethod
    def score_to_centipawns(
        score: chess.engine.Score, stm: chess.Color
    ) -> Optional[int]:
        """
        Converts an engine score to a centipawn value. If the score represents a mate,
        a large constant is returned (positive for mate in favor of the side to move,
        negative otherwise).

        :param score: The engine's evaluation score.
        :param stm: The side to move (chess.WHITE or chess.BLACK).
        :return: An integer representing the centipawn score, or None if the score is not convertible.
        """
        try:
            if score.is_mate():
                mate_in = score.mate()
                return 10000 if mate_in and mate_in > 0 else -10000
            else:
                cp = score.white().score()
                return cp if stm == chess.WHITE else -cp
        except Exception as e:
            print(f"Error converting score: {e}")
            return None

    @staticmethod
    def score_to_wdl(score: chess.engine.Score, stm: chess.Color) -> Dict[str, float]:
        """
        Converts a given engine score to a win/draw/loss (WDL) probability distribution.
        This is a simplified example. In a production system, you might use a more refined
        model or heuristic based on the centipawn score and empirical data.

        :param score: The engine's evaluation score.
        :param stm: The side to move (chess.WHITE or chess.BLACK).
        :return: A dictionary with keys "win", "draw", and "loss" mapping to probabilities.
        """
        default_wdl = {"win": 0.33, "draw": 0.34, "loss": 0.33}

        cp = ScoreConverter.score_to_centipawns(score, stm)
        if cp is None:
            return default_wdl
        
        swing = cp / 1000.0 
        if stm == chess.WHITE:
            win = 0.5 + swing
            loss = 0.5 - swing
        else:
            win = 0.5 - swing
            loss = 0.5 + swing

        win = max(0.0, min(1.0, win))
        loss = max(0.0, min(1.0, loss))
        draw = max(0.0, 1.0 - win - loss)

        return {"win": win, "draw": draw, "loss": loss}

    @staticmethod
    def mate_score_to_string(score: chess.engine.Score) -> str:
        """
        Returns a human-readable string for mate scores or centipawn scores.

        :param score: The engine's evaluation score.
        :return: A string describing the score.
        """
        if score.is_mate():
            mate_in = score.mate()
            if mate_in is None:
                return "Mate"
            return f"Mate in {abs(mate_in)}"
        else:
            cp = score.white().score()
            return f"{cp} cp"
