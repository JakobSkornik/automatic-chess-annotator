from __future__ import annotations
import logging
from typing import List, Optional, Tuple

from app.models.Move import Move

logger = logging.getLogger(__name__)

class KeyMomentDetector:
    """
    Detects key moments in a chess game based on move analysis.
    """

    def detect(
        self,
        current_move: Move,
        previous_move: Optional[Move],
        pvs_for_move: Optional[List[List[Move]]],
    ) -> Optional[str]:
        """
        Determines if a move is a key moment and returns the type of moment.
        """
        if not previous_move or current_move.score is None or previous_move.score is None:
            return None

        is_white_move = current_move.depth % 2 == 1
        score_change = current_move.score - previous_move.score

        log_msg = (
            f"Move {current_move.depth} ({'White' if is_white_move else 'Black'}): "
            f"Prev Score: {previous_move.score}, Curr Score: {current_move.score}, "
            f"Raw Score Change: {score_change}"
        )

        # Adjust for black's perspective
        if not is_white_move:
            score_change = -score_change

        log_msg += f", Perspective Score Change: {score_change}"
        logger.info(log_msg)

        # Blunder detection
        if score_change <= -200:
            logger.info("Blunder detected.")
            return "blunder"

        # Mistake detection
        if score_change <= -100:
            logger.info("Mistake detected.")
            return "mistake"

        # Missed opportunity detection
        if pvs_for_move and pvs_for_move[0]:
            best_move_in_pv = pvs_for_move[0][0]
            if best_move_in_pv and best_move_in_pv.score is not None:
                best_move_score = best_move_in_pv.score
                opportunity_diff = best_move_score - current_move.score
                if not is_white_move:
                    opportunity_diff = -opportunity_diff
                
                logger.info(
                    f"Missed opportunity check: Best move score: {best_move_score}, "
                    f"Current move score: {current_move.score}, "
                    f"Opportunity diff: {opportunity_diff}"
                )
                
                if opportunity_diff >= 200:
                    logger.info("Missed opportunity detected.")
                    return "missed_opportunity"

        return None
