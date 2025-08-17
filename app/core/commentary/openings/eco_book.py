from __future__ import annotations
import os
import json
import chess
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

@dataclass
class OpeningInfo:
    code: str
    name: str
    variation: Optional[str] = None

class ECOBook:
    """A comprehensive ECO lookup service using a JSON database."""

    def __init__(self) -> None:
        self._eco_data: Dict[str, OpeningInfo] = self._load_eco_data()

    def _load_eco_data(self) -> Dict[str, OpeningInfo]:
        """Loads ECO data from JSON files and builds a lookup dictionary."""
        eco_dict: Dict[str, OpeningInfo] = {}
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir, filename), "r") as f:
                    data = json.load(f)
                    for fen, entry in data.items():
                        moves_str = entry.get("moves", "")
                        # The moves are in SAN format with move numbers, like "1. e4 e5"
                        # We need to convert this to a UCI sequence
                        board = chess.Board()
                        uci_sequence = []
                        try:
                            for san_move in moves_str.split():
                                if "." not in san_move:
                                    move = board.parse_san(san_move)
                                    uci_sequence.append(move.uci())
                                    board.push(move)
                            eco_dict[" ".join(uci_sequence)] = OpeningInfo(
                                code=entry["eco"],
                                name=entry["name"],
                                variation=entry.get("v"),
                            )
                        except Exception:
                            continue
        logger.info(f"Loaded {len(eco_dict)} opening positions from ECO data.")
        return eco_dict

    def _build_uci_sequence(self, board: chess.Board) -> str:
        """Builds a UCI move sequence from a board state."""
        moves = []
        temp_board = board.copy()
        while temp_board.move_stack:
            move = temp_board.pop()
            moves.insert(0, move.uci())
        return " ".join(moves)

    def match(self, uci_moves: List[str]) -> Optional[OpeningInfo]:
        """Return the longest matching opening for the given UCI move list."""
        if not uci_moves:
            return None

        # Create a board and play the moves to get the correct fen
        board = chess.Board()
        for move in uci_moves:
            board.push_uci(move)
        
        # Build the uci sequence and look it up
        uci_sequence = self._build_uci_sequence(board)
        logger.info(f"Looking up UCI sequence: '{uci_sequence}'")
        result = self._eco_data.get(uci_sequence)
        if result:
            logger.info(f"Found opening: {result.name}")
        else:
            logger.info("No opening found for this sequence.")
        return result


