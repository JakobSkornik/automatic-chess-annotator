from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import chess
import chess.pgn

logger = logging.getLogger(__name__)


@dataclass
class OpeningInfo:
    code: str
    name: str
    variation: str | None = None


def mainline_uci_list(game: chess.pgn.Game) -> list[str]:
    """UCI plies in mainline order."""
    out: list[str] = []
    board = game.board()
    for m in game.mainline_moves():
        out.append(board.uci(m))
        board.push(m)
    return out


def _fen_lookup_variants(fen: str) -> list[str]:
    """Try exact FEN and common normalizations for ECO JSON keys."""
    parts = fen.split()
    seen: list[str] = []
    for candidate in (
        fen,
        " ".join(parts[:4]) + " 0 1" if len(parts) >= 4 else fen,
        " ".join(parts[:4]) + " - 0 1" if len(parts) >= 4 else fen,
    ):
        if candidate not in seen:
            seen.append(candidate)
    return seen


def parse_pgn_eco_tag(headers: chess.pgn.Headers) -> str | None:
    """
    Valid ECO codes are A00–E99 (letter + two digits), case-sensitive per PGN convention.
    Returns None and logs a warning if malformed.
    """
    raw = (headers.get("ECO") or "").strip().strip('"')
    if not raw:
        return None
    if len(raw) != 3 or raw[0] not in "ABCDEabcde" or not raw[1:].isdigit():
        logger.warning(
            "Invalid ECO PGN tag %r — ignoring; internal book will be used", raw
        )
        return None
    return raw.upper()


_ABSENT_OPENING_NAMES = frozenset(
    {"", "?", "unknown", "custom", "opening unknown", "general", "irregular"}
)


def is_absent_opening_header(name: str | None) -> bool:
    if not name:
        return True
    s = name.strip()
    if not s:
        return True
    return s.lower() in _ABSENT_OPENING_NAMES


def merge_opening_with_headers(
    detected: OpeningInfo | None,
    header_opening: str,
    header_eco: str | None,
) -> OpeningInfo | None:
    """
    Internal book is source of truth for ECO code when detection succeeds.
    Opening *name* may come from the PGN when it clearly matches the detected family.
    """
    ho = header_opening.strip() if header_opening else ""
    if detected:
        code = detected.code
        name = detected.name
        var = detected.variation
        if not is_absent_opening_header(ho):
            eco_match = header_eco and header_eco == code
            prefix_match = (
                header_eco and len(header_eco) >= 2 and code.startswith(header_eco[:2])
            )
            name_match = ho.lower() in name.lower() or name.lower() in ho.lower()
            if eco_match or prefix_match or name_match:
                name = ho
        return OpeningInfo(code=code, name=name, variation=var)
    if header_eco and not is_absent_opening_header(ho):
        return OpeningInfo(code=header_eco, name=ho)
    if header_eco:
        return OpeningInfo(code=header_eco, name=ho or "Unknown")
    return None


def detect_opening(
    game: chess.pgn.Game, book: ECOBook | None = None
) -> tuple[OpeningInfo | None, int]:
    """
    Longest-prefix ECO match for the full mainline (for game-level metadata).
    Returns (OpeningInfo | None, matched_ply_count).
    """
    b = book or ECOBook()
    uci_moves = mainline_uci_list(game)
    return b.match(uci_moves)


class ECOBook:
    """ECO lookup: longest UCI-prefix match, then FEN (transposition) fallback."""

    def __init__(self) -> None:
        self._by_uci: dict[str, OpeningInfo] = {}
        self._by_fen: dict[str, OpeningInfo] = {}
        self._load()

    def _load(self) -> None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        for filename in os.listdir(data_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(data_dir, filename)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for fen_key, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                eco = entry.get("eco")
                name = entry.get("name")
                if not eco or not name:
                    continue
                info = OpeningInfo(
                    code=str(eco),
                    name=str(name),
                    variation=entry.get("v"),
                )
                self._by_fen[fen_key] = info
                moves_str = entry.get("moves", "")
                board = chess.Board()
                uci_sequence: list[str] = []
                try:
                    for san_move in moves_str.split():
                        if "." in san_move:
                            continue
                        move = board.parse_san(san_move)
                        uci_sequence.append(move.uci())
                        board.push(move)
                    if uci_sequence:
                        key = " ".join(uci_sequence)
                        self._by_uci[key] = info
                except Exception:
                    continue
        logger.info(
            "ECO book: %s UCI prefixes, %s FEN keys",
            len(self._by_uci),
            len(self._by_fen),
        )

    def match(self, uci_moves: list[str]) -> tuple[OpeningInfo | None, int]:
        """Longest matching opening for the given UCI prefix; returns (info, ply_count_matched)."""
        if not uci_moves:
            return None, 0
        for n in range(len(uci_moves), 0, -1):
            prefix = uci_moves[:n]
            seq = " ".join(prefix)
            hit = self._by_uci.get(seq)
            if hit:
                return hit, n
            board = chess.Board()
            try:
                for u in prefix:
                    board.push_uci(u)
            except Exception:
                continue
            fen = board.fen()
            for variant in _fen_lookup_variants(fen):
                hit = self._by_fen.get(variant)
                if hit:
                    return hit, n
        return None, 0
