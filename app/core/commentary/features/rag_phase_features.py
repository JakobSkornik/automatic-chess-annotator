"""Shared phase, opening, endgame, and middlegame feature extraction for RAG corpus + retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess

from app.core.commentary.openings.eco_book import ECOBook, OpeningInfo, mainline_uci_list

# Align with engine: early game through full move 10; endgame by minor+major count.
_EARLY_FULLMOVE_CUTOFF = 10
_MINOR_MAJOR_END_THRESHOLD = 7  # < this count => "end" (endgame) when past opening window

# Corpus schema version (independent of positional ENCODER_VERSION)
CORPUS_VERSION = "2"

RagPhase = str  # "opening" | "middlegame" | "endgame"


def _minor_major_count(board: chess.Board) -> int:
    s = 0
    for c in (chess.WHITE, chess.BLACK):
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            s += len(board.pieces(pt, c))
    return s


def _engine_raw_phase(board: chess.Board) -> str:
    """Mirror app.core.engine.analysis_retriever._determine_game_phase values."""
    if board.fullmove_number <= _EARLY_FULLMOVE_CUTOFF:
        return "early"
    if _minor_major_count(board) < _MINOR_MAJOR_END_THRESHOLD:
        return "end"
    return "mid"


def map_engine_phase_to_rag(raw: str) -> RagPhase:
    if raw == "early":
        return "opening"
    if raw == "end":
        return "endgame"
    return "middlegame"


def classify_rag_phase(board: chess.Board) -> RagPhase:
    """
    Classify the position into opening / middlegame / endgame.
    Uses fullmove number and material (same heuristics as live engine analysis).
    """
    return map_engine_phase_to_rag(_engine_raw_phase(board))


def opening_ply_bucket(ply: int) -> str:
    """
    Half-move index buckets for opening retrieval (1-based ply after a move, as in the annotator).
    """
    p = int(max(0, ply))
    if p <= 6:
        return "1-6"
    if p <= 12:
        return "7-12"
    if p <= 20:
        return "13-20"
    return "21+"


@dataclass
class OpeningTags:
    opening_eco: str
    opening_name: str
    opening_matched_ply: int
    opening_prefix: str
    opening_ply_bucket: str

    def as_dict(self) -> dict:
        return {
            "opening_eco": self.opening_eco,
            "opening_name": self.opening_name,
            "opening_matched_ply": str(self.opening_matched_ply),
            "opening_prefix": self.opening_prefix,
            "opening_ply_bucket": self.opening_ply_bucket,
        }


def _norm_eco(s: str) -> str:
    t = re.sub(r"[^A-Ea-e0-9/]", "", (s or "").strip())
    if len(t) >= 3 and t[0].upper() in "ABCDE" and t[1:3].isdigit():
        return t[:3].upper() + t[3:]
    return t[:16]


def _prefix2(eco: str) -> str:
    e = (eco or "").strip().upper()
    if len(e) >= 2:
        return e[:2]
    return e


def opening_info_from_uci_list(
    uci_prefix: List[str], eco_book: Optional[ECOBook] = None, header_eco: str = ""
) -> Tuple[Optional[OpeningInfo], int, str]:
    """
    Longest-prefix ECO match for the UCI list up to the current position.
    Returns (info, matched_ply_count, effective_eco_string).
    """
    b = eco_book or ECOBook()
    info, n = b.match(uci_prefix)
    he = _norm_eco(header_eco)
    code = (info.code if info else "") or he
    name = (info.name if info else "")
    if info is None and he and len(he) >= 3 and he[0].upper() in "ABCDE" and he[1:3].isdigit():
        code3 = he[:3].upper()
        info = OpeningInfo(code=code3, name=name or "HeaderECO", variation=None)
        n = 0
        code = code3
    return info, n, code


def build_opening_tags(
    uci_prefix: List[str], ply: int, eco_book: Optional[ECOBook] = None, header_eco: str = ""
) -> OpeningTags:
    info, matched_ply, _ = opening_info_from_uci_list(uci_prefix, eco_book, header_eco)
    code = (info.code if info else "") or _norm_eco(header_eco) or ""
    name = (info.name if info else "")[:120]
    prefix = _prefix2(code) if code else ( _prefix2(header_eco) if header_eco else "" )
    return OpeningTags(
        opening_eco=code[:16],
        opening_name=name,
        opening_matched_ply=matched_ply,
        opening_prefix=prefix,
        opening_ply_bucket=opening_ply_bucket(ply),
    )


def material_tuple(board: chess.Board) -> Tuple[int, ...]:
    return tuple(
        len(board.pieces(pt, c))
        for c in (chess.WHITE, chess.BLACK)
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _side_material_str(board: chess.Board, color: chess.Color) -> str:
    order = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
    parts: List[str] = ["K"]
    for pt in order:
        n = len(board.pieces(pt, color))
        sym = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}[pt]
        parts.extend([sym] * n)
    return "".join(parts)


def material_signature(board: chess.Board) -> str:
    w = _side_material_str(board, chess.WHITE)
    b = _side_material_str(board, chess.BLACK)
    return f"{w}v{b}"


def material_bucket(board: chess.Board) -> str:
    mm = _minor_major_count(board)
    wq, bq = len(board.pieces(chess.QUEEN, chess.WHITE)), len(board.pieces(chess.QUEEN, chess.BLACK))
    if mm < _MINOR_MAJOR_END_THRESHOLD:
        return "endgame_sharp"
    if wq == 0 and bq == 0:
        return "queenless"
    if wq + bq == 1:
        return "single_queen"
    return "heavy_pieces"


def _square_is_light(sq: int) -> bool:
    """True if the square is a light square (python-chess has no chess.square_color)."""
    return bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[sq])


def endgame_signature(board: chess.Board) -> str:
    """
    Space-separated tags for endgame RAG: material class, simple patterns, pawn wing.
    """
    tags: List[str] = []
    tags.append(f"mat_sig:{material_signature(board)}")
    wq, bq = len(board.pieces(chess.QUEEN, chess.WHITE)), len(board.pieces(chess.QUEEN, chess.BLACK))
    wr, br = len(board.pieces(chess.ROOK, chess.WHITE)), len(board.pieces(chess.ROOK, chess.BLACK))
    wb, bb = len(board.pieces(chess.BISHOP, chess.WHITE)), len(board.pieces(chess.BISHOP, chess.BLACK))
    wn, bn = len(board.pieces(chess.KNIGHT, chess.WHITE)), len(board.pieces(chess.KNIGHT, chess.BLACK))
    if wq == 0 and bq == 0 and wr + br >= 1:
        tags.append("pattern:rook_ending")
    if wq == 0 and bq == 0 and wn + bn >= 1 and wr + br == 0:
        tags.append("pattern:minor_ending")
    if wq == 0 and bq == 0 and wb == 1 and bb == 1:
        wsq = list(board.pieces(chess.BISHOP, chess.WHITE))
        bsq = list(board.pieces(chess.BISHOP, chess.BLACK))
        if wsq and bsq and _square_is_light(wsq[0]) != _square_is_light(bsq[0]):
            tags.append("pattern:opposite_bishops")
        else:
            tags.append("pattern:same_color_bishops")
    if wq + bq == 0 and len(list(board.pieces(chess.PAWN, chess.WHITE))) + len(
        list(board.pieces(chess.PAWN, chess.BLACK))
    ) <= 4:
        tags.append("pattern:pawn_light")

    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    if wk is not None and bk is not None:
        kd = chess.square_distance(wk, bk)
        if kd <= 2:
            tags.append(f"king_prox:touching")
        elif kd <= 4:
            tags.append(f"king_prox:close")
        else:
            tags.append(f"king_prox:far")

    # Pawn on kingside/queenside majority (coarse)
    wpf = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)]
    bpf = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)]
    if wpf and max(wpf) >= 4:
        tags.append("wing:wk_pawn_kingside")
    if bpf and min(bpf) <= 3:
        tags.append("wing:bk_pawn_queenside")
    return " ".join(tags)


def _file_openness(board: chess.Board, f: int) -> str:
    pawns = 0
    for r in range(8):
        p = board.piece_at(chess.square(f, r))
        if p and p.piece_type == chess.PAWN:
            pawns += 1
    if pawns == 0:
        return "open"
    if pawns == 1:
        return "semiopen"
    return "closed"


def middlegame_strategic_tags(board: chess.Board) -> str:
    """
    Extra lexical tokens (whitespace) for middlegame BM25, beyond positional encoder fields.
    """
    parts: List[str] = []
    wc, bc = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(
        chess.WHITE
    )
    wc2, bc2 = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(
        chess.BLACK
    )
    if not (wc or wc2 or bc or bc2):
        parts.append("castle:none_available")
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    if wk is not None and bk is not None:
        wf, bf = chess.square_file(wk), chess.square_file(bk)
        if wf <= 3 and bf >= 6 or (wf >= 6 and bf <= 3):
            parts.append("castling_pattern:opposite_side")
        elif abs(wf - bf) <= 2 and abs(chess.square_rank(wk) - chess.square_rank(bk)) >= 3:
            parts.append("castling_pattern:same_side")
    for f in (2, 3, 4, 5, 6):
        parts.append(f"file{chess.FILE_NAMES[f]}:{_file_openness(board, f)}")

    for c, tag in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        if len(board.pieces(chess.BISHOP, c)) >= 2:
            parts.append(f"bishop_pair_{tag}")
    for c, tag in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        n = len(board.pieces(chess.QUEEN, c))
        if n:
            parts.append(f"queen_count_{tag}:{n}")
    return " ".join(parts)


def pawn_structure_fingerprint(board: chess.Board) -> str:
    """Short hashable fingerprint from pawn files (per side)."""
    wf = [0] * 8
    bf = [0] * 8
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        wf[chess.square_file(sq)] += 1
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        bf[chess.square_file(sq)] += 1
    w_s = "".join(str(min(x, 3)) for x in wf)
    b_s = "".join(str(min(x, 3)) for x in bf)
    return f"wf:{w_s}_bf:{b_s}"


def uci_list_from_pgn_start(game: chess.pgn.Game, plies: int) -> List[str]:
    """UCI plies of the first `plies` mainline half-moves from the start position."""
    return mainline_uci_list(game)[: int(max(0, plies))]
