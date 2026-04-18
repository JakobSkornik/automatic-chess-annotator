"""Tantivy BM25 retriever for Bolčič-style positional tokens (embedded index, no external DB)."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

import chess
import tantivy
from tantivy import Occur, Query

from app.core.commentary.features.positional_tokens import encode_position_strings_only
from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever

logger = logging.getLogger(__name__)


class _NoopRetriever(RAGRetriever):
    """No examples when RAG_BM25_PATH is unset or invalid."""

    async def retrieve(self, query: RAGQuery, top_k: int = 2) -> List[RAGResult]:
        return []


def get_default_retriever() -> RAGRetriever:
    raw = os.environ.get("RAG_BM25_PATH")
    if not raw:
        logger.warning("RAG: RAG_BM25_PATH not set; commentary will run without examples")
        return _NoopRetriever()
    path = os.path.abspath(os.path.expanduser(raw.strip()))
    if not os.path.isdir(path):
        logger.warning("RAG: RAG_BM25_PATH=%s not a directory; using noop", raw)
        return _NoopRetriever()
    return TantivyPositionalRetriever(path)


_FIELD_BOOSTS = [
    ("static_attributes", 1.0),
    ("pawn_structure", 1.0),
    ("center", 1.0),
    ("dynamic_general", 2.0),
    ("dynamic_solution", 2.0),
    ("king_placement", 1.2),
    ("imbalance_signature", 1.2),
]


def _material_tuple(board: chess.Board) -> Tuple[int, ...]:
    return tuple(
        len(board.pieces(pt, c))
        for c in (chess.WHITE, chess.BLACK)
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _material_match(bq: chess.Board, bd: chess.Board) -> float:
    return 1.0 if _material_tuple(bq) == _material_tuple(bd) else 0.0


def _king_side_bucket(f: int) -> int:
    if f <= 3:
        return 0
    if f >= 6:
        return 2
    return 1


def _king_placement_match(bq: chess.Board, bd: chess.Board) -> float:
    wkq, bkq = bq.king(chess.WHITE), bq.king(chess.BLACK)
    wkd, bkd = bd.king(chess.WHITE), bd.king(chess.BLACK)
    if wkq is None or bkq is None or wkd is None or bkd is None:
        return 0.0
    if wkq == wkd and bkq == bkd:
        return 1.0
    wf_q, wf_d = chess.square_file(wkq), chess.square_file(wkd)
    bf_q, bf_d = chess.square_file(bkq), chess.square_file(bkd)
    if _king_side_bucket(wf_q) == _king_side_bucket(wf_d) and _king_side_bucket(bf_q) == _king_side_bucket(
        bf_d
    ):
        return 0.5
    return 0.0


def _pawn_skeleton_jaccard(bq: chess.Board, bd: chess.Board) -> float:
    wq = set(bq.pieces(chess.PAWN, chess.WHITE)) | set(bq.pieces(chess.PAWN, chess.BLACK))
    wd = set(bd.pieces(chess.PAWN, chess.WHITE)) | set(bd.pieces(chess.PAWN, chess.BLACK))
    union = len(wq | wd)
    if union == 0:
        return 1.0
    return len(wq & wd) / union


def _piece_map_hamming(bq: chess.Board, bd: chess.Board) -> float:
    same = 0
    for sq in chess.SQUARES:
        if bq.piece_at(sq) == bd.piece_at(sq):
            same += 1
    return same / 64.0


def _eco_match_score(qeco: str, deco: str, phase: Optional[str], ply: Optional[int]) -> float:
    if phase == "endgame" or (ply is not None and ply > 30):
        return 0.0
    if not qeco or not deco:
        return 0.0
    if qeco == deco:
        return 1.0
    if len(qeco) >= 2 and len(deco) >= 2 and qeco[:2] == deco[:2]:
        return 0.7
    if qeco[0] == deco[0]:
        return 0.4
    return 0.0


def _board_similarity(
    bq: chess.Board,
    bd: chess.Board,
    *,
    qeco: str,
    deco: str,
    phase: Optional[str],
    ply: Optional[int],
) -> float:
    mat = _material_match(bq, bd)
    king = _king_placement_match(bq, bd)
    skel = _pawn_skeleton_jaccard(bq, bd)
    ham = _piece_map_hamming(bq, bd)
    eco = _eco_match_score(qeco, deco, phase, ply)
    return 0.30 * mat + 0.20 * king + 0.20 * skel + 0.15 * ham + 0.15 * eco


class TantivyPositionalRetriever(RAGRetriever):
    """BM25 + board-similarity rerank; requires RAGQuery.fen and pv_san."""

    def __init__(self, index_path: str) -> None:
        self._path = os.path.abspath(os.path.expanduser(index_path))
        self._index: tantivy.Index | None = None

    def _get_index(self) -> tantivy.Index:
        if self._index is not None:
            return self._index
        if not os.path.isdir(self._path):
            raise FileNotFoundError(f"Tantivy index not found: {self._path}")
        self._index = tantivy.Index.open(self._path)
        return self._index

    async def retrieve(self, query: RAGQuery, top_k: int = 2) -> List[RAGResult]:
        if not query.fen or not query.pv_san:
            return []
        try:
            board = chess.Board(query.fen)
        except Exception:
            logger.warning("Tantivy RAG: invalid FEN")
            return []

        fields = encode_position_strings_only(board, list(query.pv_san))
        fields["player_color"] = "w" if board.turn == chess.WHITE else "b"

        try:
            index = self._get_index()
        except Exception as e:
            logger.warning("Tantivy RAG: cannot open index: %s", e)
            return []

        index.reload()
        searcher = index.searcher()

        subqueries: List[tuple] = []
        for fname, boost in _FIELD_BOOSTS:
            text = (fields.get(fname) or "").strip()
            if not text:
                continue
            try:
                q = index.parse_query(text, default_field_names=[fname])
            except Exception as e:
                logger.debug("Tantivy parse skip field %s: %s", fname, e)
                continue
            subqueries.append((Occur.Should, Query.boost_query(q, boost)))

        if not subqueries:
            return []

        try:
            q_pc = index.parse_query(fields["player_color"], default_field_names=["player_color"])
        except Exception:
            return []
        subqueries.append((Occur.Must, q_pc))

        bool_q = Query.boolean_query(subqueries)
        try:
            hits = searcher.search(bool_q, limit=max(20, top_k * 4)).hits
        except Exception as e:
            logger.warning("Tantivy RAG search failed: %s", e)
            return []

        qeco = (query.eco or "").strip()
        phase = query.phase
        ply = query.ply

        scored: List[Tuple[float, Any, float]] = []
        raw_scores: List[float] = []
        for score, addr in hits:
            try:
                doc = searcher.doc(addr)
            except Exception:
                continue
            raw_scores.append(float(score) if score is not None else 0.0)
            scored.append((float(score) if score is not None else 0.0, doc, addr))

        max_bm25 = max(raw_scores) if raw_scores else 1.0
        if max_bm25 <= 0:
            max_bm25 = 1.0

        finals: List[Tuple[float, Any]] = []
        for bm25, doc, _addr in scored:
            fen = (doc.get_first("fen") or "").strip()
            pv_san = (doc.get_first("pv_san") or "").strip()
            deco = (doc.get_first("eco") or "").strip()
            ann = (doc.get_first("annotation_text") or "").strip()
            plies_ann_raw = doc.get_first("plies_to_next_annotation")
            try:
                plies_ann = int(plies_ann_raw) if plies_ann_raw is not None else 0
            except Exception:
                plies_ann = 0
            try:
                bd = chess.Board(fen) if fen else None
            except Exception:
                bd = None
            bm25_norm = bm25 / max_bm25
            if bd is not None:
                bsim = _board_similarity(board, bd, qeco=qeco, deco=deco, phase=phase, ply=ply)
            else:
                bsim = 0.0
            final = 0.7 * bm25_norm + 0.3 * bsim
            ann_boost = 1.0 / (1 + max(plies_ann, 0))
            final *= ann_boost
            finals.append((final, (doc, pv_san, fen, deco, ann, plies_ann, bm25)))

        finals.sort(key=lambda x: x[0], reverse=True)

        out: List[RAGResult] = []
        seen_fen: set[str] = set()
        for final, payload in finals:
            doc, pv_san, fen, deco, ann, plies_ann, bm25 = payload
            if len(out) >= top_k:
                break
            source = (doc.get_first("source") or "").strip()
            if fen and fen in seen_fen:
                continue
            if fen:
                seen_fen.add(fen)
            if ann:
                prefix = f"(annotation +{plies_ann} plies) " if plies_ann > 0 else ""
                text_body = prefix + ann
            else:
                text_body = f"PV: {pv_san}"[:1200]
            out.append(
                RAGResult(
                    source=source or "lichess",
                    fen=fen or None,
                    annotation_text=text_body[:1200],
                    relevance_tags={
                        "eco": deco,
                        "pv": pv_san[:200] if pv_san else "",
                        "plies_to_next_annotation": str(plies_ann),
                    },
                    similarity_score=float(final),
                )
            )
            logger.debug(
                "Tantivy rerank final=%.4f bm25=%.4f source=%s fen=%.50s",
                final,
                bm25,
                source or "lichess",
                fen or "",
            )

        return out
