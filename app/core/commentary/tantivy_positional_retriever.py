"""Tantivy BM25 retriever for Bolčič-style positional tokens (embedded index, no external DB)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import chess
import tantivy
from tantivy import Occur, Query

from app.core.commentary.features.positional_tokens import encode_position_strings_only
from app.core.commentary.features.rag_phase_features import (
    classify_rag_phase,
    endgame_signature,
    middlegame_strategic_tags,
    opening_ply_bucket,
    pawn_structure_fingerprint,
)
from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever

logger = logging.getLogger(__name__)


def _read_corpus_version(index_path: str) -> str:
    meta = Path(index_path) / "metadata.json"
    if not meta.is_file():
        return "1"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return str(data.get("corpus_version", "1"))
    except Exception:
        return "1"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _min_score_for_phase(phase: str) -> float:
    p = (phase or "middlegame").lower()
    if p == "opening":
        return _env_float("RAG_MIN_SCORE_OPENING", 0.45)
    if p == "endgame":
        return _env_float("RAG_MIN_SCORE_ENDGAME", 0.40)
    return _env_float("RAG_MIN_SCORE_MIDDLEGAME", 0.42)


def _jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set((a or "").split()), set((b or "").split())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _field_boosts_for_phase(phase: str) -> List[Tuple[str, float]]:
    """Field boosts: phase-specific emphasis (quiet middlegame weights structure)."""
    p = (phase or "middlegame").lower()
    if p == "opening":
        return [
            ("static_attributes", 1.1),
            ("pawn_structure", 1.1),
            ("center", 1.25),
            ("dynamic_general", 1.1),
            ("dynamic_solution", 1.15),
            ("king_placement", 1.0),
            ("imbalance_signature", 1.0),
            ("strategic_tags", 0.0),
            ("endgame_signature", 0.0),
        ]
    if p == "endgame":
        return [
            ("static_attributes", 0.9),
            ("pawn_structure", 1.2),
            ("center", 1.0),
            ("dynamic_general", 0.85),
            ("dynamic_solution", 0.85),
            ("king_placement", 1.15),
            ("imbalance_signature", 1.1),
            ("strategic_tags", 0.0),
            ("endgame_signature", 2.4),
        ]
    return [
        ("static_attributes", 1.0),
        ("pawn_structure", 1.05),
        ("center", 1.05),
        ("dynamic_general", 1.85),
        ("dynamic_solution", 1.85),
        ("king_placement", 1.2),
        ("imbalance_signature", 1.2),
        ("strategic_tags", 1.25),
        ("endgame_signature", 0.0),
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


def _eco_match_score(qeco: str, deco: str, *, zero_eco: bool) -> float:
    if zero_eco or not qeco or not deco:
        return 0.0
    if qeco == deco:
        return 1.0
    if len(qeco) >= 2 and len(deco) >= 2 and qeco[:2] == deco[:2]:
        return 0.7
    if qeco[0] == deco[0]:
        return 0.4
    return 0.0


def _board_similarity_middlegame(
    bq: chess.Board,
    bd: chess.Board,
    qeco: str,
    deco: str,
    *,
    v2: bool,
    query_phase: Optional[str],
    ply: Optional[int],
) -> float:
    mat = _material_match(bq, bd)
    king = _king_placement_match(bq, bd)
    skel = _pawn_skeleton_jaccard(bq, bd)
    ham = _piece_map_hamming(bq, bd)
    if v2:
        zero = False
    else:
        qp = (query_phase or "").lower()
        zero = qp == "endgame" or (ply is not None and ply > 30)
    eco = _eco_match_score(qeco, deco, zero_eco=zero)
    return 0.30 * mat + 0.20 * king + 0.20 * skel + 0.15 * ham + 0.15 * eco


def _board_similarity_opening(
    bq: chess.Board, bd: chess.Board, qeco: str, deco: str, doc_opening_eco: str
) -> float:
    mat = _material_match(bq, bd)
    king = _king_placement_match(bq, bd)
    skel = _pawn_skeleton_jaccard(bq, bd)
    ham = _piece_map_hamming(bq, bd)
    oeco = (doc_opening_eco or deco or "").strip()
    eco1 = _eco_match_score(qeco, oeco, zero_eco=False)
    eco2 = _eco_match_score(qeco, (deco or "").strip(), zero_eco=False)
    eco = max(eco1, eco2)
    return 0.15 * mat + 0.12 * king + 0.16 * skel + 0.10 * ham + 0.47 * eco


def _board_similarity_endgame(
    bq: chess.Board, bd: chess.Board, q_sig: str, d_sig: str
) -> float:
    mat = _material_match(bq, bd)
    king = _king_placement_match(bq, bd)
    skel = _pawn_skeleton_jaccard(bq, bd)
    ham = _piece_map_hamming(bq, bd)
    sig = _jaccard_tokens(q_sig, d_sig)
    return 0.28 * mat + 0.20 * king + 0.22 * skel + 0.10 * ham + 0.20 * sig


def _fpc_match(qfp: str, dfp: str) -> float:
    return 1.0 if (qfp and dfp and qfp == dfp) else 0.0


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


class TantivyPositionalRetriever(RAGRetriever):
    """BM25 + board-similarity rerank; v2: phase filters + per-phase scoring + min score threshold."""

    def __init__(self, index_path: str) -> None:
        self._path = os.path.abspath(os.path.expanduser(index_path))
        self._index: tantivy.Index | None = None
        self._corpus_version: Optional[str] = None

    def _get_index(self) -> tantivy.Index:
        if self._index is not None:
            return self._index
        if not os.path.isdir(self._path):
            raise FileNotFoundError(f"Tantivy index not found: {self._path}")
        self._index = tantivy.Index.open(self._path)
        return self._index

    def _v2_corpus(self) -> bool:
        if self._corpus_version is None:
            self._corpus_version = _read_corpus_version(self._path)
        return self._corpus_version == "2"

    async def retrieve(self, query: RAGQuery, top_k: int = 2) -> List[RAGResult]:
        if not query.fen or not query.pv_san:
            return []
        try:
            board = chess.Board(query.fen)
        except Exception:
            logger.warning("Tantivy RAG: invalid FEN")
            return []

        rag_phase = classify_rag_phase(board)
        qeco = (query.eco or "").strip()
        eco_prefix = (query.eco_prefix or "").strip() or (qeco[:2] if len(qeco) >= 2 else "")
        min_score = _min_score_for_phase(rag_phase)
        v2 = self._v2_corpus()

        # Query-side endgame / middlegame text fields
        end_q = endgame_signature(board) if rag_phase == "endgame" else ""
        strat_q = middlegame_strategic_tags(board) if rag_phase == "middlegame" else ""
        fpc_q = pawn_structure_fingerprint(board)
        op_bucket = opening_ply_bucket((query.ply or 0) if query.ply is not None else 0)

        base_enc = encode_position_strings_only(board, list(query.pv_san))
        base_enc["player_color"] = "w" if board.turn == chess.WHITE else "b"

        try:
            index = self._get_index()
        except Exception as e:
            logger.warning("Tantivy RAG: cannot open index: %s", e)
            return []

        index.reload()
        searcher = index.searcher()
        subqueries: List[tuple] = []
        for fname, boost in _field_boosts_for_phase(rag_phase):
            if boost <= 0:
                continue
            text = (base_enc.get(fname) or "").strip()
            if not text and fname in ("strategic_tags", "endgame_signature"):
                if fname == "strategic_tags" and strat_q:
                    text = strat_q
                elif fname == "endgame_signature" and end_q:
                    text = end_q
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
            q_pc = index.parse_query(base_enc["player_color"], default_field_names=["player_color"])
        except Exception:
            return []
        subqueries.append((Occur.Must, q_pc))

        if v2:
            try:
                q_ph = index.parse_query(rag_phase, default_field_names=["rag_phase"])
                subqueries.append((Occur.Must, q_ph))
            except Exception as e:
                logger.warning("Tantivy RAG: rag_phase filter failed; index may be v1. Error: %s", e)
                v2 = False

        if v2 and rag_phase == "opening" and (qeco or eco_prefix):
            try:
                if qeco and len(qeco) >= 3:
                    q_e = index.parse_query(qeco[:16], default_field_names=["opening_eco"])
                    subqueries.append((Occur.Should, Query.boost_query(q_e, 2.2)))
                if eco_prefix:
                    q_p = index.parse_query(eco_prefix, default_field_names=["opening_prefix"])
                    subqueries.append((Occur.Should, Query.boost_query(q_p, 1.4)))
                q_b = index.parse_query(op_bucket, default_field_names=["opening_ply_bucket"])
                subqueries.append((Occur.Should, Query.boost_query(q_b, 1.1)))
            except Exception as e:
                logger.debug("Opening boost parse skip: %s", e)

        if v2 and rag_phase == "endgame" and end_q:
            try:
                q_e = index.parse_query(end_q, default_field_names=["endgame_signature"])
                subqueries.append((Occur.Should, Query.boost_query(q_e, 0.5)))
            except Exception as e:
                logger.debug("Endgame extra pass skip: %s", e)

        for m in (query.tactical_motifs or [])[:4]:
            if not m or rag_phase != "middlegame":
                continue
            try:
                qm = index.parse_query(str(m), default_field_names=["strategic_tags"])
                subqueries.append((Occur.Should, Query.boost_query(qm, 0.4)))
            except Exception:
                pass

        bool_q = Query.boolean_query(subqueries)
        try:
            limit = max(30, top_k * 6)
            hits = searcher.search(bool_q, limit=limit).hits
        except Exception as e:
            logger.warning("Tantivy RAG search failed: %s", e)
            return []

        q_phase = rag_phase
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
            oeco = (doc.get_first("opening_eco") or "").strip() if v2 else ""
            d_sig = (doc.get_first("endgame_sig") or "").strip() if v2 else ""
            d_fp = (doc.get_first("pawn_fingerprint") or "").strip() if v2 else ""
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
                if q_phase == "endgame":
                    bsim = _board_similarity_endgame(board, bd, end_q, d_sig)
                elif q_phase == "opening":
                    bsim = _board_similarity_opening(board, bd, qeco, deco, oeco)
                else:
                    bsim = _board_similarity_middlegame(
                        board,
                        bd,
                        qeco,
                        deco,
                        v2=v2,
                        query_phase=query.phase,
                        ply=ply,
                    )
                if v2 and q_phase == "middlegame":
                    bsim = 0.9 * bsim + 0.1 * _fpc_match(fpc_q, d_fp)
            else:
                bsim = 0.0
            w_bm, w_sm = 0.7, 0.3
            if q_phase == "endgame":
                w_bm, w_sm = 0.55, 0.45
            elif q_phase == "opening":
                w_bm, w_sm = 0.6, 0.4
            final = w_bm * bm25_norm + w_sm * bsim
            ann_boost = 1.0 / (1 + max(plies_ann, 0))
            final *= ann_boost
            finals.append(
                (final, (doc, pv_san, fen, deco, oeco, ann, plies_ann, bm25, d_sig, d_fp, bm25_norm, bsim))
            )

        finals.sort(key=lambda x: x[0], reverse=True)
        if not finals:
            return []
        best_score = finals[0][0]
        if best_score < min_score:
            logger.info(
                "RAG: no pass threshold phase=%s best_score=%.4f min=%.4f (retrieving nothing)",
                rag_phase,
                best_score,
                min_score,
            )
            return []

        out: List[RAGResult] = []
        seen_fen: set[str] = set()
        for final, payload in finals:
            if final < min_score:
                break
            if len(out) >= top_k:
                break
            (doc, pv_san, fen, deco, oeco, ann, plies_ann, bm25, d_sig, d_fp, bm25n, bsim) = payload
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
            tags = {
                "phase": rag_phase,
                "corpus": f"v{self._corpus_version or _read_corpus_version(self._path)}",
                "eco": oeco or deco,
                "pv": pv_san[:200] if pv_san else "",
                "plies_to_next_annotation": str(plies_ann),
                "bm25_norm": f"{bm25n:.3f}",
                "board_sim": f"{bsim:.3f}",
            }
            if v2 and d_sig:
                tags["endgame_sig"] = d_sig[:120]
            if v2 and oeco:
                tags["opening_eco"] = oeco
            if v2 and (doc.get_first("opening_ply_bucket") or ""):
                tags["opening_ply_bucket"] = (doc.get_first("opening_ply_bucket") or "").strip()
            if v2 and (query.opening_name or (doc.get_first("opening_name") or "")):
                if query.opening_name:
                    tags["opening"] = (query.opening_name or "")[:80]
            out.append(
                RAGResult(
                    source=source or "lichess",
                    fen=fen or None,
                    annotation_text=text_body[:1200],
                    relevance_tags=tags,
                    similarity_score=float(final),
                )
            )
            logger.debug(
                "Tantivy rerank final=%.4f bm25=%.4f phase=%s source=%s",
                final,
                bm25,
                rag_phase,
                source or "lichess",
            )

        return out
