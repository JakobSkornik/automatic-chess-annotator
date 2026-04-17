"""Tantivy BM25 retriever for Bolčič-style positional tokens (embedded index, no external DB)."""

from __future__ import annotations

import logging
import os
from typing import List

import chess
import tantivy
from tantivy import Occur, Query

from app.core.commentary.features.positional_tokens import encode_position_strings_only
from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever

logger = logging.getLogger(__name__)

_FIELD_BOOSTS = [
    ("static_attributes", 1.0),
    ("pawn_structure", 1.0),
    ("center", 1.0),
    ("dynamic_general", 2.0),
    ("dynamic_solution", 2.0),
]


class TantivyPositionalRetriever(RAGRetriever):
    """BM25 over five text fields; requires RAGQuery.fen and pv_san."""

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
            hits = searcher.search(bool_q, limit=max(top_k * 2, 8)).hits
        except Exception as e:
            logger.warning("Tantivy RAG search failed: %s", e)
            return []

        out: List[RAGResult] = []
        seen_fen: set[str] = set()
        for score, addr in hits:
            if len(out) >= top_k:
                break
            try:
                doc = searcher.doc(addr)
            except Exception:
                continue
            fen = (doc.get_first("fen") or "").strip()
            pv_san = (doc.get_first("pv_san") or "").strip()
            source = (doc.get_first("source") or "").strip()
            eco = (doc.get_first("eco") or "").strip()
            if fen and fen in seen_fen:
                continue
            if fen:
                seen_fen.add(fen)
            out.append(
                RAGResult(
                    source=source or "lichess",
                    fen=fen or None,
                    annotation_text=f"PV: {pv_san}"[:1200],
                    relevance_tags={"eco": eco, "pv": pv_san[:200] if pv_san else ""},
                    similarity_score=float(score) if score is not None else None,
                )
            )
            logger.debug(
                "Tantivy BM25 hit score=%.4f source=%s fen=%.50s",
                float(score) if score is not None else 0.0,
                source or "lichess",
                fen or "",
            )

        return out
