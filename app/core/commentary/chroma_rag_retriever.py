"""ChromaDB-backed RAG retriever for master-game annotation embeddings."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.core.commentary.composite_retriever import CompositeRetriever
from app.core.commentary.rag_retriever import NullRetriever, RAGQuery, RAGResult, RAGRetriever
from app.core.commentary.tantivy_positional_retriever import TantivyPositionalRetriever

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "chess_annotations"


def _build_query_text(query: RAGQuery) -> str:
    parts: List[str] = []
    if query.phase:
        parts.append(query.phase)
    if query.pawn_structure_type:
        parts.append(f"{query.pawn_structure_type} center")
    if query.theme_hint:
        parts.append(query.theme_hint)
    if query.tactical_motifs:
        parts.append(", ".join(query.tactical_motifs[:3]))
    if query.material_imbalance and query.material_imbalance != "equal":
        parts.append(query.material_imbalance)
    return ". ".join(parts) if parts else "chess position"


def _build_where_filter(query: RAGQuery) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    if query.phase:
        clauses.append({"phase": {"$eq": query.phase}})
    if query.pawn_structure_type:
        clauses.append({"pawn_structure_type": {"$eq": query.pawn_structure_type}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _distance_to_score(distance: float) -> float:
    """Map Chroma distance to a rough [0, 1] similarity for logging/UI."""
    return round(max(0.0, 1.0 / (1.0 + float(distance))), 4)


class ChromaRAGRetriever(RAGRetriever):
    """Retrieve annotated examples from a persistent ChromaDB collection."""

    def __init__(self, chroma_path: str) -> None:
        self._chroma_path = os.path.abspath(chroma_path)
        self._openai = OpenAI()
        self._collection = None

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        import chromadb

        client = chromadb.PersistentClient(path=self._chroma_path)
        self._collection = client.get_collection(COLLECTION_NAME)
        return self._collection

    async def retrieve(self, query: RAGQuery, top_k: int = 2) -> List[RAGResult]:
        query_text = _build_query_text(query)
        where_filter = _build_where_filter(query)
        t0 = time.perf_counter()
        try:
            coll = self._get_collection()
            resp = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
            emb = resp.data[0].embedding

            def _run_query(w: Optional[Dict[str, Any]]) -> Any:
                return coll.query(
                    query_embeddings=[emb],
                    n_results=top_k,
                    where=w,
                    include=["documents", "metadatas", "distances"],
                )

            raw = _run_query(where_filter)
            hits = self._results_to_rag_results(raw)
            if not hits and where_filter is not None:
                raw = _run_query(None)
                hits = self._results_to_rag_results(raw)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "RAG embed+query: text=%.100s where=%s top_k=%d latency_ms=%.1f hits=%d",
                query_text,
                where_filter,
                top_k,
                elapsed_ms,
                len(hits),
            )
            return hits
        except Exception as e:
            logger.warning("Chroma RAG retrieval failed, returning no hits: %s", e, exc_info=True)
            return []

    def _results_to_rag_results(self, raw: Any) -> List[RAGResult]:
        if not isinstance(raw, dict):
            raw = {k: getattr(raw, k) for k in ("ids", "distances", "documents", "metadatas") if hasattr(raw, k)}
        ids = raw.get("ids") or [[]]
        docs = raw.get("documents") or [[]]
        metas = raw.get("metadatas") or [[]]
        dists = raw.get("distances") or [[]]
        if not ids[0]:
            return []

        out: List[RAGResult] = []
        for i in range(len(ids[0])):
            meta = metas[0][i] or {}
            doc = docs[0][i] or ""
            dm = dists[0][i] if dists and dists[0] else 0.0
            score = _distance_to_score(float(dm))
            tags: Dict[str, str] = {}
            for k in ("phase", "pawn_structure_type", "san"):
                if k in meta and meta[k] is not None:
                    tags[k] = str(meta[k])
            out.append(
                RAGResult(
                    source=str(meta.get("source_file", "unknown")),
                    fen=str(meta["fen"]) if meta.get("fen") else None,
                    annotation_text=doc,
                    relevance_tags=tags,
                    similarity_score=score,
                )
            )
        return out


def get_default_retriever() -> RAGRetriever:
    """Chroma when RAG_CHROMA_PATH exists; Tantivy BM25 when RAG_BM25_PATH exists; composite if both."""
    rs: List[RAGRetriever] = []

    bm25_raw = os.environ.get("RAG_BM25_PATH")
    if bm25_raw:
        bm25_path = os.path.abspath(os.path.expanduser(bm25_raw.strip()))
        if os.path.isdir(bm25_path):
            try:
                rs.append(TantivyPositionalRetriever(bm25_path))
                logger.info("RAG: using TantivyPositionalRetriever at %s", bm25_path)
            except Exception as e:
                logger.warning("RAG: RAG_BM25_PATH invalid (%s), skipping BM25: %s", bm25_path, e)
        else:
            logger.warning("RAG: RAG_BM25_PATH=%s is not a directory, skipping BM25", bm25_raw)

    chroma_raw = os.environ.get("RAG_CHROMA_PATH")
    if chroma_raw:
        chroma_path = os.path.abspath(os.path.expanduser(chroma_raw.strip()))
        if os.path.isdir(chroma_path):
            rs.append(ChromaRAGRetriever(chroma_path))
            logger.info("RAG: using ChromaRAGRetriever at %s", chroma_path)
        else:
            logger.warning("RAG: RAG_CHROMA_PATH=%s is not a directory, skipping Chroma", chroma_raw)

    if not rs:
        logger.info("RAG: no Chroma/BM25 paths valid, using NullRetriever")
        return NullRetriever()
    if len(rs) == 1:
        return rs[0]
    return CompositeRetriever(rs)
