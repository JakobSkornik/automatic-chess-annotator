"""Combine multiple RAGRetriever backends (e.g. Tantivy BM25 + Chroma embeddings)."""

from __future__ import annotations

import asyncio
from typing import List, Set, Tuple

from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever


def _result_key(r: RAGResult) -> Tuple[str, str, str]:
    return (r.fen or "", r.source, (r.annotation_text or "")[:160])


def _interleave_unique(blocks: List[List[RAGResult]], max_results: int) -> List[RAGResult]:
    """Round-robin from each retriever, dedupe by (fen, source, text prefix)."""
    if not blocks or max_results <= 0:
        return []
    seen: Set[Tuple[str, str, str]] = set()
    out: List[RAGResult] = []
    max_len = max(len(b) for b in blocks)
    for i in range(max_len):
        for b in blocks:
            if len(out) >= max_results:
                return out
            if i >= len(b):
                continue
            r = b[i]
            k = _result_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


class CompositeRetriever(RAGRetriever):
    """Fan out to child retrievers and interleave unique hits."""

    def __init__(self, retrievers: List[RAGRetriever]) -> None:
        self._rs = [r for r in retrievers if r is not None]
        if not self._rs:
            raise ValueError("CompositeRetriever needs at least one retriever")

    async def retrieve(self, query: RAGQuery, top_k: int = 2) -> List[RAGResult]:
        results = await asyncio.gather(*(r.retrieve(query, top_k) for r in self._rs))
        cap = max(1, top_k * len(self._rs))
        return _interleave_unique(list(results), cap)
