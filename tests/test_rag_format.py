"""Tests for RAG snippet formatting / filtering."""

from __future__ import annotations

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.rag_retriever import RAGResult


def _long_enough_prose() -> str:
    return (
        "White develops naturally while Black fianchettoes the king bishop. "
        "Both sides contest the center with pawns and minor pieces. "
        "The middlegame brings opposite-side castling and mutual threats. "
        "Tactical motifs appear along open files and weak squares near the king. "
        "Eventually simplifications favor the side with better coordination."
    )


def test_format_rag_block_strips_source_and_truncates_at_sentence() -> None:
    text = _long_enough_prose() + " Extra tail without closing sentence boundary words"
    r = RAGResult(
        source=r"C:\secrets\book.pgn",
        annotation_text=text,
        relevance_tags={"phase": "opening", "opening_name": "Test"},
        similarity_score=0.9,
    )
    block, snippets = AdvancedCommentService._format_rag_block([r], max_chars=400)
    assert "source=" not in block
    assert "secrets" not in block
    assert snippets and len(snippets[0]) <= 400


def test_format_rag_filters_low_score() -> None:
    r = RAGResult(
        source="x.pgn",
        annotation_text=_long_enough_prose(),
        similarity_score=0.50,
    )
    block, snippets = AdvancedCommentService._format_rag_block([r])
    assert block == ""
    assert snippets == []
