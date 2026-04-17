"""Tantivy BM25 retriever smoke tests (in-memory index)."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import chess
import tantivy

from app.core.commentary.features.positional_tokens import encode_position
from app.core.commentary.rag_retriever import RAGQuery
from app.core.commentary.tantivy_positional_retriever import TantivyPositionalRetriever


def _build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    for name in (
        "static_attributes",
        "pawn_structure",
        "center",
        "dynamic_general",
        "dynamic_solution",
    ):
        sb.add_text_field(name, tokenizer_name="whitespace", stored=False)
    for name in ("player_color", "game_id", "eco", "source", "fen", "pv_san"):
        sb.add_text_field(name, tokenizer_name="raw", stored=True)
    return sb.build()


class TestTantivyBM25(unittest.TestCase):
    def test_self_retrieval_top_hit(self) -> None:
        """Indexed FEN should rank highly when queried with the same encoded features."""
        board = chess.Board()
        pv = ["e4", "e5", "Nf3", "Nc6", "Bc4"]
        enc = encode_position(board, pv)
        target_fen = board.fen()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            schema = _build_schema()
            idx = tantivy.Index(schema, path=str(path))
            writer = idx.writer()
            doc = tantivy.Document()
            for k in (
                "static_attributes",
                "pawn_structure",
                "center",
                "dynamic_general",
                "dynamic_solution",
            ):
                doc.add_text(k, enc[k])
            doc.add_text("player_color", enc["player_color"])
            doc.add_text("game_id", "test:0")
            doc.add_text("eco", "B20")
            doc.add_text("source", "unit_test.pgn")
            doc.add_text("fen", target_fen)
            doc.add_text("pv_san", enc["pv_san"])
            writer.add_document(doc)
            writer.commit()
            idx.reload()

            retriever = TantivyPositionalRetriever(str(path))

            async def _run() -> None:
                q = RAGQuery(fen=target_fen, pv_san=pv)
                hits = await retriever.retrieve(q, top_k=2)
                self.assertTrue(hits, "expected at least one BM25 hit")
                top = hits[0]
                self.assertEqual(top.fen, target_fen)
                self.assertIsNotNone(top.similarity_score)
                self.assertGreater(top.similarity_score or 0, 0.0)

            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
