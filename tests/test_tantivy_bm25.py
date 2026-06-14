"""Tantivy BM25 retriever smoke tests (in-memory index)."""

import asyncio
import json
import os
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
        "king_placement",
        "imbalance_signature",
    ):
        sb.add_text_field(name, tokenizer_name="whitespace", stored=False)
    for name in (
        "player_color",
        "game_id",
        "eco",
        "source",
        "fen",
        "pv_san",
        "annotation_text",
        "annotation_ply",
        "plies_to_next_annotation",
    ):
        sb.add_text_field(name, tokenizer_name="raw", stored=True)
    return sb.build()


def _add_doc(
    writer: tantivy.IndexWriter,
    enc: dict,
    fen: str,
    eco: str,
    *,
    ann: str = "",
    plies_ann: str = "0",
) -> None:
    doc = tantivy.Document()
    for k in (
        "static_attributes",
        "pawn_structure",
        "center",
        "dynamic_general",
        "dynamic_solution",
        "king_placement",
        "imbalance_signature",
    ):
        doc.add_text(k, enc[k])
    doc.add_text("player_color", enc["player_color"])
    doc.add_text("game_id", "test:0")
    doc.add_text("eco", eco)
    doc.add_text("source", "unit_test.pgn")
    doc.add_text("fen", fen)
    doc.add_text("pv_san", enc["pv_san"])
    doc.add_text("annotation_text", ann)
    doc.add_text("annotation_ply", "")
    doc.add_text("plies_to_next_annotation", plies_ann)
    writer.add_document(doc)


def _build_schema_v2() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    for name in (
        "static_attributes",
        "pawn_structure",
        "center",
        "dynamic_general",
        "dynamic_solution",
        "king_placement",
        "imbalance_signature",
        "strategic_tags",
        "endgame_signature",
    ):
        sb.add_text_field(name, tokenizer_name="whitespace", stored=False)
    for name in (
        "player_color",
        "game_id",
        "eco",
        "source",
        "fen",
        "pv_san",
        "annotation_text",
        "annotation_ply",
        "plies_to_next_annotation",
        "rag_phase",
        "opening_eco",
        "opening_prefix",
        "opening_name",
        "opening_matched_ply",
        "opening_ply_bucket",
        "material_signature",
        "material_bucket",
        "pawn_fingerprint",
        "corpus_version",
        "endgame_sig",
    ):
        sb.add_text_field(name, tokenizer_name="raw", stored=True)
    return sb.build()


def _add_doc_v2(
    writer: tantivy.IndexWriter,
    enc: dict,
    fen: str,
    eco: str,
    *,
    rag_phase: str = "opening",
    ann: str = "",
    plies_ann: str = "0",
    opening_eco: str = "",
    opening_ply_bucket: str = "1-6",
    endgame_sig: str = "",
    strat: str = "",
    pfp: str = "",
) -> None:
    doc = tantivy.Document()
    for k in (
        "static_attributes",
        "pawn_structure",
        "center",
        "dynamic_general",
        "dynamic_solution",
        "king_placement",
        "imbalance_signature",
    ):
        doc.add_text(k, enc[k])
    doc.add_text("strategic_tags", strat)
    doc.add_text("endgame_signature", endgame_sig)
    doc.add_text("player_color", enc["player_color"])
    doc.add_text("game_id", "test:v2")
    doc.add_text("eco", eco)
    doc.add_text("source", "unit_test.pgn")
    doc.add_text("fen", fen)
    doc.add_text("pv_san", enc["pv_san"])
    doc.add_text("annotation_text", ann)
    doc.add_text("annotation_ply", "")
    doc.add_text("plies_to_next_annotation", plies_ann)
    doc.add_text("rag_phase", rag_phase)
    doc.add_text("opening_eco", opening_eco or eco)
    doc.add_text(
        "opening_prefix", (opening_eco or eco)[:2] if (opening_eco or eco) else ""
    )
    doc.add_text("opening_name", "")
    doc.add_text("opening_matched_ply", "0")
    doc.add_text("opening_ply_bucket", opening_ply_bucket)
    doc.add_text("material_signature", "")
    doc.add_text("material_bucket", "")
    doc.add_text("pawn_fingerprint", pfp)
    doc.add_text("corpus_version", "2")
    doc.add_text("endgame_sig", endgame_sig)
    writer.add_document(doc)


def _write_v2_metadata(path: Path) -> None:
    (path / "metadata.json").write_text(
        json.dumps({"corpus_version": "2"}), encoding="utf-8"
    )


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
            _add_doc(writer, enc, target_fen, "B20")
            writer.commit()
            idx.reload()

            retriever = TantivyPositionalRetriever(str(path))

            async def _run() -> None:
                q = RAGQuery(
                    fen=target_fen, pv_san=pv, eco="B20", ply=8, phase="opening"
                )
                hits = await retriever.retrieve(q, top_k=2)
                self.assertTrue(hits, "expected at least one BM25 hit")
                top = hits[0]
                self.assertEqual(top.fen, target_fen)
                self.assertIsNotNone(top.similarity_score)
                self.assertGreater(top.similarity_score or 0, 0.0)

            asyncio.run(_run())

    def test_pawn_skeleton_rerank_prefers_matching_structure(self) -> None:
        """Near-miss: same-ish BM25 tokens but better board_sim should rank first."""
        b0 = chess.Board()
        pv = ["d4", "d5", "c4"]
        enc0 = encode_position(b0, pv)
        fen0 = b0.fen()

        b1 = chess.Board()
        b1.push_san("e4")
        b1.push_san("e5")
        enc1 = encode_position(b1, pv)
        fen1 = b1.fen()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            schema = _build_schema()
            idx = tantivy.Index(schema, path=str(path))
            writer = idx.writer()
            _add_doc(writer, enc1, fen1, "C20")
            _add_doc(writer, enc0, fen0, "D30")
            writer.commit()
            idx.reload()

            retriever = TantivyPositionalRetriever(str(path))

            async def _run() -> None:
                q = RAGQuery(fen=fen0, pv_san=pv, eco="D30", ply=12, phase="opening")
                hits = await retriever.retrieve(q, top_k=2)
                self.assertGreaterEqual(len(hits), 1)
                self.assertEqual(hits[0].fen, fen0)

            asyncio.run(_run())

    def test_eco_query_runs_in_middlegame_and_endgame(self) -> None:
        """RAGQuery.eco is accepted; retrieval succeeds (ECO gate exercised in endgame)."""
        board = chess.Board()
        pv = ["e4", "e5", "Nf3"]
        enc = encode_position(board, pv)
        fen = board.fen()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            schema = _build_schema()
            idx = tantivy.Index(schema, path=str(path))
            writer = idx.writer()
            _add_doc(writer, enc, fen, "B12")
            writer.commit()
            idx.reload()

            retriever = TantivyPositionalRetriever(str(path))

            async def _run_mid() -> None:
                q = RAGQuery(fen=fen, pv_san=pv, eco="B12", ply=10, phase="middlegame")
                hits = await retriever.retrieve(q, top_k=2)
                self.assertTrue(hits)

            async def _run_end() -> None:
                q = RAGQuery(fen=fen, pv_san=pv, eco="B12", ply=40, phase="endgame")
                hits = await retriever.retrieve(q, top_k=2)
                self.assertTrue(hits)

            asyncio.run(_run_mid())
            asyncio.run(_run_end())

    def test_v2_phase_must_excludes_mismatched_row(self) -> None:
        """v2 index: rag_phase Must filter — wrong phase on doc yields no hits."""
        board = chess.Board()
        pv = ["e4", "e5", "Nf3"]
        enc = encode_position(board, pv)
        fen = board.fen()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _write_v2_metadata(path)
            schema = _build_schema_v2()
            idx = tantivy.Index(schema, path=str(path))
            writer = idx.writer()
            _add_doc_v2(
                writer, enc, fen, "B12", rag_phase="middlegame", opening_eco="B12"
            )
            writer.commit()
            idx.reload()

            retriever = TantivyPositionalRetriever(str(path))

            async def _run() -> None:
                q = RAGQuery(fen=fen, pv_san=pv, eco="B12", ply=4, phase="opening")
                hits = await retriever.retrieve(q, top_k=2)
                self.assertEqual(hits, [])

            asyncio.run(_run())

    def test_v2_threshold_returns_empty(self) -> None:
        """When min score is impossibly high, retrieve nothing."""
        board = chess.Board()
        pv = ["e4", "e5", "Nf3", "Nc6", "Bc4"]
        enc = encode_position(board, pv)
        fen = board.fen()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _write_v2_metadata(path)
            schema = _build_schema_v2()
            idx = tantivy.Index(schema, path=str(path))
            writer = idx.writer()
            _add_doc_v2(
                writer,
                enc,
                fen,
                "B20",
                rag_phase="opening",
                opening_eco="B20",
                opening_ply_bucket="1-6",
            )
            writer.commit()
            idx.reload()

            retriever = TantivyPositionalRetriever(str(path))

            async def _run() -> None:
                os.environ["RAG_MIN_SCORE_OPENING"] = "1.01"
                try:
                    q = RAGQuery(fen=fen, pv_san=pv, eco="B20", ply=5, phase="opening")
                    hits = await retriever.retrieve(q, top_k=2)
                    self.assertEqual(hits, [])
                finally:
                    os.environ.pop("RAG_MIN_SCORE_OPENING", None)

            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
