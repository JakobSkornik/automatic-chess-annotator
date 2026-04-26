"""Tests for RAG phase / opening / endgame feature helpers."""

import unittest

import chess

from app.core.commentary.features.rag_phase_features import (
    CORPUS_VERSION,
    classify_rag_phase,
    endgame_signature,
    material_signature,
    opening_ply_bucket,
    pawn_structure_fingerprint,
)


class TestRagPhaseFeatures(unittest.TestCase):
    def test_corpus_version_set(self) -> None:
        self.assertEqual(CORPUS_VERSION, "2")

    def test_start_position_is_opening(self) -> None:
        b = chess.Board()
        self.assertEqual(classify_rag_phase(b), "opening")

    def test_opening_ply_bucket(self) -> None:
        self.assertEqual(opening_ply_bucket(4), "1-6")
        self.assertEqual(opening_ply_bucket(10), "7-12")
        self.assertEqual(opening_ply_bucket(25), "21+")

    def test_material_signature_symmetric_start(self) -> None:
        b = chess.Board()
        s = material_signature(b)
        self.assertIn("v", s)
        self.assertTrue(s.startswith("K") or "K" in s)

    def test_pawn_fingerprint_stable(self) -> None:
        b = chess.Board()
        self.assertIn("wf:", pawn_structure_fingerprint(b))

    def test_endgame_signature_has_material_tag(self) -> None:
        b = chess.Board("8/8/8/8/8/8/4K3/4k3 w - - 0 1")
        sig = endgame_signature(b)
        self.assertIn("mat_sig:", sig)


if __name__ == "__main__":
    unittest.main()
