"""Annotated PGN corpus helpers (junk filter + distance)."""

import unittest
from io import StringIO

import chess.pgn

from app.core.commentary.annotation_corpus import (
    is_junk_comment,
    nearest_annotation_distance,
    ply_to_comment_map_from_game,
)


class TestAnnotationCorpus(unittest.TestCase):
    def test_junk_comment_drops_short(self) -> None:
        self.assertTrue(is_junk_comment("?!"))
        self.assertTrue(is_junk_comment("short"))

    def test_valid_comment_kept(self) -> None:
        t = "This is a long enough human sentence about the plan on the queenside."
        self.assertFalse(is_junk_comment(t))

    def test_future_annotation_distance_prefers_sooner(self) -> None:
        ply_map = {10: "Valid comment text that is long enough for the filter here."}
        self.assertEqual(nearest_annotation_distance(ply_map, 10, max_delta=4), 0)
        self.assertIsNone(nearest_annotation_distance(ply_map, 5, max_delta=4))

    def test_forward_attachment_two_long_comments(self) -> None:
        """Comments at plies 4 and 9; forward-only window [0,4]."""
        c4 = "Comment at ply four with enough characters here."
        c9 = "Comment at ply nine with enough characters here too."
        pc = {4: c4, 9: c9}
        self.assertEqual(nearest_annotation_distance(pc, 4, max_delta=4), 0)
        self.assertEqual(nearest_annotation_distance(pc, 5, max_delta=4), 4)
        self.assertEqual(nearest_annotation_distance(pc, 8, max_delta=4), 1)
        self.assertEqual(nearest_annotation_distance(pc, 9, max_delta=4), 0)
        self.assertIsNone(nearest_annotation_distance(pc, 10, max_delta=4))

    def test_junk_not_in_ply_map(self) -> None:
        """Short / NAG comments must not appear in ply_to_comment map."""
        pgn = """[Event "x"]
1. e4 {?!} e5 2. Nf3 {short} Nc6 *
"""
        g = chess.pgn.read_game(StringIO(pgn))
        m = ply_to_comment_map_from_game(g)
        self.assertEqual(m, {})

    def test_ply_to_comment_map_from_annotated_pgn(self) -> None:
        pgn = """[Event "x"]
1. e4 {This human comment is definitely long enough for the filter.} e5 2. Nf3 Nc6 *
"""
        g = chess.pgn.read_game(StringIO(pgn))
        m = ply_to_comment_map_from_game(g)
        self.assertIn(1, m)
        self.assertIn("human comment", m[1])


if __name__ == "__main__":
    unittest.main()
