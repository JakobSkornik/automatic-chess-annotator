"""Corpus: recursive PGN discovery (same rglob contract as build_bm25_corpus)."""

import tempfile
import unittest
from pathlib import Path


class TestRecursivePgnCollection(unittest.TestCase):
    def test_rglob_finds_nested_pgns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub" / "deeper").mkdir(parents=True)
            (root / "sub" / "deeper" / "a.pgn").write_text(
                '[Event "x"]\n1. e4 e5\n', encoding="utf-8"
            )
            (root / "top.pgn").write_text('[Event "y"]\n1. d4 d5\n', encoding="utf-8")
            found = sorted(p for p in root.rglob("*.pgn") if p.is_file())
            self.assertEqual(len(found), 2)
            self.assertTrue(any("deeper" in str(p) for p in found))


if __name__ == "__main__":
    unittest.main()
