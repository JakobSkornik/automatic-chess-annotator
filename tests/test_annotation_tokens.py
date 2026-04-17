"""Tests for auto_tokenize and token protection."""

import unittest

import chess

from app.core.commentary.annotation_tokens import auto_tokenize


class TestAutoTokenize(unittest.TestCase):
    def test_san_and_eval_retokenization(self) -> None:
        """Bare SAN and signed evals become [move:] / [eval:] (Scandinavian-style FEN)."""
        b = chess.Board()
        b.push_san("e4")
        b.push_san("d5")
        b.push_san("exd5")
        fen_before = b.fen()
        b.push_san("Qxd5")
        fen_after = b.fen()
        s = "Qxd5 simplifies; better is Nf6. Eval drops from +0.34 to +0.20."
        out = auto_tokenize(s, fen_before, fen_after)
        self.assertEqual(
            out,
            "[move:Qxd5] simplifies; better is [move:Nf6]. Eval drops from [eval:+0.34] to [eval:+0.20].",
        )

    def test_pv_content_not_retokenized(self) -> None:
        """SAN-like text inside an existing [pv:...] block stays untouched."""
        b = chess.Board()
        fen_before = b.fen()
        b.push_san("e4")
        fen_after = b.fen()
        s = "See [pv:e4 e5 Nf3] for the idea."
        out = auto_tokenize(s, fen_before, fen_after)
        self.assertEqual(out, s)


if __name__ == "__main__":
    unittest.main()
