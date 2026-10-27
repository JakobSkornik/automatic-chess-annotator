"""Cross-rule invariants on legal trajectories, including reversed colours."""

import logging

import chess
import pytest

from app.core.commentary.features.envisioned import diff_vectors
from app.core.commentary.features.guid_features import compute_feature_vector
from app.core.commentary.rules.catalog import ALL_RULES, run_rules
from app.core.commentary.rules.context import _Ctx


LINES = [
    ("startpos", "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Qxd4 Qxd4 Nxd4"),
    ("startpos", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6"),
    ("4k3/pp6/8/8/8/2K5/PP6/8 w - - 0 1", "a4 a5 b4 axb4+ Kxb4 Kd7"),
]


def positions():
    for fen, sans in LINES:
        board = chess.Board() if fen == "startpos" else chess.Board(fen)
        for san in sans.split():
            before = board.copy()
            board.push_san(san)
            yield before, board.copy()


def signature(claims, *, mirror=False):
    # Text can legitimately vary with its position seed; semantic selection cannot.
    return sorted(
        (c.rule_id,
         ({"white": "black", "black": "white"}.get(c.beneficiary, c.beneficiary)
          if mirror else c.beneficiary),
         c.delta_cp)
        for c in claims
    )


@pytest.mark.parametrize("before,after", list(positions()))
def test_all_registered_rules_are_colour_symmetric_and_do_not_raise(before, after, caplog):
    def evaluate(start, leaf):
        diff = diff_vectors(compute_feature_vector(start), compute_feature_vector(leaf))
        mover = "WHITE" if start.turn else "BLACK"
        ctx = _Ctx(diff, "end", mover, 0, start, leaf)
        # Call directly too: run_rules deliberately logs and swallows exceptions.
        for rule in ALL_RULES:
            assert isinstance(rule(ctx), list), rule.__name__
        return run_rules(diff, phase="end", mover=mover, eval_cp=0,
                         start_board=start, leaf_board=leaf, cap=False)

    with caplog.at_level(logging.WARNING):
        claims = evaluate(before, after)
        mirrored = evaluate(before.mirror(), after.mirror())
    assert not any("failed:" in record.message for record in caplog.records)
    assert signature(claims) == signature(mirrored, mirror=True)


@pytest.mark.parametrize("before,after", list(positions()))
def test_unchanged_position_does_not_invent_a_change(before, after):
    vec = compute_feature_vector(after)
    claims = run_rules(diff_vectors(vec, vec), phase="end", mover="WHITE",
                       eval_cp=0, start_board=after, leaf_board=after, cap=False)
    assert all(c.rule_id == "material_standing" for c in claims)
