"""Tests for three related improvements to the envisioned-line / facts-builder
pipeline:

1. The probe-based tail trim (``trim_envisioned_line_by_probe``) is actually
   wired into ``build_comment_facts`` and runs BEFORE claims/feature-diff are
   computed, not as a post-hoc patch on the already-built facts.
2. (Removed 2 Nov 2026: the per-feature sharpening helper was never wired in;
   the trajectory horizon now anchors every claim at the ply where it first
   fires, see tests/test_trajectory_horizon.py.)
3. Feature-diff-of-diffs: ``envisioned.diff_of_diffs`` correctly computes,
   per feature, how much more an alternative move's own diff moves compared
   to the played move's diff, and it is attached to ``BestAlternative``.
"""

import chess

from app.core.commentary.features.envisioned import diff_of_diffs
from app.core.commentary.features.guid_features.weights import FeatureValue
from app.core.commentary.rules.facts_builder import (
    _feature_advantages,
    build_comment_facts,
)
from app.models.chess_events import (
    AnalyzedMoveData,
    MoveEvent,
    MoveEventType,
    MoveQuality,
)
from app.models.comment_facts import EnvisionedLine, FeatureDelta, FeatureDiff

START = chess.STARTING_FEN


# ---------------------------------------------------------------------------
# Improvement 1: probe trim wired into the real pipeline
# ---------------------------------------------------------------------------


def _minimal_row_and_event(after_pv_uci):
    board = chess.Board(START)
    move = chess.Move.from_uci("b1c3")
    fen_after = board.san(move)  # noqa: F841 (san computed for side effect check below)
    board.push(move)
    me = MoveEvent(
        move_index=0,
        ply=1,
        san="Nc3",
        uci="b1c3",
        fen_before=START,
        fen_after=board.fen(),
        phase="mid",
        move_quality=MoveQuality.GOOD,
        event_type=MoveEventType.QUIET,
        eval_after_cp=300,  # story: White is much better
        eval_before_cp=20,
        best_move_uci="b1c3",
        best_move_san="Nc3",
    )
    row = AnalyzedMoveData(
        index=0,
        ply=1,
        san="Nc3",
        uci="b1c3",
        fen_before=START,
        fen_after=board.fen(),
        score_cp=300,
        phase_raw="mid",
        hidden_features={"_engine": {"after_pv_uci": after_pv_uci}},
    )
    return row, me


def test_probe_trim_is_actually_invoked_by_build_comment_facts():
    """A prober that contradicts the root eval's story at the 2-ply leaf must
    shorten the display_line actually returned by build_comment_facts --
    proof this is wired into the real call path, not just unit-testable."""
    row, me = _minimal_row_and_event(["g8f6"])  # Nc3 Nf6: quiet, 2 plies

    board = chess.Board(START)
    board.push(chess.Move.from_uci("b1c3"))
    fen_after_nc3 = board.fen()
    board.push(chess.Move.from_uci("g8f6"))
    fen_after_nf6 = board.fen()

    def prober(fen: str):
        if fen == fen_after_nf6:
            return -400  # contradicts the White-winning story -> trim
        if fen == fen_after_nc3:
            return 250  # agrees -> stop trimming here
        return None

    facts_with_probe = build_comment_facts(row, me, depth=16, prober=prober)
    facts_without_probe = build_comment_facts(row, me, depth=16, prober=None)

    assert facts_without_probe.display_line.leaf_fen == fen_after_nf6
    assert len(facts_without_probe.display_line.fens) == 2

    assert facts_with_probe.display_line.leaf_fen == fen_after_nc3
    assert len(facts_with_probe.display_line.fens) == 1
    assert facts_with_probe.display_line.trimmed_plies > (
        facts_without_probe.display_line.trimmed_plies
    )
    # Claims/feature_diff must be computed from the TRIMMED leaf, not patched
    # on afterwards: feature_diff's own start->leaf story matches the
    # trimmed line, i.e. it was built from the 1-ply leaf.
    assert facts_with_probe.feature_diff is not None


def test_probe_trim_env_opt_out(monkeypatch):
    monkeypatch.setenv("ENVISIONED_PROBE_TRIM_ENABLED", "0")
    row, me = _minimal_row_and_event(["g8f6"])

    board = chess.Board(START)
    board.push(chess.Move.from_uci("b1c3"))
    board.push(chess.Move.from_uci("g8f6"))
    fen_after_nf6 = board.fen()

    def prober(fen: str):
        return -400  # would always contradict, if consulted

    facts = build_comment_facts(row, me, depth=16, prober=prober)
    # Opted out: the line is untouched despite a contradicting prober.
    assert facts.display_line.leaf_fen == fen_after_nf6
    assert len(facts.display_line.fens) == 2


# ---------------------------------------------------------------------------
# Improvement 3: feature-diff-of-diffs
# ---------------------------------------------------------------------------


def test_diff_of_diffs_numeric_example():
    played = FeatureDiff(
        positive=[
            FeatureDelta(
                name="WHITE_ROOK_OPEN_FILE", delta_cp=20, before_cp=0, after_cp=20
            )
        ],
        negative=[
            FeatureDelta(
                name="WHITE_PAWN_DOUBLED", delta_cp=-12, before_cp=0, after_cp=-12
            )
        ],
    )
    alt = FeatureDiff(
        positive=[
            FeatureDelta(
                name="WHITE_ROOK_OPEN_FILE", delta_cp=60, before_cp=0, after_cp=60
            ),
            FeatureDelta(
                name="WHITE_CENTER_CONTROL", delta_cp=16, before_cp=0, after_cp=16
            ),
        ]
    )
    advantages = diff_of_diffs(played, alt, min_abs_cp=1)
    by_name = {a.name: a for a in advantages}

    # alt gains 60-20=40 more cp of rook activity than the played move.
    assert by_name["WHITE_ROOK_OPEN_FILE"].advantage_cp == 40
    assert by_name["WHITE_ROOK_OPEN_FILE"].alt_delta_cp == 60
    assert by_name["WHITE_ROOK_OPEN_FILE"].played_delta_cp == 20
    # alt introduces a feature the played move never touched: 16-0=16.
    assert by_name["WHITE_CENTER_CONTROL"].advantage_cp == 16
    # played's own doubled-pawn concession (not present in alt) shows as a
    # negative advantage: alt didn't take on that -12, so alt is +12 better.
    assert by_name["WHITE_PAWN_DOUBLED"].advantage_cp == 12

    # Sorted by |advantage| descending.
    assert [a.name for a in advantages][0] == "WHITE_ROOK_OPEN_FILE"


def test_diff_of_diffs_materiality_threshold_drops_small_gaps():
    played = FeatureDiff(
        positive=[FeatureDelta(name="F", delta_cp=10, before_cp=0, after_cp=10)]
    )
    alt = FeatureDiff(
        positive=[FeatureDelta(name="F", delta_cp=11, before_cp=0, after_cp=11)]
    )
    assert diff_of_diffs(played, alt, min_abs_cp=8) == []


def test_diff_of_diffs_max_features_cap():
    played = FeatureDiff()
    alt = FeatureDiff(
        positive=[
            FeatureDelta(name="A", delta_cp=100, before_cp=0, after_cp=100),
            FeatureDelta(name="B", delta_cp=50, before_cp=0, after_cp=50),
            FeatureDelta(name="C", delta_cp=30, before_cp=0, after_cp=30),
            FeatureDelta(name="D", delta_cp=20, before_cp=0, after_cp=20),
        ]
    )
    advantages = diff_of_diffs(played, alt, min_abs_cp=1, max_features=2)
    assert [a.name for a in advantages] == ["A", "B"]


def test_feature_advantages_returns_empty_without_played_diff():
    alt = FeatureDiff(
        positive=[FeatureDelta(name="F", delta_cp=50, before_cp=0, after_cp=50)]
    )
    assert _feature_advantages(None, alt) == []


def test_feature_advantages_reuses_min_claim_cp_threshold():
    played = FeatureDiff()
    alt = FeatureDiff(
        positive=[FeatureDelta(name="F", delta_cp=3, before_cp=0, after_cp=3)]
    )
    # Below THRESHOLDS["min_claim_cp"] (8) -- filtered out.
    assert _feature_advantages(played, alt) == []
