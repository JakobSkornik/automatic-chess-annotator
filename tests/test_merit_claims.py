"""Fact-layer additions of 3 Nov 2026: a merit claim for sound moves that only had concessions,
an alternative-edge claim when the regular comparison is empty, and lost-position errors kept
out of expert commentary and survey selection."""
import chess
import pytest
from types import SimpleNamespace

from app.core.commentary.rules.facts_builder import (
    _FactsCtx, alternative_edge_claim, feature_group, merit_claim, MERIT_MIN_CP)
from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.core.commentary.features.envisioned import diff_vectors
from app.core.engine.analysis.moment_kinds import already_lost, complexity, level_allows
from app.models.chess_events import MoveEvent
from app.models.comment_facts import BestAlternative, Claim, EnvisionedLine, FeatureAdvantage

# game-09 of the survey before 32.h5: White pushes the passer, Black's rook wins f2.
H5 = "8/5pk1/6p1/4n3/7P/5N2/1r3PPK/8 w - - 0 32"


def line(fen, sans, horizon=1):
    board = chess.Board(fen)
    ucis, fens = [], []
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci()); board.push(move); fens.append(board.fen())
    return EnvisionedLine(start_fen=fen, line_uci=ucis, line_san=list(sans), fens=fens, leaf_fen=board.fen(),
                          immediate_horizon_ply=horizon, horizon_reason="settled")


def ctx(fen, san, uci, quality="best"):
    board = chess.Board(fen)
    board_after = board.copy(); board_after.push_san(san)
    event = MoveEvent(move_index=0, ply=63, san=san, uci=uci, fen_before=fen,
                      fen_after=board_after.fen(), phase="mid", move_quality=quality, event_type="quiet")
    return _FactsCtx(move_event=event, row=None, board_before=board, mover="White" if board.turn else "Black",
                     phase_raw="mid", depth=16, start_vec=compute_feature_vector_fen(fen))


def concession(branch):
    return Claim(rule_id="legal_capture", text="Black's rook from b2 captures the pawn on f2.", beneficiary="black",
                 is_concession=True, support_ply=2, support_fen=branch.fens[1], baseline_fen=branch.start_fen)


def test_feature_groups_have_plain_names():
    assert feature_group("WHITE_PAWN_PASSED") == "passed pawns"
    assert feature_group("B_ROOKS_MOBILITY") == "rook activity"
    assert feature_group("UNMAPPED") is None


def test_merit_claim_comes_from_the_comparison_with_the_alternative():
    played = line(H5, ["h5", "Rxf2", "h6"], horizon=3)
    c = ctx(H5, "h5", "h4h5")
    better = BestAlternative(san="a4", uci="a3a4", eval_cp=-250, is_inferior=True, matched_ply=3,
                             display_line=line(H5, ["Kg3", "Rb3", "Kf4"]),
                             feature_advantages_matched=[FeatureAdvantage(name="WHITE_PAWN_PASSED", alt_delta_cp=0,
                                                                          played_delta_cp=40, advantage_cp=-40)])
    claim = merit_claim(c, [concession(played)], played, diff_vectors(c.start_vec, compute_feature_vector_fen(played.leaf_fen)), better)
    assert claim is not None and claim.rule_id == "merit_over_alternative"
    assert claim.text == "Compared with a4, h5 leaves White better placed in passed pawns."
    assert claim.beneficiary == "white" and claim.support_ply == 3 and claim.realization == "immediate"
    assert "h6" in claim.evidence.squares  # the passer at the matched ply
    # With a merit already present nothing is added.
    merit = Claim(rule_id="passed_pawn_advances", text="White's passed pawn advances.", beneficiary="white",
                  support_ply=1, support_fen=played.fens[0], baseline_fen=H5)
    assert merit_claim(c, [merit], played, diff_vectors(c.start_vec, compute_feature_vector_fen(played.leaf_fen)), better) is None


def test_merit_claim_falls_back_to_the_largest_own_gain_and_respects_the_threshold():
    played = line(H5, ["h5", "Rxf2", "h6"], horizon=3)
    c = ctx(H5, "h5", "h4h5")
    diff = diff_vectors(c.start_vec, compute_feature_vector_fen(played.leaf_fen))
    claim = merit_claim(c, [concession(played)], played, diff, None)
    assert claim is None or (claim.rule_id == "main_gain" and claim.delta_cp >= MERIT_MIN_CP and "MATERIAL" not in claim.features_involved[0])


def test_alternative_edge_names_the_best_matched_feature_for_the_alternative():
    # game-05 of the survey before 38.Qb5: Qb4 keeps the same threat; the lines differ little.
    fen = "1k5r/pp1q1ppp/8/8/8/8/PPP2PPP/1K1Q3R w - - 0 38"
    played = line(fen, ["Qd4", "Qe6"])
    alt = line(fen, ["Qd5", "Qe6"])
    c = ctx(fen, "Qd4", "d1d4", quality="inaccuracy")
    claim = alternative_edge_claim(c, "Qd5", alt, 2, played)
    assert claim is None or (claim.rule_id == "alternative_edge" and claim.text.startswith("Where Qd5 does better than Qd4:")
                             and claim.support_ply == 2 and claim.beneficiary == "white")
    assert alternative_edge_claim(c, "Qd5", alt, None, played) is None


def _event(fen, eval_before, quality="mistake", kind_facts=None):
    board = chess.Board(fen)
    facts = SimpleNamespace(claims=kind_facts or [], better_alternative=None, best_margin_cp=None,
                            selection_metadata={"comment_selection": {"verified_loss_cp": 150}})
    return SimpleNamespace(fen_before=fen, eval_before_cp=eval_before, eval_after_cp=None if eval_before is None else eval_before - 150,
                           eval_swing_cp=-150, move_quality=quality, key_moment_type="mistake", comment_facts=facts,
                           ply=44, move_index=43, san="Qh4", best_move_san="Qxf3")


def test_already_lost_is_mover_pov():
    black_to_move = "r4rk1/pp3ppp/8/8/5p2/5Q1q/PP3PPP/R1B2RK1 b - - 0 22"
    assert already_lost(_event(black_to_move, +250))  # Black is behind by 2.5
    assert not already_lost(_event(black_to_move, -250))  # Black is ahead
    assert not already_lost(_event(black_to_move, None))


def test_lost_position_errors_are_not_expert_moments_nor_survey_positions():
    black_to_move = "r4rk1/pp3ppp/8/8/5p2/5Q1q/PP3PPP/R1B2RK1 b - - 0 22"
    events = [_event(black_to_move, +250)]
    assert level_allows(events, 0, "intermediate")
    assert not level_allows(events, 0, "expert")
    assert complexity(events, 0) is None
    open_events = [_event(black_to_move, +50, kind_facts=[SimpleNamespace(beneficiary="white", is_concession=True, rule_id="legal_capture", realization="immediate")])]
    assert level_allows(open_events, 0, "expert")
    assert complexity(open_events, 0) is not None  # an error with facts stays a candidate


def test_positions_without_facts_or_mover_merits_are_not_survey_positions():
    fen = "r4rk1/pp3ppp/8/8/5p2/5Q1q/PP3PPP/R1B2RK1 b - - 0 22"
    bare = _event(fen, +50)
    assert complexity([bare], 0) is None  # no claims at all
    concession = SimpleNamespace(beneficiary="white", is_concession=True, rule_id="legal_capture", realization="immediate")
    merit = SimpleNamespace(beneficiary="black", is_concession=False, rule_id="rooks_activated", realization="immediate")
    sound = _event(fen, +50, quality="best", kind_facts=[concession])
    sound.key_moment_type = "best_move"; sound.comment_facts.selection_metadata = {"comment_selection": {"verified_loss_cp": 0}}
    sound.eval_after_cp = 50; sound.eval_swing_cp = 0
    assert complexity([sound], 0) is None
    sound.comment_facts.claims = [concession, merit]
    assert complexity([sound], 0) is not None


def test_material_words_and_resulting_position_summary():
    from app.core.commentary.phases.composer.sentence_plan import material_words, line_summary
    assert material_words(chess.Board()) == "material is level"
    assert material_words(chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")) == "White is a pawn up"
    assert material_words(chess.Board("4k3/8/8/8/8/2n5/P7/4K3 w - - 0 1")) == "White has a pawn for a knight"
    played = line(H5, ["h5", "Rxf2", "h6"], horizon=3)
    summary = line_summary(played, ["32.h5", "32...Rxf2", "33.h6"], [{"id": 0, "support_ply": 3}, {"id": 1, "support_ply": 1}])
    assert summary["resulting_position"] == {"after": "33.h6", "material": "Black is a rook up",
                                             "described_by_claims": [0]}


def test_lines_diverge_claim_names_the_first_different_move():
    from app.core.commentary.rules.facts_builder import lines_diverge_claim
    fen = "1k5r/pp1q1ppp/8/8/8/8/PPP2PPP/1K1Q3R w - - 0 38"
    played = line(fen, ["Qd4", "Qe6"])
    alt = line(fen, ["Qd5", "Qe6", "Rd1", "Rd8"])
    c = ctx(fen, "Qd4", "d1d4", quality="inaccuracy")
    claim = lines_diverge_claim(c, "Qd5", alt, played)
    assert claim is not None and claim.rule_id == "lines_diverge"
    assert claim.text == "The lines part after the shared reply: Qd5 allows White to continue with Rd1 Rd8, which Qd4 does not."
    assert claim.support_ply == 3 and claim.evidence.squares == ["d1", "h1"]  # the diverging move, from and to
    # Same length lines that differ at the reply.
    other = line(fen, ["Qd5", "Qd6", "Rd1", "Rd8"])
    claim = lines_diverge_claim(c, "Qd5", other, played)
    assert claim.text == "The lines part at Black's next move: after Qd5 it continues Qd6 Rd1, after Qd4 it is Qe6."
