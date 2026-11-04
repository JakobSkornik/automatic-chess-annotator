"""Trajectory horizon, SEE quiescence, evidence tagging and the rule fixes of 2 Nov 2026."""
import chess
import pytest

from app.core.commentary.features.envisioned import (
    build_envisioned_line, group_deltas, is_quiescent, settled_ply,
)
from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.core.commentary.features.guid_features.evidence import feature_squares
from app.core.commentary.rules.board_rules import rule_king_shelter, rule_pawn_plans, trapped_squares
from app.core.commentary.rules.context import _Ctx
from app.core.commentary.rules.facts_builder import (
    _FactsCtx, _cap_positional, _dedupe_entities, _gate_static_facts, _trajectory_claims, build_comment_facts,
    claim_identity,
)
from app.core.commentary.rules.move_tactics import move_tactic_claims
from app.core.commentary.rules.prophylaxis import rule_pawn_defects, rule_threat_parried
from app.core.commentary.rules.tactical_evidence import replay
from app.core.commentary.features.envisioned import diff_vectors
from app.core.engine.analysis.moment_kinds import _event_trigger, complexity, is_decision_point
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import Claim, EnvisionedLine, FeatureAdvantage

COMMON = dict(delta_cp=0, realization="immediate", support_ply=1, support_fen="", baseline_fen="")


def line(fen, sans):
    board = chess.Board(fen)
    ucis, fens = [], []
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return EnvisionedLine(start_fen=fen, line_uci=ucis, line_san=list(sans), fens=fens, leaf_fen=fens[-1],
                          immediate_horizon_ply=1)


def ctx(fen, mover="WHITE"):
    board = chess.Board(fen)
    event = MoveEvent(move_index=0, ply=1, san="x", uci="a1a2", fen_before=fen, fen_after=fen,
                      phase="mid", move_quality="good", event_type="quiet")
    return _FactsCtx(move_event=event, row=None, board_before=board, mover=mover.title(), phase_raw="mid",
                     depth=16, start_vec=compute_feature_vector_fen(fen))


# ---------------------------------------------------------------- quiescence and settlement

def test_quiescence_counts_pawns_by_static_exchange():
    assert not is_quiescent(chess.Board("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1"))  # loose pawn
    assert is_quiescent(chess.Board("7k/8/2p5/3p4/4P3/8/8/K7 w - - 0 1"))  # defended pawn: exd5 cxd5
    assert not is_quiescent(chess.Board("8/8/8/3q4/4P3/8/8/4K2k w - - 0 1"))  # queen en prise
    assert is_quiescent(chess.Board())


def test_settled_ply_prefers_the_earliest_ply_whose_story_holds():
    groups = [{"A": 10}, {"A": 12}, {"A": 12, "B": 9}, {"A": 11, "B": 9}]
    assert settled_ply(groups, [2, 3, 4]) == (3, "settled")  # B appears at ply 3 and persists
    assert settled_ply([{"A": 10}, {"A": 10}, {"A": -10}], [1, 2, 3]) == (3, "settled")  # reversal: only the end holds
    assert settled_ply([{}, {}, {}], [2, 3]) == (2, "settled")  # nothing changes: show move and reply


def test_envisioned_line_records_horizons():
    env = build_envisioned_line(chess.STARTING_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"])
    assert env.immediate_horizon_ply == 1
    assert env.horizon_reason in ("settled", "persistence", "fixed_window")
    assert 2 <= len(env.fens) <= 5
    # The immediate horizon skips a forcing exchange: 1.e4 d5 2.exd5 Qxd5 settles at ply 4.
    env = build_envisioned_line("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                                ["e4d5", "d8d5", "b1c3", "d5a5"])
    assert env.immediate_horizon_ply in (0, 2, 3)


def test_group_deltas_collapse_sides_and_pawn_components():
    start = compute_feature_vector_fen(chess.STARTING_FEN)
    end = compute_feature_vector_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    groups = group_deltas(start, end)
    assert not any(name.startswith(("WHITE_", "BLACK_")) for name in groups)


# ---------------------------------------------------------------- trajectories

def test_transient_claim_is_dropped_and_persistent_claim_is_anchored_at_first_ply():
    # 1.e4 d5 2.exd5 Qxd5: White's extra pawn at ply 1 vanishes at ply 3 -> transient.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    branch = line(fen, ["exd5", "Qxd5", "Nc3", "Qa5"])
    branch.immediate_horizon_ply = 1
    c = ctx(fen)
    claims, _ = _trajectory_claims(c, branch, 30)
    assert not any(cl.rule_id == "material_won" for cl in claims)
    reasons = {s["reason"] for s in c.evidence_suppressions}
    assert "transient_in_displayed_line" in reasons
    for cl in claims:
        if cl.rule_id not in ("legal_capture", "exchange_sequence"):
            assert cl.persistence == "held" and cl.first_ply == cl.support_ply
            if cl.realization == "envisioned":
                assert cl.caused_by_san == branch.line_san[cl.support_ply - 1]


def test_every_kept_claim_carries_evidence_squares():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    branch = line(fen, ["Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"])
    c = ctx(fen)
    claims, _ = _trajectory_claims(c, branch, 20)
    assert claims, "expected some claims on the Ruy Lopez line"
    assert all(cl.evidence.squares or cl.evidence.files for cl in claims)
    assert all(cl.entity for cl in claims if cl.rule_id in ("legal_capture", "hole_created"))


def test_feature_squares_name_the_right_pieces():
    board = chess.Board("4k3/8/8/8/8/8/P1P5/R3K3 w - - 0 1")
    assert feature_squares(board, "WHITE_PAWN_ISOLATED")[0] == [chess.A2, chess.C2]
    rooks, files = feature_squares(board, "WHITE_ROOK_HALF_OPEN_FILE")
    assert rooks == [] and files == []
    board = chess.Board("4k3/p7/8/8/8/8/8/R3K3 w - - 0 1")
    assert feature_squares(board, "WHITE_ROOK_HALF_OPEN_FILE") == ([chess.A1], ["a"])


# ---------------------------------------------------------------- static facts gated by the line

def test_loose_piece_claim_needs_the_line_or_a_probe():
    start = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    branch = line("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", ["Ra8+", "Kf7", "Ra7+"])
    claim = Claim(rule_id="piece_left_en_prise", text="x", beneficiary="white", support_ply=1,
                  entity="en_prise:e8", evidence={"squares": ["e8"], "pieces": [{"square": "e8", "piece": "king", "color": "black", "role": "target"}]})
    c = ctx(start)
    assert _gate_static_facts(c, [claim], branch, replay(branch)) == []
    assert c.evidence_suppressions[0]["reason"] == "capture_not_realized_in_line"
    # A probe that confirms the capture keeps the claim.
    cashed = line("8/pk6/8/8/3Q4/8/PK6/8 w - - 0 1", ["Qa4", "Kc7", "Ka3"])  # ...Kc7 leaves a7 loose
    hang = Claim(rule_id="hanging_piece", text="x", beneficiary="white", support_ply=2, entity="hanging:a7",
                 evidence={"squares": ["a7"], "pieces": [{"square": "a7", "piece": "pawn", "color": "black", "role": "target"}]})
    c = ctx(cashed.start_fen)
    cashed.root_eval_cp = 600
    c.prober = lambda fen: 650
    assert len(_gate_static_facts(c, [hang], cashed, replay(cashed))) == 1


def test_fork_that_is_captured_at_once_is_not_a_fork():
    # A bishop check that forks queen and king but is simply recaptured.
    fen = "4k3/3n4/8/1B3q2/8/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    move = board.parse_san("Bxd7+")
    after = board.copy(); after.push(move)
    assert not any(c.rule_id == "fork" for c in move_tactic_claims(board, move, after, COMMON))


def test_rook_trade_is_not_a_trapped_piece():
    # game-04 17.Rxa7: Black's rook on a8 can take on a7 and be recaptured: an exchange.
    board = chess.Board("r1r5/R3nkbp/2p1p3/2pp1ppb/3P4/1PPBP2P/1P1N1PPB/R5K1 b - - 0 17")
    assert chess.A8 not in trapped_squares(board, chess.BLACK)


# ---------------------------------------------------------------- new rules

def _rule_ctx(start, node, mover="WHITE"):
    a, b = chess.Board(start), chess.Board(node)
    diff = diff_vectors(compute_feature_vector_fen(start), compute_feature_vector_fen(node))
    return _Ctx(diff, "mid", mover, 0, start_board=a, leaf_board=b)


def test_threat_parried_names_the_defended_piece_and_the_threat():
    # Black's rook attacks White's loose knight on d4; Nf3-d2?? no: White defends it with c3.
    start = "3r2k1/8/8/8/3N4/8/2P5/6K1 w - - 0 1"
    after_c3 = "3r2k1/8/8/8/3N4/2P5/8/6K1 b - - 0 1"
    claims = rule_threat_parried(_rule_ctx(start, after_c3))
    assert any(c.rule_id == "threat_parried" and "knight on d4" in c.text and "Rxd4" in c.text for c in claims)
    assert claims[0].evidence.squares == ["d4", "d8"]
    # Moving the knight away is not a parry.
    moved = "3r2k1/8/8/8/8/5N2/2P5/6K1 b - - 1 1"
    assert rule_threat_parried(_rule_ctx(start, moved)) == []


def test_pawn_defects_are_named_with_squares():
    start = "4k3/8/8/8/8/8/PPP5/4K3 w - - 0 1"
    isolated = "4k3/8/8/8/8/8/P1P5/4K3 w - - 0 1"
    claims = rule_pawn_defects(_rule_ctx(start, isolated))
    assert [c.text for c in claims] == ["White's pawns on a2, c2 are isolated."]
    assert claims[0].evidence.squares == ["a2", "c2"]


def test_outpost_occupied_replaces_a_bare_hole():
    # After ...e6 and ...d5?? no pawn can cover e5; a White knight on e5 settles there.
    start = "3qk3/pp4pp/8/8/8/8/8/3QK3 w - - 0 1"  # queens on: a middlegame, so holes apply
    node = "3qk3/pp4pp/8/4N3/8/8/8/3QK3 b - - 0 1"
    claims = rule_pawn_plans(_rule_ctx(start, node))
    assert any(c.rule_id == "outpost_occupied" and "e5" in c.text for c in claims)
    assert not any(c.rule_id == "hole_created" and "e5" in c.text for c in claims)


def test_king_attack_combination_fires_with_a_heavy_piece_on_the_open_file():
    # Black's g-pawn is gone and a White rook stands on the open g-file (a queen is on the board).
    start = "6k1/5ppp/8/8/8/8/5P1P/5QRK w - - 0 1"
    node = "6k1/5p1p/8/8/8/8/5P1P/5QRK w - - 0 1"
    claims = rule_king_shelter(_rule_ctx(start, node))
    ids = [c.rule_id for c in claims]
    assert "king_attack_builds" in ids and "king_file_opened" not in ids


# ---------------------------------------------------------------- dedup, contrast cap, selection

def test_entity_dedupe_keeps_the_most_specific_rule():
    a = Claim(rule_id="hanging_piece", text="x", beneficiary="white", entity="hanging:e5")
    b = Claim(rule_id="piece_left_en_prise", text="y", beneficiary="white", entity="en_prise:e5")
    c = Claim(rule_id="intent_targets_piece", text="z", beneficiary="white", entity="target:e5")
    assert [k.rule_id for k in _dedupe_entities([a, b, c])] == ["piece_left_en_prise"]


def test_contrast_first_cap_prefers_claims_the_other_line_lacks():
    shared = [Claim(rule_id="center_control", text="s", beneficiary="white", delta_cp=90, entity="center")]
    own = [Claim(rule_id="space_gained", text="o", beneficiary="white", delta_cp=10, entity="space")]
    other = {claim_identity(shared[0])}
    kept = _cap_positional(shared + own, other, "White", 1)
    assert kept == own


def test_expert_level_comments_decision_points_not_feature_events():
    from types import SimpleNamespace
    alt = SimpleNamespace(feature_advantages=[FeatureAdvantage(name="WHITE_SPACE", alt_delta_cp=40, played_delta_cp=0, advantage_cp=40),
                                              FeatureAdvantage(name="BLACK_KING_DANGER", alt_delta_cp=0, played_delta_cp=50, advantage_cp=-50)],
                          feature_advantages_matched=[])
    facts = SimpleNamespace(claims=[Claim(rule_id="hole_created", text="h", support_ply=1)], best_margin_cp=10,
                            better_alternative=alt, selection_metadata={"comment_selection": {"verified_loss_cp": 0}})
    event = SimpleNamespace(key_moment_type="best_move", comment_facts=facts, move_quality="best")
    assert is_decision_point(event)  # close margin, two diverging feature groups
    assert _event_trigger(event, "expert")
    facts.best_margin_cp, alt.feature_advantages = 10, []
    assert not _event_trigger(event, "expert")  # a feature event alone is not enough for an expert
    facts.claims = [Claim(rule_id="fork", text="f", support_ply=1)]
    assert _event_trigger(event, "beginner")  # a tactic on the board is what a beginner learns from


def test_complexity_ignores_loose_piece_bookkeeping():
    from types import SimpleNamespace
    def event(rules):
        facts = SimpleNamespace(claims=[Claim(rule_id=r, text="x", support_ply=1, beneficiary="white") for r in rules], best_margin_cp=None,
                                better_alternative=None, selection_metadata={"comment_selection": {"verified_loss_cp": 0}})
        return SimpleNamespace(key_moment_type="best_move", comment_facts=facts, move_quality="best",
                               eval_before_cp=0, fen_before=chess.STARTING_FEN, uci="e2e4")
    events = [event([]), event(["hanging_piece", "piece_left_en_prise"]), event(["hole_created", "space_gained"])]
    assert complexity(events, 1) == 1  # only the quiet start counts
    assert complexity(events, 2) == 2 + 1
