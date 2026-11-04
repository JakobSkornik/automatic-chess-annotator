"""Writer evidence added in grounded_narrative_v4/v5: severity, uniqueness, material trade, stop point."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import chess
import pytest

from app.core.commentary.phases.composer import narrative
from app.core.commentary.pipeline import game_pipeline
from app.core.commentary.rules.constants import _MATE_SCORE
from app.core.commentary.rules.facts_builder import ONLY_WIN_MARGIN_CP, _best_margin_cp
from app.models.comment_facts import BestAlternative, CommentFacts, EnvisionedLine

# game-03 of the survey, before 28...Qxb5: Black is up the exchange and a pawn.
QXB5 = "2r2rk1/1p3p1p/4p1p1/qN1pP3/8/2P1R2P/3KQPP1/8 b - - 0 28"


def line(fen, sans):
    board = chess.Board(fen)
    ucis, fens = [], []
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return EnvisionedLine(start_fen=fen, line_uci=ucis, line_san=list(sans), fens=fens, leaf_fen=board.fen())


def qxb5_facts(**update):
    alt = BestAlternative(san="Qa2+", uci="a5a2", eval_cp=-200, display_line=line(QXB5, ["Qa2+"]))
    facts = CommentFacts(ply=56, san="Qxb5", uci="a5b5", mover="Black", phase="mid",
                         verdict="throws away the advantage", eval_cp=500, eval_before_cp=-150,
                         move_quality="blunder", display_line=line(QXB5, ["Qxb5", "Qxb5"]),
                         better_alternative=alt)
    return facts.model_copy(update=update)


def writer_view(facts, key_moment_type=None):
    user, _evidence = narrative.prompt(facts, "intermediate", key_moment_type=key_moment_type)
    return json.loads(user)["evidence"]


def test_blunder_carries_grade_and_size_without_material_standing():
    evidence = writer_view(qxb5_facts(), key_moment_type="blunder")
    assessment = evidence["assessment"]
    assert assessment["classification"] == "blunder"
    assert assessment["key_moment"] == "blunder"
    assert assessment["change_for_mover"] == {"direction": "loses", "size": "very large"}
    assert assessment["alternative"]["loss_by_played_move"] == "very large"
    assert "eval_cp_white" not in assessment  # numbers stay hidden from the writer
    assert "material" not in evidence


@pytest.mark.parametrize("margin, inferior, played_unique, alternative_unique", [
    (150, True, True, False),   # played the engine's clearly best move
    (60, True, False, False),   # best, but another move is close
    (150, False, False, True),  # missed the only good move
    (None, True, False, False),
])
def test_unique_best_follows_engine_margin(margin, inferior, played_unique, alternative_unique):
    facts = qxb5_facts(best_margin_cp=margin)
    facts = facts.model_copy(update={"better_alternative": facts.better_alternative.model_copy(
        update={"is_inferior": inferior})})
    assessment = writer_view(facts)["assessment"]
    assert assessment["unique_best"] is played_unique
    assert assessment["alternative"]["unique_best"] is alternative_unique


def test_quiet_move_holds():
    facts = qxb5_facts(eval_cp=-140, eval_before_cp=-150, move_quality="best")
    assert writer_view(facts)["assessment"]["change_for_mover"] == {"direction": "holds", "size": "none"}


def test_prompt_asks_for_60_to_120_words_and_gates_uniqueness():
    assert "usually 60-120" in narrative.SYSTEM and "never more than 200" in narrative.SYSTEM
    assert "Every comment says why" in narrative.SYSTEM and "name its side" in narrative.SYSTEM
    assert "unique_best" in narrative.SYSTEM and "unique_best" in narrative.REVIEW_SYSTEM
    assert narrative.MAX_WORDS == 250


def pv(score):
    return [SimpleNamespace(score=score, move="e2e4")]


def test_best_margin_is_mover_pov_and_handles_mates():
    assert _best_margin_cp([pv(120), pv(10)], "White") == 110
    assert _best_margin_cp([pv(-120), pv(-10)], "Black") == 110
    assert _best_margin_cp([pv(_MATE_SCORE - 3), pv(400)], "White") == ONLY_WIN_MARGIN_CP
    assert _best_margin_cp([pv(_MATE_SCORE - 3), pv(_MATE_SCORE - 5)], "White") is None
    assert _best_margin_cp([pv(50)], "White") is None


def test_stop_at_move_index_ends_reverse_sweep(monkeypatch):
    events = [SimpleNamespace(move_index=i, ply=i + 1, key_moment_type="blunder", is_critical=False) for i in range(8)]
    metadata = SimpleNamespace(id="game")
    metadata.model_copy = lambda update: metadata
    state = SimpleNamespace(metadata=metadata, move_events=events, context=SimpleNamespace())
    monkeypatch.setattr(game_pipeline, "_promote_turning_points", lambda *a: None)
    monkeypatch.setattr(game_pipeline, "_apply_back_to_back_key_moment_suppression", lambda *a: None)
    monkeypatch.setattr(game_pipeline, "_coverage_order", lambda *a: ([7, 5, 3, 1], set()))
    commented = []

    async def comment(self, run, mi, pos, n_total):
        commented.append(mi)
    monkeypatch.setattr(game_pipeline.GameAnnotationPipeline, "_comment_one_key_moment", comment)

    class Service:
        @asynccontextmanager
        async def commentary_session(self, game_id):
            yield self

    asyncio.run(game_pipeline.GameAnnotationPipeline().run_llm_phases(state, Service(), stop_at_move_index=3))
    assert commented == [7, 5, 3]
    assert state.llm_done


def test_pin_is_not_reported_when_the_pinned_piece_takes_the_pinner():
    from app.core.commentary.rules.tactical_evidence import absolute_pins, tactical_claims
    # game-05 of the survey, 34.Qd4 Qxd4 cxd4: the c3 pawn "pinned" by the queen recaptures it.
    start = "5r1r/ppk2pQp/3b4/3q4/8/2PP4/PK2pNPP/1R5R w - - 0 34"
    board = chess.Board(start)
    for san in ("Qd4", "Qxd4"):
        board.push_san(san)
    assert not absolute_pins(board, chess.BLACK)  # the pawn attacks d4
    assert not [c for c in tactical_claims(line(start, ["Qd4", "Qxd4", "cxd4"])) if c.rule_id == "absolute_pin"]


def test_real_pin_is_still_reported():
    from app.core.commentary.rules.tactical_evidence import absolute_pins
    board = chess.Board("6k1/6b1/8/8/8/2P5/8/K7 w - - 0 1")  # bishop g7, pawn c3, king a1
    assert absolute_pins(board, chess.BLACK) == {(chess.G7, chess.C3, chess.A1)}


def test_queen_pinned_by_a_defended_bishop_is_still_pinned():
    from app.core.commentary.rules.tactical_evidence import absolute_pins
    # Bb5 pins the queen on d7 to the king on e8; Qxb5 would lose the queen to axb5.
    board = chess.Board("4k3/3q4/8/1B6/P7/8/8/4K3 b - - 0 1")
    assert absolute_pins(board, chess.WHITE) == {(chess.B5, chess.D7, chess.E8)}


def test_attacked_queen_that_can_take_the_attacker_is_not_hanging():
    from app.core.commentary.features.guid_features.features.threats.hanging import is_concretely_hanging
    board = chess.Board("5r1r/ppk2pQp/3b4/3q4/8/2PP4/PK2pNPP/1R5R w - - 0 34")
    board.push_san("Qd4")  # game-05: the d5 queen is attacked but simply takes on d4
    assert not is_concretely_hanging(board, chess.D5, chess.WHITE)
    # Taking the attacker would lose material (knight defended by b2): still hanging.
    defended_knight = chess.Board("4k3/8/8/3q4/8/2N5/1P6/4K3 b - - 0 1")
    assert is_concretely_hanging(defended_knight, chess.D5, chess.WHITE)
    # The queen takes the undefended rook attacking it: not hanging.
    loose_rook = chess.Board("4k3/8/8/3q4/8/8/8/3R3K b - - 0 1")
    assert not is_concretely_hanging(loose_rook, chess.D5, chess.WHITE)


def test_back_rank_needs_an_enemy_heavy_piece_and_no_guard():
    from app.core.commentary.rules.piece_rules import _back_rank_exposed
    assert _back_rank_exposed(chess.Board("4r1k1/8/8/8/8/8/5PPP/6K1 w - - 0 1"), chess.WHITE)
    assert not _back_rank_exposed(chess.Board("4b1k1/8/8/8/8/8/5PPP/6K1 w - - 0 1"), chess.WHITE)  # no rook/queen
    assert not _back_rank_exposed(chess.Board("4r1k1/8/8/8/8/8/5PPP/R5K1 w - - 0 1"), chess.WHITE)  # rook guards


def test_feature_comparison_uses_plain_group_names():
    from app.models.comment_facts import FeatureAdvantage
    facts = qxb5_facts()
    advantages = [FeatureAdvantage(name="W_BISHOP_PLUS_PAWNS_ON_COLOR", alt_delta_cp=1, played_delta_cp=16, advantage_cp=-15),
                  FeatureAdvantage(name="BLACK_KING_DANGER", alt_delta_cp=0, played_delta_cp=264, advantage_cp=-264),
                  FeatureAdvantage(name="UNMAPPED_TERM", alt_delta_cp=0, played_delta_cp=50, advantage_cp=50)]
    facts = facts.model_copy(update={"better_alternative": facts.better_alternative.model_copy(
        update={"feature_advantages": advantages})})
    comparison = writer_view(facts)["assessment"]["feature_comparison"]
    # Black moves, so White-POV negative advantages favour the alternative for Black.
    assert comparison == {"favours_played_move": [], "favours_alternative": ["pawn structure", "king safety"],
                          "compared_at": "the settled end of each line"}


def test_prompt_focuses_on_features_not_lines_or_material_standing():
    assert "one practical lesson" in narrative.SYSTEM and "simple_error" in narrative.SYSTEM
    assert "Explain the move through its features" in narrative.SYSTEM
    assert "never describe the material standing" in narrative.SYSTEM and "never retell the line" in narrative.SYSTEM


def test_static_threat_is_kept_only_when_the_line_realizes_it():
    from app.core.commentary.rules.facts_builder import _FactsCtx, _gate_static_facts
    from app.core.commentary.rules.tactical_evidence import replay
    from app.core.commentary.features.guid_features.evidence import evidence_for
    from app.models.comment_facts import Claim
    from app.models.chess_events import MoveEvent
    # game-05: after 34.Qd4 the a7 pawn is attacked, but the queens are traded at once.
    start = "5r1r/ppk2pQp/3b4/3q4/8/2PP4/PK2pNPP/1R5R w - - 0 34"
    traded = line(start, ["Qd4", "Qxd4", "cxd4", "Bb4", "Ne4"])
    board = chess.Board(start)

    def claim(branch):
        node = chess.Board(branch.fens[0])
        return Claim(rule_id="hanging_piece", text="Black's pawn on a7 is undefended and attacked.",
                     beneficiary="white", support_ply=1, support_fen=branch.fens[0], baseline_fen=start,
                     entity="hanging:a7", evidence=evidence_for(node, [chess.A7, chess.D4], roles={chess.A7: "target"}))

    def ctx(branch):
        event = MoveEvent(move_index=0, ply=67, san="Qd4", uci="g7d4", fen_before=start, fen_after=branch.fens[0],
                          phase="mid", move_quality="good", event_type="quiet")
        return _FactsCtx(move_event=event, row=None, board_before=board, mover="White", phase_raw="mid",
                         depth=16, start_vec={})

    c = ctx(traded)
    assert _gate_static_facts(c, [claim(traded)], traded, replay(traded)) == []
    assert c.evidence_suppressions[0]["reason"] == "capture_not_realized_in_line"
    # The same fact survives when the line takes the pawn.
    cashed = line("8/pk6/8/8/3Q4/8/PK6/8 w - - 0 1", ["Qd7+", "Kb6", "Qxa7+"])
    d7 = Claim(rule_id="hanging_piece", text="Black's pawn on a7 is undefended and attacked.",
               beneficiary="white", support_ply=1, support_fen=cashed.fens[0], baseline_fen=cashed.start_fen,
               entity="hanging:a7", evidence=evidence_for(chess.Board(cashed.fens[0]), [chess.A7], roles={chess.A7: "target"}))
    c2 = ctx(cashed)
    assert len(_gate_static_facts(c2, [d7], cashed, replay(cashed))) == 1


def test_clearly_worse_runner_up_stands_for_the_other_moves():
    worse = qxb5_facts()
    worse = worse.model_copy(update={"better_alternative": worse.better_alternative.model_copy(
        update={"is_inferior": True, "eval_cp": 800})})  # runner-up 3 pawns worse for Black
    assert writer_view(worse)["assessment"]["alternative"]["role"] == "runner_up_clearly_worse"
    assert writer_view(qxb5_facts())["assessment"]["alternative"]["role"] == "better_move"
    assert "runner_up_clearly_worse" in narrative.SYSTEM
