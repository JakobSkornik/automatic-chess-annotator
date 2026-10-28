"""Writer evidence added in grounded_narrative_v4: severity, uniqueness, material trade, stop point."""
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


def test_blunder_carries_grade_size_and_line_trade():
    evidence = writer_view(qxb5_facts(), key_moment_type="blunder")
    assessment = evidence["assessment"]
    assert assessment["classification"] == "blunder"
    assert assessment["key_moment"] == "blunder"
    assert assessment["change_for_mover"] == {"direction": "loses", "size": "very large"}
    assert assessment["alternative"]["loss_by_played_move"] == "very large"
    assert "eval_cp_white" not in assessment  # numbers stay hidden from the writer
    trade = evidence["material"]["main"]
    assert trade["black_gives_up"] == "a queen" and trade["white_gives_up"] == "a knight"
    assert trade["balance_before"] == "Black has a rook and a pawn for a knight."
    assert trade["balance_after"] == "White has a queen for a rook and a pawn."
    assert evidence["material"]["alternative"]["white_gives_up"] == "nothing"


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


def test_prompt_asks_for_about_150_words_and_gates_uniqueness():
    assert "about 150 words" in narrative.SYSTEM and "never more than 250" in narrative.SYSTEM
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
    events = [SimpleNamespace(move_index=i, key_moment_type="blunder", is_critical=False) for i in range(8)]
    metadata = SimpleNamespace(id="game")
    metadata.model_copy = lambda update: metadata
    state = SimpleNamespace(metadata=metadata, move_events=events, context=SimpleNamespace())
    monkeypatch.setattr(game_pipeline, "_promote_turning_points", lambda *a: None)
    monkeypatch.setattr(game_pipeline, "_apply_back_to_back_key_moment_suppression", lambda *a: None)
    monkeypatch.setattr(game_pipeline, "_key_moment_order", lambda *a: [7, 5, 3, 1])
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
