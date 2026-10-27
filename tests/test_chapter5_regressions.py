import io
import json

import chess
import chess.engine
import chess.pgn
import pytest

from app.core.commentary.features.envisioned import build_envisioned_line, is_quiescent, trim_envisioned_line_by_probe
from app.core.commentary.features.refutation_scan import RefutationScanner
from app.core.commentary.phases.composer import _audience_view, compose_facts_comment, render_facts_template, validate_facts_comment
from app.core.commentary.phases.composer.grounded import render_editor_response
from app.core.commentary.rules.facts_builder import build_comment_facts
from app.core.engine.analysis_retriever import _requires_tactical_analysis
from app.core.io.pgn_writer import game_json_to_pgn
from app.models.chess_events import AnalyzedMoveData, MoveEvent, MoveQuality
from app.models.comment_facts import Claim
from app.models.GameJson import AnalysisInfo, GameJson, GameMetadata, GameMove
from app.evaluation import aggregate, audit_game


def facts_for(fen=chess.STARTING_FEN, sans=("e4", "d6", "Bb5+", "Bd7", "Bxd7+", "Qxd7", "Nf3")):
    board = chess.Board(fen)
    moves = []
    after = None
    for san in sans:
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
        if after is None:
            after = board.fen()
    terminal = chess.Board(after)
    cp = (1000000 if terminal.turn == chess.BLACK else -1000000) if terminal.is_checkmate() else 0
    event = MoveEvent(move_index=0, ply=1, san=sans[0], uci=moves[0], fen_before=fen,
                      fen_after=after, phase="middlegame", eval_before_cp=0, eval_after_cp=cp,
                      move_quality=MoveQuality.BEST, event_type="quiet")
    row = AnalyzedMoveData(index=0, ply=1, san=sans[0], uci=moves[0], fen_before=fen,
        fen_after=after, score_cp=cp, phase_raw="mid", hidden_features={"_engine": {"after_pv_uci": moves[1:]}})
    return build_comment_facts(row, event)


@pytest.mark.parametrize("fen,san,winner,result", [
    ("7k/8/5KQ1/8/8/8/8/8 w - - 0 37", "Qg7#", "White", "1-0"),
    ("8/8/8/8/8/5kq1/8/7K b - - 0 37", "Qg2#", "Black", "0-1"),
])
def test_terminal_mate_keeps_winner_and_move_number(fen, san, winner, result):
    facts = facts_for(fen, (san,))
    text = render_facts_template(facts)
    assert facts.terminal_result == result
    assert facts.eval_mate is None
    assert f"checkmate for {winner}" in text
    assert "#0" not in text
    assert f"37{'.' if winner == 'White' else '...'}{san}" in text


def test_full_mate_sequence_is_not_trimmed_to_a_check():
    board = chess.Board("4k3/3n1p2/8/6B1/8/1Q6/8/3R2K1 w - - 0 16")
    # Use the actual Opera position from the checked-in fixture.
    from pathlib import Path
    suite = json.loads((Path(__file__).parents[1] / "benchmarks/chapter5-v1.json").read_text())
    game = chess.pgn.read_game(io.StringIO(suite["cases"][0]["pgn"]))
    board = game.board()
    moves = list(game.mainline_moves())
    for m in moves[:30]:
        board.push(m)
    line = build_envisioned_line(board.fen(), [m.uci() for m in moves[30:]])
    assert line.line_san == ["Qb8+", "Nxb8", "Rd8#"]
    assert chess.Board(line.leaf_fen).is_checkmate()


@pytest.mark.parametrize("level", ["beginner", "intermediate", "expert"])
def test_audience_preserves_evidence_and_quietness(level):
    facts = facts_for()
    view = _audience_view(facts, level)
    assert view.display_line == facts.display_line
    assert not chess.Board(view.display_line.leaf_fen).is_check()
    for claim in facts.claims:
        assert claim.support_fen == facts.display_line.fens[claim.support_ply - 1]
        assert claim.baseline_fen == facts.display_line.start_fen
        assert claim.realization == ("immediate" if claim.support_ply == 1 else "envisioned")


def test_probe_cannot_mark_check_as_quiet():
    line = facts_for().display_line.model_copy(update={"root_eval_cp": 300})
    trimmed = trim_envisioned_line_by_probe(line, lambda f: -300 if f in line.fens[5:] else 300)
    assert trimmed.leaf_quiescent == is_quiescent(chess.Board(trimmed.leaf_fen))
    assert not chess.Board(trimmed.leaf_fen).is_check()
    assert trimmed.line_san[-1] == "Bd7"  # skip the later Bxd7+ checkpoint


def test_probe_abstains_when_no_agreeing_quiet_endpoint():
    line = facts_for().display_line.model_copy(update={"root_eval_cp": 300})
    result = trim_envisioned_line_by_probe(line, lambda _: -300)
    assert not result.evidence_valid
    assert result.line_uci == line.line_uci


def test_nonquiet_start_does_not_produce_strategic_claims():
    facts = facts_for("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", ("Kxe2",))
    assert not facts.display_line.start_quiescent
    assert facts.claims == []


@pytest.mark.parametrize("white", [True, False])
def test_stable_winning_move_is_not_refuted(white):
    board = chess.Board()
    if not white:
        board.push_san("e4")
    sign = 1 if white else -1
    roots = list(board.legal_moves)[:2]
    class Engine:
        def analyse(self, b, depth, multiPv):
            if depth == 8:
                return [{"score": chess.engine.PovScore(chess.engine.Cp(sign * v), chess.WHITE), "pv": [m]} for m, v in zip(roots, [400, 390])]
            return [{"score": chess.engine.PovScore(chess.engine.Cp(sign * 410), chess.WHITE)}]
    assert RefutationScanner(Engine()).scan(board.fen(), 410) == []


def test_negative_shallow_candidates_are_ranked_relative_to_best():
    roots = list(chess.Board().legal_moves)[:2]
    class Engine:
        def analyse(self, b, depth, multiPv):
            if depth == 8:
                return [{"score": chess.engine.PovScore(chess.engine.Cp(v), chess.WHITE), "pv": [m]} for m, v in zip(roots, [-20, -30])]
            return [{"score": chess.engine.PovScore(chess.engine.Cp(-300), chess.WHITE)}]
    assert len(RefutationScanner(Engine()).scan(chess.STARTING_FEN, -300)) == 2


def test_book_mating_blunder_forces_analysis():
    board = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5"]:
        board.push_san(san)
    before = board.copy()
    board.push_san("Nf6")
    assert _requires_tactical_analysis(before, board)
    start = chess.Board()
    after = start.copy()
    after.push_san("e4")
    assert not _requires_tactical_analysis(start, after)


def test_unsupported_material_prose_is_rejected():
    facts = facts_for()
    assert not validate_facts_comment(render_facts_template(facts) + " White has an extra queen.", facts)
    with pytest.raises(ValueError):
        render_editor_response('{"text":"White has an extra queen."}', facts, "intermediate", "neutral")


def test_editor_validates_ids_and_preserves_claim_owner():
    facts = facts_for().model_copy(update={"claims": [Claim(rule_id="piece_pinned", text="Black pins a white piece.",
        text_state="A white piece is pinned by Black.", beneficiary="black", is_concession=True)]})
    text = render_editor_response('{"main":[{"id":0,"form":"state"}],"alternative":[]}', facts, "intermediate", "neutral")
    assert "A white piece is pinned by Black" in text
    for bad in ['{"main":[],"alternative":[]}', '{"main":[{"id":3,"form":"state"}],"alternative":[]}']:
        with pytest.raises(ValueError):
            render_editor_response(bad, facts, "intermediate", "neutral")


@pytest.mark.asyncio
async def test_prose_provider_response_falls_back_safely(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    class Service:
        class _provider:
            @staticmethod
            def is_configured():
                return True
        async def _llm_call_json_schema(self, *args, **kwargs):
            return '{"text":"White has an extra queen."}'
    facts = facts_for()
    result = await compose_facts_comment(Service(), facts, model=None, effort="low")
    assert result["rendering"] == "template"
    assert not result["contract_ok"]
    assert "extra queen" not in result["text"]


def test_custom_start_roundtrip_and_corruption_audit():
    fen = "8/8/8/8/8/5kq1/8/7K b - - 0 37"
    b = chess.Board(fen)
    m = b.parse_san("Qg2#")
    b.push(m)
    facts = facts_for(fen, ("Qg2#",))
    from app.core.engine.analysis.serialization import _facts_to_json
    game = GameJson(metadata=GameMetadata(id="test", white="W", black="B", result="0-1", initial_fen=fen),
        moves=[GameMove(mn=37, color="b", san="Qg2#", uci=m.uci(), fen=b.fen(),
            comment=render_facts_template(facts), final_comment=True, comment_facts=_facts_to_json(facts))],
        commentary_complete=True, analysis_info=AnalysisInfo(engine="test", depth=16, multipv=1, timestamp=0))
    pgn = game_json_to_pgn(game)
    report = audit_game(pgn, game.model_dump(), pgn, [1])
    assert report["issues"] == []
    bad = game.model_dump()
    bad["moves"][0]["comment_facts"]["terminal_result"] = "1-0"
    report = audit_game(pgn, bad, pgn, [1])
    assert report["checks"]["terminal_result"]["passed"] == 0
    assert aggregate([{"audit": report}])["checks"]["terminal_result"]["rate"] == 0
    bad["moves"][0]["score"] = {"mate": 0, "cp": None}
    report = audit_game(pgn, bad, pgn, [1])
    assert report["checks"]["terminal_score"]["passed"] == 0


def test_export_removes_empty_pv_parentheses():
    from app.core.io.pgn_writer import _flatten_tokens
    text, lines = _flatten_tokens("White keeps an edge ([pv:e4 e5]).")
    assert text == "White keeps an edge."
    assert lines == [["e4", "e5"]]


def test_custom_black_start_mover_ignores_ply_parity():
    from types import SimpleNamespace
    from app.core.commentary.key_moment_detector import _mover_is_white
    board = chess.Board("8/8/8/8/8/5kq1/8/7K b - - 0 37")
    board.push_san("Qg2#")
    assert not _mover_is_white(SimpleNamespace(position=board.fen(), depth=1))


def test_benchmark_rejects_multigame_input_and_unknown_case():
    from pathlib import Path
    from app.evaluation import parse_game
    from scripts.benchmark_annotations import load_suite
    with pytest.raises(ValueError):
        parse_game("1. e4 *\n\n1. d4 *")
    with pytest.raises(ValueError):
        load_suite(Path(__file__).parents[1] / "benchmarks/chapter5-v1.json", ["missing"])


def test_rating_aggregation_rejects_duplicate_and_missing_scores(tmp_path):
    from scripts.benchmark_annotations import score_ratings
    key = tmp_path / "key.csv"
    key.write_text("sample_id,mode\nsample,template\n", encoding="utf-8")
    ratings = tmp_path / "ratings.csv"
    header = "sample_id,rater_id,correctness,relevance,usefulness,concision\n"
    row = "sample,expert1,5,4,3,2\n"
    ratings.write_text(header + row, encoding="utf-8")
    assert score_ratings(ratings, key)["template"]["correctness"] == {"n": 1, "mean": 5}
    for body in (row + row, "sample,expert1,,4,3,2\n"):
        ratings.write_text(header + body, encoding="utf-8")
        with pytest.raises(ValueError):
            score_ratings(ratings, key)
