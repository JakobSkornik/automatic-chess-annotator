import json
from pathlib import Path

import pytest

from app.research.runner import prepare, read, run, verify_manifest
from app.evaluation import parse_game


def prepared(tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"version": "test-v1", "cases": [{"id": "example",
        "pgn": "1. e4 e5 2. Nf3 Nc6 *", "required_comment_plies": [2]}]}))
    folder = tmp_path / "experiment"
    prepare(suite, folder)
    return folder


def test_prepare_removes_future_and_preserves_target(tmp_path):
    folder = prepared(tmp_path)
    case = read(folder / "manifest.json")["cases"][0]
    moves = list(parse_game(case["pgn"]).mainline_moves())
    assert [move.uci() for move in moves] == ["e2e4", "e7e5"]
    assert case["target_uci"] == "e7e5"
    assert case["split"] == "diagnostic"
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "suite.json", folder)


def test_manifest_mutation_is_rejected(tmp_path):
    folder = prepared(tmp_path)
    value = read(folder / "manifest.json")
    value["engine"]["depth"] = 2
    (folder / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest changed"):
        verify_manifest(folder)


def test_missing_snapshots_record_every_failed_run_and_resume(tmp_path):
    folder = prepared(tmp_path)
    result = run(folder)
    assert result["runs"] == result["failed"] == 4
    assert len(list((folder / "runs").glob("*/result.json"))) == 4
    assert run(folder, resume=True)["failed"] == 4
    artifact = next((folder / "runs").glob("*/result.json"))
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        run(folder, resume=True)


def test_retry_preserves_failed_attempts(tmp_path):
    folder = prepared(tmp_path)
    run(folder)
    original = {p: p.read_bytes() for p in (folder / "runs").glob("*/result.json")}
    assert run(folder, retry_failed=True)["failed"] == 4
    assert len(list((folder / "runs").glob("*/result.json"))) == 8
    assert all(p.read_bytes() == content for p, content in original.items())
    assert run(folder, resume=True)["runs"] == 4


def test_prepare_executable_study(tmp_path):
    from app.research.runner import ROOT
    folder = tmp_path / "pilot"
    result = prepare(None, folder, study_path=ROOT / "experiments/studies/diagnostic-pilot.json")
    manifest = verify_manifest(folder)
    assert result["scheduled_runs"] == 28
    assert manifest["rubric_version"] == "guid-ch5-rubric-v1"
    assert manifest["split_audit"]["passed"]
    assert manifest["rater_ids"] == ["evaluator-a", "evaluator-b"]


def test_fragment_export_does_not_turn_analysis_into_played_moves():
    from app.models.GameJson import GameJson, GameMetadata, GameMove, AnalysisInfo
    from app.core.io.pgn_writer import game_json_to_pgn
    import chess
    board = chess.Board()
    board.push_san("e4")
    game = GameJson(metadata=GameMetadata(id="fragment", white="W", black="B", result="*"),
        moves=[GameMove(mn=1, color="w", san="e4", uci="e2e4", fen=board.fen(),
            comment="An example: [pv:e4 e5 Nf3].")],
        analysis_info=AnalysisInfo(engine="test", depth=16, multipv=1, timestamp=0))
    exported = game_json_to_pgn(game)
    assert [m.uci() for m in parse_game(exported).mainline_moves()] == ["e2e4"]
    assert "Continuation: e5 Nf3" in exported


def test_equal_feature_advantages_have_stable_tie_order():
    from app.core.commentary.features.envisioned import diff_of_diffs
    from app.models.comment_facts import FeatureDiff, FeatureDelta
    alt = FeatureDiff(positive=[FeatureDelta(name=n, delta_cp=10, before_cp=0, after_cp=10) for n in ["Z", "A"]])
    result = diff_of_diffs(FeatureDiff(), alt)
    assert [a.name for a in result] == ["A", "Z"]
