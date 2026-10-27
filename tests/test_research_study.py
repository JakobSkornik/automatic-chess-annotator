import json
from pathlib import Path

import chess
import chess.pgn
import pytest

from app.evaluation import parse_game
from app.research.study import load_study


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "experiments" / "studies" / "diagnostic-pilot.json"


def _variant_matrix():
    return [
        {"id": "B00", "horizon": "current_quiet_bounded", "claims": "current", "renderer": "template"},
        {"id": "B10", "horizon": "adaptive_feature", "claims": "current", "renderer": "template"},
        {"id": "B01", "horizon": "current_quiet_bounded", "claims": "contrastive", "renderer": "template"},
        {"id": "B11", "horizon": "adaptive_feature", "claims": "contrastive", "renderer": "template"},
    ]


def _case(case_id, pgn, *, split="development", source=None, family=None, target_ply=1):
    return {
        "id": case_id,
        "pgn": pgn,
        "target_ply": target_ply,
        "source_group": source or f"source-{case_id}",
        "family_group": family or f"family-{case_id}",
        "split": split,
        "category": "quiet_maneuver",
        "teaching_target": True,
        "provenance": {"collection": "unit-test"},
    }


def _write_study(tmp_path, cases, **updates):
    corpus = {"version": "unit-v1", "frozen": False, "cases": cases}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    config = {
        "protocol": "unit-v1",
        "experiment_id": "unit-study",
        "corpus": "corpus.json",
        "dataset_status": "development",
        "audience": "intermediate",
        "seed": 17,
        "replicates": 1,
        "rubric_version": "rubric-v1",
        "engine": {"depth": 12, "probe_depth": 6, "max_plies": 10, "base_plies": 5, "multipv": 2, "threads": 1},
        "variants": _variant_matrix(),
        "evaluation": {"rater_ids": ["evaluator-a", "evaluator-b"], "anonymous": True, "assignment": "balanced"},
    }
    config.update(updates)
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, corpus_path


def test_diagnostic_study_is_executable_and_blinded():
    study = load_study(EXAMPLE)
    assert len(study["cases"]) == 7
    assert study["split_audit"]["passed"] is True
    assert study["split_audit"]["split_counts"] == {"diagnostic": 7}
    assert {variant["id"] for variant in study["variants"]} == {"B00", "B10", "B01", "B11"}
    assert study["evaluation"] == {
        "rater_ids": ["evaluator-a", "evaluator-b"], "anonymous": True, "assignment": "balanced"
    }
    assert sum(case["teaching_target"] is False for case in study["cases"]) == 1
    assert all("reference_notes" not in case and "reference" not in case for case in study["cases"])


def test_generator_input_ends_at_target_and_hides_annotations():
    study = load_study(EXAMPLE)
    opera = next(case for case in study["cases"] if case["id"] == "opera-sacrifice-p31")
    game = parse_game(opera["pgn"])
    assert len(list(game.mainline_moves())) == opera["target_ply"] == 31
    assert game.headers["Result"] == "*"
    assert "Nxb8" not in opera["pgn"]
    assert "Rd8#" not in opera["pgn"]
    assert "Private expert" not in json.dumps(opera)
    assert all(not node.comment and not node.nags and len(node.variations) <= 1 for node in game.mainline())


def test_private_reference_notes_do_not_change_corpus_hash(tmp_path):
    config_path, corpus_path = _write_study(tmp_path, [_case("a", "1. e4 e5 *")])
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["cases"][0]["reference_notes"] = "first private assessment"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    first = load_study(config_path)
    corpus["cases"][0]["reference_notes"] = "completely different private assessment"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    second = load_study(config_path)
    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["cases"] == second["cases"]


def test_rejects_duplicate_target(tmp_path):
    pgn = "1. e4 e5 *"
    cases = [_case("a", pgn), _case("b", pgn)]
    config_path, _ = _write_study(tmp_path, cases)
    with pytest.raises(ValueError, match="duplicate target"):
        load_study(config_path)


@pytest.mark.parametrize("shared_field", ["source", "family"])
def test_rejects_group_leakage_across_splits(tmp_path, shared_field):
    first = _case("a", "1. e4 *", split="development")
    second = _case("b", "1. d4 *", split="pilot")
    second[f"{shared_field}_group"] = first[f"{shared_field}_group"]
    config_path, _ = _write_study(tmp_path, [first, second])
    with pytest.raises(ValueError, match="split leakage"):
        load_study(config_path)


def test_rejects_transposed_post_position_across_splits(tmp_path):
    cases = [
        _case("a", "1. Nf3 Nf6 2. g3 g6 *", split="development", target_ply=4),
        _case("b", "1. g3 g6 2. Nf3 Nf6 *", split="pilot", target_ply=4),
    ]
    config_path, _ = _write_study(tmp_path, cases)
    with pytest.raises(ValueError, match="position:"):
        load_study(config_path)


def test_rejects_colour_mirrored_position_across_splits(tmp_path):
    first = chess.pgn.Game()
    first.setup(chess.Board("8/8/8/8/8/8/4K3/7k w - - 0 1"))
    first.add_main_variation(chess.Move.from_uci("e2e3"))
    mirrored = chess.pgn.Game()
    mirrored.setup(first.board().mirror())
    mirrored.add_main_variation(chess.Move.from_uci("e7f7"))
    cases = [
        _case("a", str(first), split="development"),
        _case("b", str(mirrored), split="pilot"),
    ]
    config_path, _ = _write_study(tmp_path, cases)
    with pytest.raises(ValueError, match="position:"):
        load_study(config_path)


def test_rejects_unimplemented_variant_and_fake_heldout(tmp_path):
    config_path, corpus_path = _write_study(tmp_path, [_case("a", "1. e4 *")])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["variants"][1]["horizon"] = "invented_selector"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unimplemented variant"):
        load_study(config_path)

    config["variants"] = _variant_matrix()
    config["dataset_status"] = "diagnostic_not_held_out"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["cases"][0]["split"] = "held_out"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly frozen"):
        load_study(config_path)
