from __future__ import annotations

import inspect

import chess

from app.core.commentary.features import experiment_selectors as selectors
from app.core.commentary.features.guid_features import FeatureValue
from app.core.commentary.rules.facts_builder import build_comment_facts
from app.models.comment_facts import Claim, EnvisionedLine


TRACE = [
    "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
]


def _trace_fens(moves: list[str] = TRACE) -> list[str]:
    board = chess.Board()
    result = []
    for token in moves:
        board.push_uci(token)
        result.append(board.fen())
    return result


def _install_feature_trace(monkeypatch, values: list[int]) -> None:
    fens = _trace_fens()
    by_fen = {chess.STARTING_FEN: 0, **dict(zip(fens, values))}

    def vector(fen: str):
        value = by_fen[fen]
        return {"TEST_CONCEPT": FeatureValue(name="TEST_CONCEPT", value_cp=value)}

    monkeypatch.setattr(selectors, "compute_feature_vector_fen", vector)
    monkeypatch.setattr(selectors, "is_quiescent", lambda _board: True)


def test_adaptive_uses_full_bounded_trace_and_prefers_persistent_concept(monkeypatch):
    monkeypatch.setenv("ENVISIONED_BASE_PLIES", "2")
    monkeypatch.setenv("ENVISIONED_MAX_PLIES", "8")
    _install_feature_trace(monkeypatch, [0, 0, 0, 100, 100, 100, 100, 100])

    line, metadata = selectors.select_adaptive_horizon(
        chess.STARTING_FEN, TRACE, root_eval_cp=30, depth=16
    )

    assert metadata["available_plies"] == 8
    assert metadata["selected_ply"] >= 4
    assert line.leaf_fen == _trace_fens()[metadata["selected_ply"] - 1]
    assert line.evidence_valid


def test_adaptive_boundary_persistence_is_unknown(monkeypatch):
    monkeypatch.setenv("ENVISIONED_MAX_PLIES", "3")
    _install_feature_trace(monkeypatch, [0, 0, 50, 50, 50, 50, 50, 50])
    _line, metadata = selectors.select_adaptive_horizon(
        chess.STARTING_FEN, TRACE, root_eval_cp=20, depth=16
    )
    assert metadata["candidates"][-1]["persistence"] == "unknown"
    assert metadata["candidates"][-1]["persistence_score"] == 0.0


def test_adaptive_abstains_when_no_quiet_endpoint(monkeypatch):
    monkeypatch.setattr(selectors, "is_quiescent", lambda _board: False)
    line, metadata = selectors.select_adaptive_horizon(
        chess.STARTING_FEN, TRACE[:3], root_eval_cp=20, depth=16
    )
    assert not line.evidence_valid
    assert len(line.line_uci) == 3
    assert metadata["reason"] == "no_eligible_quiet_endpoint"
    assert all(not item["eligible"] for item in metadata["candidates"])


def test_adaptive_uses_frozen_probe_to_exclude_a_candidate(monkeypatch):
    _install_feature_trace(monkeypatch, [0, 0, 0, 100, 100, 100, 100, 100])
    target = _trace_fens()[3]
    seen = []

    def probe(fen: str):
        seen.append(fen)
        return -300 if fen == target else 200

    _line, metadata = selectors.select_adaptive_horizon(
        chess.STARTING_FEN, TRACE, root_eval_cp=200, depth=16, prober=probe
    )
    assert seen == _trace_fens()
    assert metadata["candidates"][3]["excluded_reason"] == "probe_contradiction"


def test_terminal_trace_is_preserved_without_strategic_scoring():
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    line, metadata = selectors.select_adaptive_horizon(
        chess.STARTING_FEN, moves, root_eval_cp=-999999, depth=16
    )
    assert line.line_uci == moves
    assert chess.Board(line.leaf_fen).is_checkmate()
    assert metadata["reason"] == "terminal"


def _claim(rule: str, text: str, value: int = 20) -> Claim:
    return Claim(
        rule_id=rule, text=text, beneficiary="white", delta_cp=value,
        features_involved=[rule.upper()],
    )


def test_contrastive_prioritizes_semantic_distinctions_not_wording():
    shared_main = _claim("space", "White gains useful room.", 50)
    distinct_main = _claim("outpost", "The knight reaches d5.", 20)
    shared_alt = _claim("space", "White controls more territory.", 45)
    distinct_alt = _claim("outpost", "The knight reaches e5.", 15)

    main, alternative, metadata = selectors.select_contrastive_claims(
        [shared_main, distinct_main], [shared_alt, distinct_alt], mover="White"
    )

    assert main[0] == distinct_main
    assert alternative[0] == distinct_alt
    assert metadata == {
        "version": "contrastive_v2", "played_only": 1,
        "alternative_only": 1, "shared": 1,
    }


def test_contrastive_preserves_existing_budgets_and_beneficiary():
    played = [_claim(f"p{i}", f"Played concept {i}.", i + 10) for i in range(4)]
    alternatives = [_claim(f"a{i}", f"Alternative concept {i}.", i + 10) for i in range(4)]
    alternatives.append(
        Claim(rule_id="black_gain", text="Black improves.", beneficiary="black", delta_cp=99)
    )
    main, alternative, _ = selectors.select_contrastive_claims(
        played, alternatives, mover="White"
    )
    assert len(main) == len(played)
    assert len(alternative) == 2
    assert all(claim.beneficiary == "white" for claim in alternative)


def test_horizon_match_is_explicit():
    def line(n: int) -> EnvisionedLine:
        return EnvisionedLine(
            start_fen=chess.STARTING_FEN, fens=[chess.STARTING_FEN] * n,
            leaf_fen=chess.STARTING_FEN,
        )

    assert selectors.horizons_match(line(3), line(4)) is True
    assert selectors.horizons_match(line(2), line(4)) is False
    assert selectors.horizons_match(line(2), None) is None


def test_production_defaults_preserve_b00():
    signature = inspect.signature(build_comment_facts)
    assert signature.parameters["horizon_policy"].default == "current_quiet_bounded"
    assert signature.parameters["claim_policy"].default == "current"



def test_contrastive_recovers_distinction_from_beyond_early_cap():
    shared = [_claim(f"shared{i}", f"Shared {i}.", 100-i) for i in range(5)]
    distinction = _claim("unique", "A useful difference.", 10)
    main, _, _ = selectors.select_contrastive_claims(shared + [distinction], shared, mover="White")
    assert main[0].rule_id == "unique"
    assert len(main) == 4


def test_contrastive_material_and_concession_guardrails():
    pool = [_claim(f"p{i}", f"Merit {i}.", 30) for i in range(8)]
    pool += [Claim(rule_id=f"cost{i}", text="Black gains.", beneficiary="black", delta_cp=50) for i in range(4)]
    pool.append(Claim(rule_id="material_won", text="White wins material.", beneficiary="white", delta_cp=100))
    main, _, _ = selectors.select_contrastive_claims(pool, [], mover="White")
    assert main[0].rule_id == "material_won"
    assert len([c for c in main if c.rule_id != "material_won"]) <= 4
    assert sum(c.is_concession for c in main) <= 2


def test_adaptive_missing_and_failed_probes_are_unknown(monkeypatch):
    _install_feature_trace(monkeypatch, [20] * 8)
    for probe in (None, lambda fen: None, lambda fen: 1/0):
        _, metadata = selectors.select_adaptive_horizon(chess.STARTING_FEN, TRACE, root_eval_cp=200, depth=16, prober=probe)
        assert all(c["probe_status"] == "unknown" for c in metadata["candidates"])


def test_adaptive_tracks_loss_and_computes_each_state_once(monkeypatch):
    _install_feature_trace(monkeypatch, [50, 50, 0, 0, 60, 60, 60, 60])
    original = selectors.compute_feature_vector_fen
    calls = []
    def vector(fen):
        calls.append(fen)
        return original(fen)
    monkeypatch.setattr(selectors, "compute_feature_vector_fen", vector)
    _, metadata = selectors.select_adaptive_horizon(chess.STARTING_FEN, TRACE, root_eval_cp=20, depth=16)
    trajectory = metadata["trajectories"]["TEST_CONCEPT"]
    assert trajectory["first_appearance_ply"] == 1
    assert trajectory["loss_plies"] == [3]
    assert metadata["candidates"][0]["persistence_score"] == 0
    assert len(calls) == len(set(calls)) == 9


def test_feature_groups_do_not_add_side_and_aggregate_magnitudes():
    def vector(values):
        return {k: FeatureValue(name=k, value_cp=v) for k, v in values.items()}
    names = {"WHITE_PAWN_DOUBLED": 30, "BLACK_PAWN_DOUBLED": 20, "EVALUATE_PAWNS": 40}
    assert selectors._group_deltas(vector(dict.fromkeys(names, 0)), vector(names)) == {"PAWN_STRUCTURE": 40}


def test_rule_candidate_pool_opt_in_keeps_floor_and_material_exemption(monkeypatch):
    from app.core.commentary.rules import catalog
    from app.models.comment_facts import FeatureDiff
    material = _claim("material_won", "White wins material.", 100)
    claims = [material] + [_claim(f"shared{i}", f"Shared {i}.", 90-i) for i in range(5)]
    claims += [_claim("unique", "A small useful distinction.", 15), _claim("noise", "Noise.", 1)]
    monkeypatch.setattr(catalog, "ALL_RULES", [lambda ctx: claims])
    baseline = catalog.run_rules(FeatureDiff(), phase="mid", mover="White")
    pool = catalog.run_rules(FeatureDiff(), phase="mid", mover="White", cap=False)
    assert len(baseline) == 5
    assert baseline[0] == material
    assert "unique" not in {c.rule_id for c in baseline}
    assert "unique" in {c.rule_id for c in pool}
    assert "noise" not in {c.rule_id for c in pool}
    selected, _, _ = selectors.select_contrastive_claims(pool, pool[1:6], mover="White")
    assert len(selected) == len(baseline)
    assert selected[0] == material
    assert "unique" in {c.rule_id for c in selected}
