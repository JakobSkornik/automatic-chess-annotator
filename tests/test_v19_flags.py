"""The v19 switches: off by default (v18 behaviour), on with ACA_PROFILE=v19 or a per-flag override."""
import json
from types import SimpleNamespace

import chess
import pytest

from app.core.commentary import pipeline_flags
from app.core.commentary.event_extractor import _mate_class
from app.core.commentary.phases.composer import narrative
from app.core.engine.analysis import moment_kinds
from app.core.engine.analysis.constants import MATE_SCORE
from app.models.chess_events import MoveQuality
from app.models.comment_facts import BestAlternative, Claim, CommentFacts, EnvisionedLine, FeatureAdvantage


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ACA_PROFILE", raising=False)
    for name in pipeline_flags.FLAGS:
        monkeypatch.delenv(f"ACA_V19_{name.upper()}", raising=False)


def test_all_flags_off_by_default_and_on_with_v19_profile(monkeypatch):
    assert not any(pipeline_flags.active().values())
    assert pipeline_flags.profile() == "v18"
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert all(pipeline_flags.active().values())
    monkeypatch.setenv("ACA_V19_FORK_PROBE", "0")
    assert not pipeline_flags.enabled("fork_probe") and pipeline_flags.enabled("mate_classes")
    with pytest.raises(KeyError):
        pipeline_flags.enabled("no_such_flag")


def test_fingerprint_records_profile_flags_and_engine_settings(monkeypatch):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    monkeypatch.setenv("ACA_RUN_LABEL", "thesis-v19")
    fp = pipeline_flags.fingerprint()
    assert fp["profile"] == "v19" and fp["run_label"] == "thesis-v19"
    assert set(fp["flags"]) == set(pipeline_flags.FLAGS)
    assert {"depth", "multipv", "threads", "hash_mb", "nnue", "leaf_probe_depth"} <= set(fp["engine"])


# --- mate classes -------------------------------------------------------------

def _mate(n):  # White-POV mate-coded score, n plies/moves to mate
    return MATE_SCORE - n if n > 0 else -(MATE_SCORE + n)


@pytest.mark.parametrize("best, played, expected", [
    (_mate(3), _mate(7), MoveQuality.BEST),          # slower mate for the same side is not a loss
    (_mate(3), 350, MoveQuality.MISTAKE),            # gives up the mate, still decided
    (_mate(3), 40, MoveQuality.BLUNDER),             # gives up the mate into a balanced game
    (_mate(3), _mate(-5), MoveQuality.BLUNDER),      # from mating to being mated
    (250, _mate(4), MoveQuality.BEST),               # finds a mate where the best was numeric
    (250, _mate(-4), MoveQuality.BLUNDER),           # walks into mate
    (60, 20, None),                                  # no mate involved: ordinary bands apply
])
def test_mate_class_table(best, played, expected):
    assert _mate_class(best, played) == expected


# --- decision point groups ------------------------------------------------------

def _event(advantages):
    alt = BestAlternative(san="Nf3", uci="g1f3", feature_advantages=[
        FeatureAdvantage(name=n, alt_delta_cp=cp, played_delta_cp=0, advantage_cp=cp) for n, cp in advantages])
    facts = CommentFacts(ply=1, san="e4", uci="e2e4", mover="White", phase="mid", verdict="",
                         better_alternative=alt)
    return SimpleNamespace(comment_facts=facts)


def test_feature_divergence_flat_in_v18_grouped_in_v19(monkeypatch):
    # Three pawn features at 15 cp each: flat rule sees none (< 30); grouped rule sums pawns = 45 >= 25.
    event = _event([("PAWN_ISOLATED", 15), ("PAWN_DOUBLED", 15), ("WEAK_PAWNS", 15), ("KING_DANGER", 35)])
    assert moment_kinds.feature_divergence(event) == 1  # only KING_DANGER passes the flat 30
    monkeypatch.setenv("ACA_V19_DECISION_POINT_GROUPS", "1")
    assert moment_kinds.feature_divergence(event) == 1  # pawns 45 >= 25; king 35 < 60
    event = _event([("MATERIAL_BALANCE", 60), ("MATERIAL_BALANCE_X", 50), ("ROOK_OPEN_FILE", 25)])
    assert moment_kinds.feature_divergence(event) == 2  # material 110 >= 100; pieces 25 >= 20


def test_every_catalog_feature_has_a_group():
    from app.core.commentary.features.guid_features.feature_catalog import CATALOG
    for name in CATALOG:
        assert moment_kinds.feature_group(name) in moment_kinds.FEATURE_GROUP_THRESHOLDS_CP


# --- writer: order note, sentence schema, checks 10 and 11 ----------------------

def _facts():
    board = chess.Board()
    start = board.fen()
    board.push_san("e4")
    first = board.fen()
    board.push_san("e5")
    return CommentFacts(ply=1, san="e4", uci="e2e4", mover="White", phase="mid",
        verdict="holds the balance", eval_cp=15,
        display_line=EnvisionedLine(start_fen=start, line_uci=["e2e4", "e7e5"], line_san=["e4", "e5"],
            fens=[first, board.fen()], leaf_fen=board.fen()),
        claims=[Claim(rule_id="test", text="White has a pawn on e4.", baseline_fen=start,
                      support_fen=first, support_ply=1)])


def test_system_prompt_and_schema_follow_the_switches(monkeypatch):
    assert narrative.system_prompt() == narrative.SYSTEM and narrative.schema() is narrative.SCHEMA
    assert "processed backwards" in narrative.SYSTEM
    monkeypatch.setenv("ACA_PROFILE", "v19")
    text = narrative.system_prompt()
    assert "processed backwards" not in text and "from the first move to the last" in text
    assert "TRANSIENT CLAIMS" in text and "sentences" in text
    assert "links" in narrative.schema()["properties"]["sentences"]["items"]["required"]
    monkeypatch.setenv("ACA_V19_STATEMENT_LINKS", "0")
    assert narrative.schema() is narrative.BLOCK_SCHEMA
    monkeypatch.setenv("ACA_V19_COMMENT_BLOCKS", "0")
    assert narrative.schema() is narrative.SENTENCE_SCHEMA


def test_check_10_rejects_a_move_that_is_neither_in_the_lines_nor_legal(monkeypatch):
    _, evidence = narrative.prompt(_facts(), "intermediate")
    bad = json.dumps({"text": "After Nf3 and later Qxf7 White wins.", "main": [0], "alternative": []})
    ok = json.dumps({"text": "White can follow with Nf3, developing.", "main": [0], "alternative": []})
    narrative.parse_proposal(bad, evidence)  # v18: not checked
    monkeypatch.setenv("ACA_V19_VALIDATOR_PIECE_MOVES", "1")
    narrative.parse_proposal(ok, evidence)  # Nf3 is legal after 1.e4 e5
    with pytest.raises(ValueError, match="Move not in the displayed lines"):
        narrative.parse_proposal(bad, evidence)  # Qxf7 is not legal in any position of the line


def test_check_11_sentence_citations(monkeypatch):
    _, evidence = narrative.prompt(_facts(), "intermediate")
    monkeypatch.setenv("ACA_V19_VALIDATOR_SENTENCE_CITES", "1")
    good = {"sentences": [
        {"text": "White stakes a claim in the centre with the pawn on e4.", "main": [0], "alternative": []},
        {"text": "The position stays balanced.", "main": [], "alternative": []}]}
    proposal, reviews = narrative.parse_proposal(json.dumps(good), evidence)
    assert proposal == {"text": "White stakes a claim in the centre with the pawn on e4. The position stays balanced.",
                        "main": [0], "alternative": []}
    assert reviews[0]["text"] == proposal["text"]
    uncited_square = {"sentences": [{"text": "The pawn on e4 is strong.", "main": [], "alternative": []}]}
    with pytest.raises(ValueError, match="without a citation"):
        narrative.parse_proposal(json.dumps(uncited_square), evidence)
    twice = {"sentences": [{"text": "A pawn on e4.", "main": [0], "alternative": []},
                           {"text": "Again e4.", "main": [0], "alternative": []}]}
    with pytest.raises(ValueError, match="duplicate"):
        narrative.parse_proposal(json.dumps(twice), evidence)
    with pytest.raises(ValueError, match="sentences"):
        narrative.parse_proposal(json.dumps({"text": "x", "main": [], "alternative": []}), evidence)


# --- facts builder: transient claims at errors, fork probe ------------------------

from app.core.commentary.features.guid_features import compute_feature_vector_fen  # noqa: E402
from app.core.commentary.rules.facts_builder import _FactsCtx, _gate_static_facts, _trajectory_claims  # noqa: E402
from app.core.commentary.rules.tactical_evidence import replay  # noqa: E402
from app.models.chess_events import MoveEvent  # noqa: E402


def _line(fen, sans):
    board = chess.Board(fen)
    ucis, fens = [], []
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return EnvisionedLine(start_fen=fen, line_uci=ucis, line_san=list(sans), fens=fens, leaf_fen=fens[-1],
                          immediate_horizon_ply=1)


def _ctx(fen, uci, quality="good", mover="White"):
    event = MoveEvent(move_index=0, ply=1, san="x", uci=uci, fen_before=fen, fen_after=fen,
                      phase="mid", move_quality=quality, event_type="quiet")
    return _FactsCtx(move_event=event, row=None, board_before=chess.Board(fen), mover=mover, phase_raw="mid",
                     depth=16, start_vec=compute_feature_vector_fen(fen))


def test_transient_opponent_gain_is_kept_only_at_errors_with_the_switch(monkeypatch):
    # 1.Kd2?? Nxa2 2.Kd3 Nb4+ 3.Kxd4: Black is a pawn up from ply 2, and material is level again at ply 5.
    fen = "4k3/8/8/8/3p4/2n5/P6P/4K3 w - - 0 1"
    branch = _line(fen, ["Kd2", "Nxa2", "Kd3", "Nb4+", "Kxd4"])
    branch.root_eval_cp = 0

    def material_for_black(ctx):
        claims, _ = _trajectory_claims(ctx, branch, 30)
        return [c for c in claims if c.rule_id == "material_won" and c.beneficiary == "black"]

    c = _ctx(fen, "e1d2", "blunder")
    assert material_for_black(c) == []  # v18: transient claims are dropped
    assert "transient_in_displayed_line" in {s["reason"] for s in c.evidence_suppressions}
    monkeypatch.setenv("ACA_V19_TRANSIENT_AT_ERRORS", "1")
    assert material_for_black(_ctx(fen, "e1d2", "good")) == []  # a sound move: still dropped
    kept = material_for_black(_ctx(fen, "e1d2", "blunder"))
    assert kept and all(c.persistence == "transient" and c.first_ply == 2 and c.last_ply >= 3 for c in kept)
    assert all(c.evidence.squares for c in kept)  # evidence filled like a held claim


def test_fork_probe_confirms_or_refutes_an_uncashed_fork(monkeypatch):
    # 1.Nb6 forks the rook on a8 and the bishop on c8; the line never takes either.
    fen = "r1bk4/8/8/8/2N5/8/8/4K3 w - - 0 1"
    branch = _line(fen, ["Nb6", "Ke7", "Kd2", "Kd6"])
    branch.root_eval_cp = 300
    claim = Claim(rule_id="fork", text="x", beneficiary="white", support_ply=1, entity="fork:b6",
                  evidence={"squares": ["b6"], "pieces": [
                      {"square": "b6", "piece": "knight", "color": "white", "role": "subject"},
                      {"square": "a8", "piece": "rook", "color": "black", "role": "target"},
                      {"square": "c8", "piece": "bishop", "color": "black", "role": "target"}]})
    c = _ctx(fen, "c4b6")
    assert _gate_static_facts(c, [claim], branch, replay(branch)) == []
    assert c.evidence_suppressions[-1]["reason"] == "fork_not_cashed_in_line"
    monkeypatch.setenv("ACA_V19_FORK_PROBE", "1")
    c = _ctx(fen, "c4b6")
    c.prober = lambda probe_fen: 320  # taking a target keeps White's eval
    assert len(_gate_static_facts(c, [claim], branch, replay(branch))) == 1
    c = _ctx(fen, "c4b6")
    c.prober = lambda probe_fen: 100  # taking a target loses 200 cp: refuted
    assert _gate_static_facts(c, [claim], branch, replay(branch)) == []
    assert c.evidence_suppressions[-1]["reason"] == "fork_refuted_by_probe"
    c = _ctx(fen, "c4b6")  # no prober: unknown, dropped as before
    assert _gate_static_facts(c, [claim], branch, replay(branch)) == []
    assert c.evidence_suppressions[-1]["reason"] == "fork_not_cashed_in_line"
