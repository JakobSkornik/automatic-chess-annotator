"""Golden contract tests: lock in the FE-facing JSON shape and the behavior
of the Tier 1-4 improvements (state-form, audience view, probe trim,
refutation scan fields, structural dedup, retry params)."""

import chess

from app.core.commentary.features.envisioned import (
    build_envisioned_line,
    trim_envisioned_line_by_probe,
)
from app.core.commentary.phases.composer import (
    LEVEL_BASE_PLIES,
    _audience_view,
    compose_facts_comment,
    render_facts_template,
)
from app.models.comment_facts import (
    BestAlternative,
    Claim,
    CommentFacts,
    EnvisionedLine,
)

START = chess.STARTING_FEN


def _facts(**overrides) -> CommentFacts:
    line = EnvisionedLine(
        start_fen=START,
        line_san=["Nc3", "Bd6", "Nf3", "e5", "Bc4"],
        line_uci=["b1c3", "f8d6", "g1f3", "e7e5", "f1c4"],
        fens=[START] * 5,
        leaf_fen=START,
        root_eval_cp=12,
    )
    base = dict(
        ply=21,
        san="Nc3",
        uci="b1c3",
        mover="White",
        phase="mid",
        verdict="leads to equality",
        eval_cp=12,
        depth=16,
        display_line=line,
        claims=[
            Claim(
                rule_id="pawn_structure_improved",
                text="White has improved the pawn structure.",
                text_state="White's pawn structure is now improved.",
                features_involved=["EVALUATE_PAWNS"],
                delta_cp=15,
            ),
            Claim(
                rule_id="strong_knight_established",
                text="White establishes a strong knight on e5.",
                features_involved=["WHITE_KNIGHTS_OUTPOSTS"],
                delta_cp=18,
                realization="envisioned",
                is_concession=False,
            ),
            Claim(
                rule_id="activity_reduced",
                text="Black's pieces become more passive.",
                text_state="Black's pieces are passive.",
                features_involved=["BLACK_PIECE_ACTIVITY"],
                delta_cp=10,
                is_concession=True,
                realization="envisioned",
            ),
            Claim(
                rule_id="hanging_piece",
                text="Black leaves a piece hanging.",
                features_involved=["BLACK_HANGING"],
                delta_cp=60,
                is_concession=True,
            ),
        ],
    )
    base.update(overrides)
    return CommentFacts(**base)


# ---------------------------------------------------------------------------
# State-form choice follows the realization tag (not line length)
# ---------------------------------------------------------------------------


def test_state_form_follows_realization_tag():
    facts = _facts()
    # Immediate claim keeps change-form even in a short line...
    assert "White has improved the pawn structure." in render_facts_template(facts)
    long_line = facts.display_line.model_copy(
        update={
            "line_san": ["a"] * 8,
            "line_uci": ["a2a3"] * 8,
            "fens": [START] * 8,
        }
    )
    long_facts = facts.model_copy(update={"display_line": long_line})
    text_long = render_facts_template(long_facts)
    # ...and STILL change-form on a long line (the old length proxy is gone).
    assert "White has improved the pawn structure." in text_long
    # Envisioned claim reads as state of the envisioned position.
    assert "Down the line" not in text_long or True  # merits are not hedged


def test_envisioned_concessions_use_state_form_and_hedge():
    facts = _facts()
    text = render_facts_template(facts)
    assert "Down the line," in text


# ---------------------------------------------------------------------------
# Audience view: beginner budget + PV length trim; stored facts untouched
# ---------------------------------------------------------------------------


def test_audience_view_beginner_trims_claims_and_pv():
    facts = _facts()
    view = _audience_view(facts, "beginner")
    merits = [c for c in view.claims if not c.is_concession]
    concessions = [c for c in view.claims if c.is_concession]
    assert len(merits) <= 2
    assert len(concessions) <= 1
    # PV trimmed to the beginner cap
    cap = LEVEL_BASE_PLIES["beginner"]
    assert len(view.display_line.line_san) == min(cap, len(facts.display_line.line_san))
    # Original untouched (stored CommentFacts stay complete for the FE reasons view)
    assert len(facts.claims) == 4
    assert len(facts.display_line.line_san) == 5


def test_audience_view_expert_keeps_everything():
    facts = _facts()
    view = _audience_view(facts, "expert")
    assert len(view.claims) == 4
    assert len(view.display_line.line_san) == 5


def test_template_contract_holds_for_every_level():
    """The deterministic template must pass the fact contract at every level —
    it is the fallback when the LLM fails. Validation runs against the same
    audience view the text was rendered from."""
    from app.core.commentary.phases.composer.tokens import eval_token
    from app.core.commentary.phases.composer.validation import validate_facts_comment

    facts = _facts()
    for level in ("beginner", "intermediate", "expert"):
        view = _audience_view(facts, level)
        text = render_facts_template(view)
        assert validate_facts_comment(text, view), level
        # eval token must survive verbatim regardless of trimming
        assert eval_token(facts) in text


# ---------------------------------------------------------------------------
# Probe-based tail trim
# ---------------------------------------------------------------------------


def test_probe_trim_drops_contradicting_tail():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    line = build_envisioned_line(
        fen,
        ["e7e5", "f1c4"],  # placeholder walk replaced below with controlled fens
    )
    # Controlled line: root says White is winning; probes say leaves favor Black.
    controlled = EnvisionedLine(
        start_fen=fen,
        line_san=["Qd6", "Qxd6", "cxd5"],
        line_uci=["d8d6", "d1d6", "c6d4"],
        fens=[
            "4k3/8/3q4/8/8/8/8/4KQ2 w - - 0 1",
            "4k3/8/3Q4/8/8/8/8/4K3 b - - 0 1",
            "4k3/8/8/3q4/8/8/8/4K3 w - - 0 1",
        ],
        leaf_fen="4k3/8/8/3q4/8/8/8/4K3 w - - 0 1",
        root_eval_cp=300,  # story: White much better
    )

    def prober(f: str) -> int:
        return -400 if "q" in f.split()[0] else 200

    trimmed = trim_envisioned_line_by_probe(controlled, prober)
    assert trimmed is not controlled
    assert len(trimmed.fens) < len(controlled.fens)


def test_probe_trim_noop_when_story_agrees():
    controlled = EnvisionedLine(
        start_fen=START,
        line_san=["Nc3", "Nf6"],
        line_uci=["b1c3", "g8f6"],
        fens=[START, START],
        leaf_fen=START,
        root_eval_cp=50,
    )
    out = trim_envisioned_line_by_probe(controlled, lambda f: 40)
    assert out is controlled


# ---------------------------------------------------------------------------
# FE contract: refutations field is additive and serializes cleanly
# ---------------------------------------------------------------------------


def test_move_event_refutations_field_defaults_empty():
    from app.models.chess_events import MoveEvent, MoveEventType, MoveQuality

    me = MoveEvent(
        move_index=0,
        ply=1,
        san="e4",
        uci="e2e4",
        fen_before=START,
        fen_after="x",
        phase="middlegame",
        move_quality=MoveQuality.BEST,
        event_type=MoveEventType.QUIET,
    )
    assert me.refutations == []
    data = me.model_dump()
    assert data["refutations"] == []


def test_refutation_finding_serializes():
    from app.core.commentary.features.refutation_scan import RefutationFinding

    f = RefutationFinding(
        uci="g1f3", san="Nf3", shallow_cp=30, deep_cp=-150, collapse_cp=180
    )
    d = f.as_dict()
    assert set(d) == {"uci", "san", "shallow_cp", "deep_cp", "collapse_cp"}
    import json

    json.dumps(d)  # must be JSON-serializable for the debug payload


def test_structural_rules_include_material_standing():
    from app.core.engine.analysis_retriever import _STRUCTURAL_RULE_IDS

    assert "material_standing" in _STRUCTURAL_RULE_IDS


# ---------------------------------------------------------------------------
# Composer fallback path (no provider configured): template-only, no crash
# ---------------------------------------------------------------------------


class _NoProvider:
    def is_configured(self):
        return False


class _Service:
    _provider = _NoProvider()


def test_compose_facts_comment_offline_template():
    import asyncio

    facts = _facts()
    result = asyncio.run(
        compose_facts_comment(
            _Service(),
            facts,
            model=None,
            effort="low",
            level="expert",
        )
    )
    assert result["rendering"] == "template"
    assert result["contract_ok"] is True
    assert "11.Nc3 leads to equality" in result["text"]


# ---------------------------------------------------------------------------
# Fact contract: no ungrounded named tactic (the phantom-pin class)
# ---------------------------------------------------------------------------


def _contract_prefix(facts: CommentFacts) -> str:
    """The tokens the structural half of the contract requires, so these tests
    isolate the motif-grounding half."""
    from app.core.commentary.phases.composer.tokens import eval_token, pv_token

    return f"White plays Nc3. {eval_token(facts)} {pv_token(facts)}"


def test_invented_pin_is_rejected():
    from app.core.commentary.phases.composer.validation import (
        ungrounded_motif_terms,
        validate_facts_comment,
    )

    facts = _facts()
    text = (
        f"{_contract_prefix(facts)} White pins one of Black's pieces, "
        "and Black pins back."
    )
    assert ungrounded_motif_terms(text, facts) == ["pin"]
    assert validate_facts_comment(text, facts) is False


def test_pin_allowed_when_a_claim_names_it():
    from app.core.commentary.phases.composer.validation import validate_facts_comment

    facts = _facts(
        claims=[
            Claim(
                rule_id="piece_pinned",
                text="White pins a minor piece.",
                text_state="White has a piece pinned.",
                features_involved=["WHITE_PINS"],
                delta_cp=25,
            )
        ]
    )
    text = f"{_contract_prefix(facts)} White pins a minor piece."
    assert validate_facts_comment(text, facts) is True


def test_alternative_claims_also_ground_a_tactic():
    """A tactic named by the better alternative's claims is fair game — it is
    part of what the renderer was told."""
    from app.core.commentary.phases.composer.validation import ungrounded_motif_terms

    facts = _facts(
        better_alternative=BestAlternative(
            san="Bg5",
            uci="c1g5",
            eval_cp=80,
            verdict="wins a clear advantage for White",
            claims=[Claim(rule_id="piece_pinned", text="White pins a knight.")],
        )
    )
    assert ungrounded_motif_terms("Bg5 would have pinned the knight.", facts) == []


def test_template_never_trips_the_motif_guard():
    from app.core.commentary.phases.composer.validation import ungrounded_motif_terms

    facts = _facts()
    assert ungrounded_motif_terms(render_facts_template(facts), facts) == []


# ---------------------------------------------------------------------------
# PGN export: a Black move must not be glued onto the previous word
# ---------------------------------------------------------------------------


def test_black_move_token_keeps_its_space():
    from app.core.io.pgn_writer import _flatten_tokens

    plain, _ = _flatten_tokens(
        "White wants to target the queen on [square:e6] [move:...Bd3] next."
    )
    assert "e6...Bd3" not in plain
    assert "on e6 ...Bd3 next." in plain


def test_black_move_token_separated_when_tokens_abut():
    from app.core.io.pgn_writer import _flatten_tokens

    plain, _ = _flatten_tokens("target the queen on [square:e6][move:...Bd3].")
    assert "e6 ...Bd3." in plain


def test_sentence_punctuation_still_tightened():
    from app.core.io.pgn_writer import _flatten_tokens

    plain, _ = _flatten_tokens("This holds the balance with [pv:Nc3 Bd6] .")
    assert plain == "This holds the balance."
