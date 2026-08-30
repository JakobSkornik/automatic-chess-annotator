"""Wiring tests: the new Tier 1-4 pieces integrate with the pipeline objects
without needing a live engine — CommentFacts -> GameMove.debug.refutations,
dedup windows, retry params, and the audience view through MoveCommentary."""

import asyncio

import chess

from app.core.commentary.features.envisioned import build_envisioned_line
from app.models.chess_events import (
    AnalyzedMoveData,
    MoveEvent,
    MoveEventType,
    MoveQuality,
)
from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine

START = chess.STARTING_FEN


def _facts() -> CommentFacts:
    line = EnvisionedLine(
        start_fen=START,
        line_san=["Nc3", "Bd6", "Nf3", "e5"],
        line_uci=["b1c3", "f8d6", "g1f3", "e7e5"],
        fens=[START] * 4,
        leaf_fen=START,
        root_eval_cp=-30,
    )
    return CommentFacts(
        ply=21,
        san="Nc3",
        uci="b1c3",
        mover="White",
        phase="mid",
        verdict="lets the advantage slip away to equality",
        eval_cp=12,
        eval_before_cp=80,
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
                rule_id="hanging_piece",
                text="Black leaves a piece hanging.",
                features_involved=["BLACK_HANGING"],
                delta_cp=60,
                is_concession=True,
            ),
        ],
    )


def _move_event(facts: CommentFacts | None = None) -> MoveEvent:
    return MoveEvent(
        move_index=0,
        ply=facts.ply if facts else 21,
        san=facts.san if facts else "Nc3",
        uci=facts.uci if facts else "b1c3",
        fen_before=START,
        fen_after=START,
        phase="middlegame",
        move_quality=MoveQuality.MISTAKE,
        event_type=MoveEventType.QUIET,
        comment_facts=facts,
        refutations=[
            {
                "uci": "g1f3",
                "san": "Nf3",
                "shallow_cp": 40,
                "deep_cp": -120,
                "collapse_cp": 160,
            }
        ],
    )


class TestDebugRefutationsExposed:
    def test_debug_carries_refutations(self):
        from app.core.engine.analysis.comment_assembly import _build_move_debug
        from app.models.Move import Move

        facts = _facts()
        me = _move_event(facts)
        analyzed = Move(
            id=1,
            position=START,
            move="b1c3",
            context="mainline",
            depth=21,
            isAnalyzed=True,
            piece="N",
        )
        dbg = _build_move_debug(analyzed, me, facts, "some comment", "mid")
        assert dbg is not None
        assert dbg["refutations"] == [
            {
                "uci": "g1f3",
                "san": "Nf3",
                "shallow_cp": 40,
                "deep_cp": -120,
                "collapse_cp": 160,
            }
        ]

    def test_debug_none_without_comment(self):
        from app.core.engine.analysis.comment_assembly import _build_move_debug
        from app.models.Move import Move

        me = _move_event(_facts())
        analyzed = Move(
            id=1,
            position=START,
            move="b1c3",
            context="mainline",
            depth=21,
            isAnalyzed=True,
            piece="N",
        )
        assert _build_move_debug(analyzed, me, _facts(), None, "mid") is None


class TestDedupWindows:
    def test_structural_window_applies_to_material_standing(self):
        from app.core.engine.analysis_retriever import (
            _STRUCTURAL_RULE_IDS,
            _dedup_claims,
        )

        assert "material_standing" in _STRUCTURAL_RULE_IDS
        facts = _facts().model_copy(
            update={
                "claims": [
                    Claim(
                        rule_id="material_standing",
                        text="White is a rook up.",
                        features_involved=["MATERIAL_BALANCE"],
                        delta_cp=500,
                    )
                ]
            }
        )
        last: dict = {}
        kept = _dedup_claims(facts, ply=10, last_claim_ply=last, claim_window=6, structural_window=24)
        assert len(kept.claims) == 1  # first occurrence kept
        again = _dedup_claims(kept, ply=14, last_claim_ply=last, claim_window=6, structural_window=24)
        assert again.claims == []  # within the 24-ply structural window -> muted


class TestAudienceViewThroughModels:
    def test_audience_view_serializes_for_fe(self):
        from app.core.commentary.phases.composer import _audience_view
        import json

        view = _audience_view(_facts(), "beginner")
        d = json.loads(view.model_dump_json())
        assert d["display_line"]["line_san"] == ["Nc3", "Bd6", "Nf3"]  # trimmed to 3
        # claims budget: merits <=2, concessions <=1
        merits = [c for c in d["claims"] if not c.get("is_concession")]
        assert len(merits) <= 2

    def test_full_facts_still_carry_complete_line(self):
        from app.core.commentary.phases.composer import _audience_view

        facts = _facts()
        _audience_view(facts, "beginner")
        assert len(facts.display_line.line_san) == 4  # untouched


class TestRefutationScanUnit:
    def test_scan_disabled_yields_nothing(self, monkeypatch):
        from app.core.commentary.features.refutation_scan import RefutationScanner

        class FakeEngine:
            def analyse(self, *a, **k):
                raise AssertionError("engine must not be called when disabled")

        monkeypatch.setenv("REFUTATION_SCAN", "0")
        s = RefutationScanner(FakeEngine())
        assert s.scan(chess.STARTING_FEN, 0) == []

    def test_max_positions_cap(self, monkeypatch):
        from app.core.commentary.features.refutation_scan import RefutationScanner

        class CountingEngine:
            def __init__(self):
                self.calls = 0

            def analyse(self, *a, **k):
                self.calls += 1
                return []

        monkeypatch.setenv("REFUTATION_SCAN_MAX_POSITIONS", "2")
        eng = CountingEngine()
        s = RefutationScanner(eng)

        class FakeScore:
            def white(self):
                class W:
                    @staticmethod
                    def score(mate_score):
                        return 0

                return W()

        for _ in range(5):
            s.scan(chess.STARTING_FEN, 0)
        assert s.positions_scanned == 2
