"""Tests for compact_game_context_for_move (no digest spoilers)."""

from __future__ import annotations

from app.core.commentary.advanced_comment_service import compact_game_context_for_move


def test_compact_omits_overall_story_and_future_spoilers() -> None:
    digest = {
        "overall_story": "Black wins",
        "opening_character": "e4 e5",
        "strategic_archetype": "iqp",
        "turning_points": [
            {"ply": 10, "san": "d4", "why": "x", "motif": "none"},
            {"ply": 25, "san": "Rg1", "why": "y", "motif": "clearance"},
        ],
        "phase_story": [
            {"phase": "opening", "summary": "dev"},
            {"phase": "middlegame", "summary": "fight"},
        ],
        "winning_side_plan": "Black converts",
        "losing_side_mistakes": "White blundered",
    }
    ctx = compact_game_context_for_move(digest, current_ply=12, current_phase="opening")
    assert "overall_story" not in ctx
    assert "winning_side_plan" not in ctx
    assert "losing_side_mistakes" not in ctx
    assert ctx["strategic_archetype"] == "iqp"
    assert ctx["opening_character"] == "e4 e5"
    past = ctx["turning_points"]
    assert len(past) == 1 and past[0]["ply"] == 10
    assert ctx["phase_story"] == [{"phase": "opening", "summary": "dev"}]


def test_compact_drops_opening_character_after_ply_20() -> None:
    digest = {
        "opening_character": "only early",
        "strategic_archetype": "other",
        "turning_points": [],
        "phase_story": [{"phase": "middlegame", "summary": "x"}],
    }
    ctx = compact_game_context_for_move(digest, current_ply=22, current_phase="middlegame")
    assert "opening_character" not in ctx


def test_compact_keeps_phase_story_while_trimming_turning_points() -> None:
    long_why = "noise-" * 55
    tps = [{"ply": i, "san": "e4", "why": long_why, "motif": "none"} for i in range(1, 17)]
    digest = {
        "strategic_archetype": "other",
        "turning_points": tps,
        "phase_story": [{"phase": "middlegame", "summary": "PHASE_MARKER_UNIQUE_XYZ"}],
    }
    ctx = compact_game_context_for_move(digest, current_ply=30, current_phase="middlegame")
    summaries = ctx.get("phase_story") or []
    assert summaries and summaries[0].get("summary") == "PHASE_MARKER_UNIQUE_XYZ"
    assert len(ctx.get("turning_points") or []) < len(tps)
