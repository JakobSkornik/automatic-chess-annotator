"""Regression test for the back-to-back key-moment stub-reference bug.

``_apply_back_to_back_key_moment_suppression`` (analysis/key_moments.py)
demotes the second of two same-type key moments within 2 plies: it clears
``key_moment_type`` and sets ``brief_commentary`` / ``commentary_stub_ref_ply``,
promising the reader a short stub instead of a full re-explanation. But
``_resolve_move_comment`` only ever reached the stub-rendering branch of
``_fallback_comment`` through a code path gated on ``move_event.key_moment_type``
being truthy -- exactly the field the suppression step had just cleared to
None. The result, confirmed empirically against a real annotated-game batch:
the move keeps its "!"/"?!" quality badge (``classification``/``annotation``,
sourced from the pre-suppression ``AnalyzedMoveData.key_moment_type`` cache)
but ``comment`` stays None -- a badge with nothing behind it.
"""

from app.core.engine.analysis.comment_assembly import _resolve_move_comment
from app.models.chess_events import AnalyzedMoveData, MoveEvent, MoveEventType
from app.models.chess_events import MoveQuality
from app.models.Move import Move

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _analyzed_move() -> Move:
    return Move(
        id=0,
        position=START,
        move="c3c4",
        context="",
        depth=91,
        isAnalyzed=True,
        piece="r",
        hiddenFeatures={},
    )


def _row(**overrides) -> AnalyzedMoveData:
    base = dict(
        index=0,
        ply=91,
        san="Rc4",
        uci="c3c4",
        fen_before=START,
        fen_after=START,
        phase_raw="end",
        analyzed_move=_analyzed_move(),
    )
    base.update(overrides)
    return AnalyzedMoveData(**base)


def _suppressed_move_event(**overrides) -> MoveEvent:
    """A move demoted by back-to-back suppression: key_moment_type cleared,
    brief_commentary set, pointing back at an earlier ply."""
    base = dict(
        move_index=0,
        ply=91,
        san="Rc4",
        uci="c3c4",
        fen_before=START,
        fen_after=START,
        phase="end",
        move_quality=MoveQuality.BEST,
        event_type=MoveEventType.QUIET,
        key_moment_type=None,
        brief_commentary=True,
        commentary_stub_ref_ply=90,
    )
    base.update(overrides)
    return MoveEvent(**base)


def test_suppressed_move_gets_stub_comment_not_silence():
    """The core bug: a brief_commentary move must not end up with comment=None."""
    move_event = _suppressed_move_event()
    row = _row()

    result = _resolve_move_comment(
        row, row.analyzed_move, move_event, key_moment=None, side_ok=True
    )

    assert result.comment is not None, (
        "back-to-back suppression promised a stub reference but the move "
        "ended up with no comment at all"
    )
    assert "90" in result.comment
    assert "commentary" in result.comment.lower()


def test_non_suppressed_key_moment_unaffected():
    """A normal (non-suppressed) key moment with no LLM comment and no
    instructive facts still falls through to silence, unchanged by this fix."""
    move_event = _suppressed_move_event(
        key_moment_type="great_move",
        brief_commentary=False,
        commentary_stub_ref_ply=None,
    )
    row = _row()

    result = _resolve_move_comment(
        row, row.analyzed_move, move_event, key_moment="great_move", side_ok=True
    )

    # No facts, no LLM comment -> "nothing instructive to say" for an optional
    # "!"-type move, same behavior as before this fix.
    assert result.comment is None


def test_quiet_non_key_moment_still_silent():
    """A plain non-key-moment, non-suppressed move stays fully silent."""
    move_event = _suppressed_move_event(
        key_moment_type=None,
        brief_commentary=False,
        commentary_stub_ref_ply=None,
    )
    row = _row()

    result = _resolve_move_comment(
        row, row.analyzed_move, move_event, key_moment=None, side_ok=True
    )

    assert result.comment is None


def test_suppressed_move_respects_side_gate():
    """A suppressed move on the non-selected commentary side stays silent too."""
    move_event = _suppressed_move_event()
    row = _row()

    result = _resolve_move_comment(
        row, row.analyzed_move, move_event, key_moment=None, side_ok=False
    )

    assert result.comment is None
