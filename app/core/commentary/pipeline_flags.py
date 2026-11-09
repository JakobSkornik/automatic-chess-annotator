"""Behaviour switches for the v19 pipeline changes (8 Nov 2026).

Every switch defaults to the v18 behaviour, so stored material and the
regression tests keep reproducing until a switch is turned on. Turn all of
them on with ``ACA_PROFILE=v19``; override one with ``ACA_V19_<NAME>=1|0``
(the override wins over the profile).

The thesis describes the behaviour behind each switch; ``fingerprint()`` is
written into every analysis record so a reader can tell which behaviour
produced it.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

# name -> (one-line description, where it acts)
FLAGS: dict[str, tuple[str, str]] = {
    "writer_forward_order": (
        "The writer comments moves from the first to the last; the session history holds "
        "only earlier accepted comments, so no later-game knowledge is visible.",
        "pipeline/game_pipeline.py, phases/composer/narrative.py"),
    "transient_at_errors": (
        "At an inaccuracy, mistake or blunder a transient claim that favours the opponent "
        "and holds for at least two plies is kept (marked transient) instead of dropped.",
        "rules/facts_builder.py"),
    "quiet_end_comparison": (
        "The two lines are compared only at their own quiet ends; the matched-depth "
        "comparison at the shorter line's length is not produced.",
        "rules/facts_builder.py"),
    "fork_probe": (
        "An unrealised fork is confirmed by the shallow probe (capture of a target keeps the "
        "attacker's eval within PROBE_TOLERANCE_CP) like the other static facts; otherwise "
        "it is dropped.",
        "rules/facts_builder.py"),
    "validator_piece_moves": (
        "Check 10: every piece move named in the text (SAN with a piece letter) must be a "
        "legal move from the position before the move or a move of one of the engine lines.",
        "phases/composer/narrative.py"),
    "validator_sentence_cites": (
        "Check 11: the writer returns sentences, each with its own citations; a sentence "
        "without a citation is rejected unless it only states the evaluation placeholders.",
        "phases/composer/narrative.py"),
    "mate_classes": (
        "A move that gives up a forced mate is a mistake when the position stays decided and "
        "a blunder otherwise; a difference in the distance to mate is not a loss.",
        "event_extractor.py"),
    "decision_point_groups": (
        "Decision point (second way): the top two moves are within CLOSE_MARGIN_CP and the two "
        "lines diverge in at least two feature groups (material, pawns, pieces, king, "
        "tactics), each above its group threshold.",
        "engine/analysis/moment_kinds.py"),
    "transition_once": (
        "A phase boundary (piece count crossing the endgame threshold) anchors an "
        "endgame_transition key moment only the first time it is crossed in a game.",
        "event_extractor.py"),
    "no_missed_opportunity": (
        "The separate missed_opportunity event is not raised; such moves are covered by the "
        "blunder class and its gate.",
        "key_moment_detector.py"),
}

_PROFILES = {"v18": set(), "v19": set(FLAGS)}


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def profile() -> str:
    return os.environ.get("ACA_PROFILE", "v18").strip().lower() or "v18"


def enabled(name: str) -> bool:
    """Whether the switch ``name`` is on (override > profile > off)."""
    if name not in FLAGS:
        raise KeyError(f"unknown pipeline flag {name!r}")
    override = os.environ.get(f"ACA_V19_{name.upper()}")
    if override is not None and override.strip() != "":
        return _truthy(override)
    return name in _PROFILES.get(profile(), set())


def active() -> dict[str, bool]:
    return {name: enabled(name) for name in FLAGS}


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git missing
        return None


def fingerprint() -> dict:
    """What produced an analysis: profile, switches, engine settings, code commit."""
    from app.core.engine.analysis.constants import DEFAULT_ANALYSIS_DEPTH, DEFAULT_PV_COUNT

    return {
        "profile": profile(),
        "flags": active(),
        "run_label": os.environ.get("ACA_RUN_LABEL") or None,
        "engine": {
            "depth": DEFAULT_ANALYSIS_DEPTH,
            "multipv": DEFAULT_PV_COUNT,
            "threads": int(os.environ.get("STOCKFISH_THREADS", "4")),
            "hash_mb": int(os.environ.get("STOCKFISH_HASH_MB", "32")),
            "nnue": _truthy(os.environ.get("STOCKFISH_USE_NNUE", "0")),
            "leaf_probe_depth": int(os.environ.get("LEAF_PROBE_DEPTH", "6")),
        },
        "code_commit": _git_commit(),
    }
