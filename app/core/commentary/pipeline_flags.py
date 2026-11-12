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
    # --- after the meeting with the supervisor, 11 Nov 2026 ---
    "scaled_class_limits": (
        "The limits of the quality classes depend on the evaluation before the move: a class "
        "ends where the mover's expected score (logistic curve, 400 cp per decade) has dropped "
        "by 2, 5, 10 or 20 percentage points, instead of at a loss of 20, 50, 100 or 200 cp.",
        "event_extractor.py"),
    "template_without_lines": (
        "The template comment prints no variation and no numeric evaluation (both are shown "
        "beside the board), and a statement joined to a lead-in starts in lower case.",
        "phases/composer/template.py"),
    "comment_blocks": (
        "A comment has two blocks, the played move and then the alternative on its own line, "
        "and every move named in it carries its move number.",
        "phases/composer/template.py, phases/composer/narrative.py, phases/composer/plain.py"),
    "hash_cleared_per_game": (
        "The engine's hash table is cleared before the analysis of each game, so the result "
        "for a game does not depend on which games were analysed before it.",
        "engine/engine_connector.py, engine/analysis_retriever.py"),
    "claimless_parts": (
        "A part of the comment whose line has no claim is written as its move and the "
        "assessment only; a move with no claim on either line is not sent to the model and "
        "keeps the template text.",
        "phases/composer/narrative.py, phases/composer/__init__.py"),
    "statement_wording": (
        "Wording of statements and verdicts: a sound move in a worse position is not said to "
        "make matters worse, an exchange of equal pieces is named as an exchange, a pawn break "
        "the pawn could have played at once is not offered, a parried threat is one plain "
        "sentence, a bishop is hemmed in or freed only when the pawns on its colour change; "
        "the template names the move of the line a statement first holds after, joins the "
        "punishing reply with its first statement, leaves out the "
        "material count in the middle of an exchange, joins at most two statements and gives "
        "an explanation once.",
        "rules/verdicts.py, rules/tactical_evidence.py, rules/prophylaxis.py, "
        "rules/feature_rules.py, features/intent.py, "
        "phases/composer/template.py"),
    "other_moves_concede": (
        "When the played move is the engine's choice and the next-best move is clearly worse, "
        "that move is neither named nor explained: the template writes what other moves concede "
        "(a piece or pawn, a feature), and the full writer gets that list in place of the "
        "runner-up, its line and its claims.",
        "phases/composer/framing.py, phases/composer/template.py, phases/composer/narrative.py"),
    "base_writer": (
        "Survey version B is the base model: it gets the position and the two engine lines with "
        "their evaluations as text, one independent request per move, with the writing rules "
        "of the full writer (numbered moves, sides named, no figures, paragraph rule, other "
        "moves concede) and none of its evidence; only the length is checked.",
        "phases/composer/plain.py, phases/composer/__init__.py"),
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
