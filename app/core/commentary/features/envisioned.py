"""Envisioned-position machinery (Guid §5.3), trajectory horizon (2 Nov 2026).

The envisioned position is the end of a *shortened* principal variation. The
endpoint is chosen by *settlement*: the earliest quiet ply after which no
feature group newly crosses its threshold or reverses direction for the rest
of the bounded window (hard cap 12 plies). When no ply settles, the ply whose
feature changes persist best to the end of the window is used; the fixed
display window is only the last resort. The line also records its *immediate
horizon*: the first settled ply after the move (1-3), at which "what the move
does" is read.

Quiescence is SEE-based and counts pawns: a position is quiet when the side
not to move has nothing the mover can win material on by a capture sequence.

Everything here is engine-free: PVs and evals come from searches the engine
pass already paid for.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

import chess

from app.core.commentary.features.guid_features import (
    FeatureVector,
    compute_feature_vector_fen,
)
from app.core.commentary.features.see import en_prise_squares
from app.models.comment_facts import (
    EnvisionedLine,
    FeatureAdvantage,
    FeatureDelta,
    FeatureDiff,
)

# Smallest feature-group change (cp) that counts as a concept appearing.
MIN_GROUP_CP = 8
# A displayed line shows at least the move and the reply, unless the game ends.
MIN_SETTLE_PLIES = 2
# The immediate horizon is the first quiet, non-forcing ply of the kept line:
# after a quiet move that is ply 1, after an exchange the ply where it settles.


def base_display_plies() -> int:
    """Fallback display target when no ply settles; Guid prescribes no fixed count."""
    try:
        return max(1, int(os.environ.get("ENVISIONED_BASE_PLIES", "5")))
    except ValueError:
        return 5


def max_display_plies() -> int:
    """Hard safety cap on the displayed line."""
    try:
        return max(1, int(os.environ.get("ENVISIONED_MAX_PLIES", "12")))
    except ValueError:
        return 12


def probe_trim_enabled() -> bool:
    """Opt-out knob for the probe-based horizon-effect trim. Defaults to enabled."""
    return os.environ.get("ENVISIONED_PROBE_TRIM_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def is_quiescent(board: chess.Board) -> bool:
    """Static quiescence: not in check and nothing en prise by static exchange.

    En prise = a piece *or pawn* of the side not to move that the mover can win
    material on, playing out the capture sequence on that square (SEE). The
    earlier heuristic ignored pawns, so a pawn exchange counted as quiet and
    claims were read mid-exchange (changed 2 Nov 2026)."""
    if board.is_check():
        return False
    return not en_prise_squares(board, not board.turn)


def _is_forcing(board_before: chess.Board, move: chess.Move) -> bool:
    """Capture, promotion, or check — moves a quiet line must not end on."""
    if board_before.is_capture(move) or move.promotion:
        return True
    b = board_before.copy(stack=False)
    b.push(move)
    return b.is_check()


_SIDE_PREFIX = re.compile(r"^(WHITE_|BLACK_)")
_PAWN_GROUP = {"EVALUATE_PAWNS", "PAWN_DOUBLED", "PAWN_ISOLATED", "PAWN_BACKWARD", "WEAK_PAWNS"}


def group_deltas(start: FeatureVector, end: FeatureVector) -> dict[str, int]:
    """Feature-group changes (start -> end) above ``MIN_GROUP_CP``.

    Side and net aliases collapse into one group, and the pawn aggregate and
    its components into one; the strongest member represents the group so
    correlated magnitudes are not summed."""
    groups: dict[str, int] = {}
    for name, value in start.items():
        if name not in end:
            continue
        delta = end[name].value_cp - value.value_cp
        if abs(delta) < MIN_GROUP_CP:
            continue
        base = _SIDE_PREFIX.sub("", name)
        if base in _PAWN_GROUP:
            base = "PAWN_STRUCTURE"
        if abs(delta) > abs(groups.get(base, 0)):
            groups[base] = delta
    return groups


# A ply is settled when its feature story is stable: the groups present there
# persist (same sign, above threshold) through the next PERSIST_WINDOW plies,
# and no new group appears within the look-ahead. Persistence is measured over
# a bounded window, not to the end of the line: an engine line keeps changing
# for twelve plies, so "stable until the end" would only ever select the end.
SETTLE_PERSISTENCE = 0.75
SETTLE_LOOKAHEAD = 2
PERSIST_WINDOW = 4


def _persistence(groups: list[dict[str, int]], k: int) -> float:
    """Share of the groups at ply k that still hold through the next PERSIST_WINDOW plies."""
    at_k = groups[k - 1]
    later = groups[k:k + PERSIST_WINDOW]
    if not at_k or not later:
        return 1.0
    return sum(1 for name, value in at_k.items()
               if all(name in g and g[name] * value > 0 for g in later)) / len(at_k)


def settled_ply(groups: list[dict[str, int]], candidates: list[int]) -> tuple[int | None, str]:
    """The earliest candidate ply whose feature story is stable.

    ``groups[k-1]`` holds the group deltas start -> ply k. Ply k is settled when
    at least ``SETTLE_PERSISTENCE`` of its groups persist through the next
    ``PERSIST_WINDOW`` plies and no group absent at k appears within the next
    ``SETTLE_LOOKAHEAD`` plies. The earliest such ply is the endpoint, so the line stops where the
    move's consequences have materialized rather than at the end of the
    window. When none is settled, the candidate with the best persistence wins
    if at least half of its groups persist; otherwise ``(None, reason)``."""
    n = len(groups)
    for k in candidates:
        at_k = groups[k - 1]
        ahead = groups[k:min(n, k + SETTLE_LOOKAHEAD)]
        if any(name not in at_k for g in ahead for name in g):
            continue
        if _persistence(groups, k) >= SETTLE_PERSISTENCE:
            return k, "settled"
    best: tuple[float, int] | None = None
    for k in candidates:
        if not groups[k - 1]:
            continue
        persistent = _persistence(groups, k)
        if best is None or persistent > best[0]:
            best = (persistent, k)
    if best is not None and best[0] >= 0.5:
        return best[1], "persistence"
    return None, "no_settled_ply"


def build_envisioned_line(
    start_fen: str,
    line_uci: list[str],
    *,
    root_eval_cp: int | None = None,
    depth: int | None = None,
    max_plies: int | None = None,
) -> EnvisionedLine:
    """Choose the endpoint by settlement, preserving terminal forcing lines.

    If no quiet endpoint exists, retain the available tactical continuation
    and mark its evidence unsuitable for strategic claims.
    """
    hard_cap = (
        max(1, max_plies)
        if max_plies is not None
        else max(base_display_plies(), max_display_plies())
    )
    base = min(base_display_plies(), hard_cap)
    board = chess.Board(start_fen)
    start_q = is_quiescent(board)

    moves: list[chess.Move] = []
    sans: list[str] = []
    fens: list[str] = []
    boards_before: list[chess.Board] = []
    for uci in line_uci[:hard_cap]:
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                break
        except Exception:
            break
        boards_before.append(board.copy(stack=False))
        sans.append(board.san(mv))
        board.push(mv)
        moves.append(mv)
        fens.append(board.fen())
        if board.is_game_over():
            break

    walked = len(moves)
    quiet_cache: dict[int, bool] = {}

    def _quiet(k: int) -> bool:
        if k not in quiet_cache:
            quiet_cache[k] = is_quiescent(chess.Board(fens[k - 1]))
        return quiet_cache[k]

    def _unsettled(k: int) -> bool:
        """Ply k ends on a forcing move or a non-quiescent leaf."""
        if k == 0 or chess.Board(fens[k - 1]).is_game_over():
            return False
        return _is_forcing(boards_before[k - 1], moves[k - 1]) or not _quiet(k)

    reason = "fixed_window"
    settled: int | None = None
    if walked and chess.Board(fens[-1]).is_game_over():
        keep, reason = walked, "terminal"
    else:
        candidates = [k for k in range(min(MIN_SETTLE_PLIES, walked), walked + 1) if k and not _unsettled(k)]
        if candidates:
            start_vec = compute_feature_vector_fen(start_fen)
            groups = [group_deltas(start_vec, compute_feature_vector_fen(fen)) for fen in fens]
            settled, reason = settled_ply(groups, candidates)
        if settled is not None:
            keep = settled
        else:
            # Last resort: the fixed window, extended to finish forcing play.
            keep = min(base, walked)
            while keep < walked and _unsettled(keep):
                keep += 1
            if keep and _unsettled(keep):
                quiet_prefixes = [k for k in range(1, keep) if not _unsettled(k)]
                if quiet_prefixes:
                    keep = quiet_prefixes[-1]
            reason = "fixed_window"
    immediate = next((k for k in range(1, keep + 1) if not _unsettled(k)), 0)
    moves = moves[:keep]
    sans = sans[:keep]
    fens = fens[:keep]
    trimmed = walked - keep

    leaf_fen = fens[-1] if fens else start_fen
    leaf_q = is_quiescent(chess.Board(leaf_fen))
    return EnvisionedLine(
        start_fen=start_fen,
        line_uci=[m.uci() for m in moves],
        line_san=sans,
        fens=fens,
        leaf_fen=leaf_fen,
        root_eval_cp=root_eval_cp,
        depth=depth,
        trimmed_plies=trimmed,
        start_quiescent=start_q,
        leaf_quiescent=leaf_q,
        evidence_valid=bool(keep) and not _unsettled(keep),
        immediate_horizon_ply=immediate,
        settled_ply=settled,
        horizon_reason=reason,
    )


def trim_envisioned_line_by_probe(
    line: EnvisionedLine,
    prober: Callable[[str], int | None],
    tolerance_cp: int = 150,
) -> EnvisionedLine:
    """Trim the line's tail while a shallow sanity probe contradicts it.

    A statically quiescent leaf can still be tactically loaded (discovered
    attack, skewer). ``prober(fen)`` returns the engine's cheap eval of a
    position; when the leaf's probe disagrees IN SIGN with the line's backed-up
    eval beyond ``tolerance_cp``, tail plies are dropped one at a time (each
    re-probed) and the longest prefix whose eval story survives is kept.
    Engine-free lines and missing evals pass through untouched."""
    if (
        line.root_eval_cp is None
        or abs(line.root_eval_cp) >= 999000
        or len(line.fens) < 2
        or chess.Board(line.leaf_fen).is_game_over()
    ):
        return line

    def _contradicts(idx: int) -> bool:
        leaf_eval = prober(line.fens[idx])
        if leaf_eval is None:
            return False
        # Both scores are White-POV. Equality has no winning-side story.
        return (line.root_eval_cp > tolerance_cp and leaf_eval < -tolerance_cp) or (
            line.root_eval_cp < -tolerance_cp and leaf_eval > tolerance_cp
        )

    if not _contradicts(len(line.fens) - 1):
        return line
    keep = 0
    for k in range(len(line.fens) - 1, 0, -1):
        before = chess.Board(line.start_fen if k == 1 else line.fens[k - 2])
        move = chess.Move.from_uci(line.line_uci[k - 1])
        leaf = chess.Board(line.fens[k - 1])
        if is_quiescent(leaf) and not _is_forcing(before, move) and not _contradicts(k - 1):
            keep = k
            break
    if not keep:
        return line.model_copy(update={"evidence_valid": False})
    trimmed_line = line.model_copy(
        update={
            "line_uci": line.line_uci[:keep],
            "line_san": line.line_san[:keep],
            "fens": line.fens[:keep],
            "leaf_fen": line.fens[keep - 1],
            "trimmed_plies": line.trimmed_plies + (len(line.fens) - keep),
            "leaf_quiescent": is_quiescent(chess.Board(line.fens[keep - 1])),
            "feature_series": {name: values[:keep + 1] for name, values in line.feature_series.items()},
            "immediate_horizon_ply": min(line.immediate_horizon_ply, keep),
            "settled_ply": min(line.settled_ply, keep) if line.settled_ply else None,
            "horizon_reason": f"{line.horizon_reason}+probe_trim",
        }
    )
    return trimmed_line


def diff_vectors(
    start_vec: FeatureVector,
    leaf_vec: FeatureVector,
    *,
    min_abs_cp: int = 1,
) -> FeatureDiff:
    positive: list[FeatureDelta] = []
    negative: list[FeatureDelta] = []
    for name in leaf_vec:
        before = start_vec.get(name)
        after = leaf_vec.get(name)
        if before is None or after is None:
            continue
        delta = after.value_cp - before.value_cp
        flag_changed = (
            before.flag is not None
            and after.flag is not None
            and before.flag != after.flag
        )
        if abs(delta) < min_abs_cp and not flag_changed:
            continue
        fd = FeatureDelta(
            name=name,
            delta_cp=delta,
            before_cp=before.value_cp,
            after_cp=after.value_cp,
            flag_before=before.flag,
            flag_after=after.flag,
        )
        if delta > 0:
            positive.append(fd)
        elif delta < 0:
            negative.append(fd)
        elif flag_changed:
            # value unchanged but the count moved (rare) — file under positive
            positive.append(fd)
    positive.sort(key=lambda d: -abs(d.delta_cp))
    negative.sort(key=lambda d: -abs(d.delta_cp))
    return FeatureDiff(positive=positive, negative=negative)


def diff_of_diffs(
    played: FeatureDiff,
    alt: FeatureDiff,
    *,
    min_abs_cp: int = 1,
    max_features: int | None = None,
) -> list[FeatureAdvantage]:
    """Feature-diff-of-diffs: for each named feature present in either the
    played move's diff or the alternative's diff, how many more (White-POV)
    centipawns does the ALTERNATIVE'S own start->leaf diff move that feature
    by, compared to the PLAYED move's start->leaf diff on the same feature?

    Every feature name here is already White-POV signed (``WHITE_*``/``BLACK_*``
    are separate, oppositely-signed features per ``vector.py``), so the two
    diffs' ``delta_cp`` values are directly comparable by name."""

    def _by_name(diff: FeatureDiff) -> dict[str, int]:
        return {d.name: d.delta_cp for d in (diff.positive + diff.negative)}

    played_map = _by_name(played)
    alt_map = _by_name(alt)
    names = set(played_map) | set(alt_map)

    advantages: list[FeatureAdvantage] = []
    for name in names:
        played_cp = played_map.get(name, 0)
        alt_cp = alt_map.get(name, 0)
        advantage_cp = alt_cp - played_cp
        if abs(advantage_cp) < min_abs_cp:
            continue
        advantages.append(
            FeatureAdvantage(
                name=name,
                alt_delta_cp=alt_cp,
                played_delta_cp=played_cp,
                advantage_cp=advantage_cp,
            )
        )
    advantages.sort(key=lambda a: (-abs(a.advantage_cp), a.name))
    if max_features is not None:
        advantages = advantages[:max_features]
    return advantages


def envisioned_for_played_move(
    fen_before: str,
    played_uci: str,
    after_pv_uci: list[str],
    *,
    played_eval_cp: int | None,
    depth: int | None,
) -> EnvisionedLine:
    """Envisioned line for the played move: played move + engine continuation."""
    return build_envisioned_line(
        fen_before,
        [played_uci, *list(after_pv_uci or [])],
        root_eval_cp=played_eval_cp,
        depth=depth,
    )


def envisioned_for_best_move(
    fen_before: str,
    best_pv_uci: list[str],
    *,
    best_eval_cp: int | None,
    depth: int | None,
) -> EnvisionedLine:
    """Envisioned line for the engine's best move: PV1 from the pre-move position."""
    return build_envisioned_line(
        fen_before,
        list(best_pv_uci or []),
        root_eval_cp=best_eval_cp,
        depth=depth,
    )
