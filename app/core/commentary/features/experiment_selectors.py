"""Frozen selectors used by the thesis factorial experiment.

The functions in this module are deliberately engine-free.  They consume the
same bounded PV and optional frozen probe results as the repaired baseline, so
an experimental arm cannot gain an accidental search-budget advantage.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import chess

from app.core.commentary.features.envisioned import is_quiescent, max_display_plies
from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.models.comment_facts import Claim, EnvisionedLine


# Frozen development parameters.  Changing any value creates a new selector
# version rather than silently changing the thesis arm.
ADAPTIVE_SELECTOR_VERSION = "adaptive_feature_groups_v2"
ADAPTIVE_RELEVANCE_WEIGHT = 0.55
ADAPTIVE_PERSISTENCE_WEIGHT = 0.30
ADAPTIVE_LENGTH_WEIGHT = 0.10
ADAPTIVE_REDUNDANCY_WEIGHT = 0.05
ADAPTIVE_NEAR_BEST_TOLERANCE = 0.05
ADAPTIVE_MIN_FEATURE_CP = 8
ADAPTIVE_PROBE_TOLERANCE_CP = 150
CONTRASTIVE_SELECTOR_VERSION = "contrastive_v2"
CONTRASTIVE_HORIZON_TOLERANCE_PLIES = 1

Prober = Callable[[str], int | None]


def _walk_bounded_trace(
    start_fen: str, line_uci: list[str]
) -> tuple[list[str], list[str], list[str], list[bool]]:
    board = chess.Board(start_fen)
    ucis: list[str] = []
    sans: list[str] = []
    fens: list[str] = []
    forcing: list[bool] = []
    for token in line_uci[: max_display_plies()]:
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        sans.append(board.san(move))
        was_forcing = board.is_capture(move) or bool(move.promotion)
        board.push(move)
        was_forcing = was_forcing or board.is_check()
        ucis.append(move.uci())
        fens.append(board.fen())
        forcing.append(was_forcing)
        if board.is_game_over():
            break
    return ucis, sans, fens, forcing


def _probe_status(root_eval_cp: int | None, fen: str, prober: Prober | None) -> str:
    if prober is None or root_eval_cp is None or abs(root_eval_cp) >= 999000:
        return "unknown"
    try:
        value = prober(fen)
    except Exception:
        return "unknown"
    if value is None:
        return "unknown"
    contradicted = (
        (root_eval_cp > ADAPTIVE_PROBE_TOLERANCE_CP and value < -ADAPTIVE_PROBE_TOLERANCE_CP)
        or (root_eval_cp < -ADAPTIVE_PROBE_TOLERANCE_CP and value > ADAPTIVE_PROBE_TOLERANCE_CP)
    )
    return "contradicted" if contradicted else "passed"


def _group_deltas(start: dict, end: dict) -> dict[str, int]:
    """Heuristic feature groups, not verified rule concepts.

    Collapse side/net aliases and pawn aggregate/component overlap. Use the
    strongest member instead of summing correlated magnitudes.
    """
    groups: dict[str, int] = {}
    for name, value in start.items():
        if name not in end:
            continue
        delta = end[name].value_cp - value.value_cp
        if abs(delta) < ADAPTIVE_MIN_FEATURE_CP:
            continue
        base = re.sub(r"^(WHITE_|BLACK_)", "", name)
        if base in {"EVALUATE_PAWNS", "PAWN_DOUBLED", "PAWN_ISOLATED", "PAWN_BACKWARD", "WEAK_PAWNS"}:
            base = "PAWN_STRUCTURE"
        if abs(delta) > abs(groups.get(base, 0)):
            groups[base] = delta
    return groups


def select_adaptive_horizon(
    start_fen: str,
    line_uci: list[str],
    *,
    root_eval_cp: int | None,
    depth: int | None,
    prober: Prober | None = None,
) -> tuple[EnvisionedLine, dict]:
    """Select a quiet endpoint from the complete bounded available trace.

    Candidate scores reward distinct heuristic feature groups and persistence,
    while applying small length and redundancy penalties.  Boundary features
    have unknown persistence and receive no stability credit.  The shortest
    endpoint within a frozen tolerance of the best score wins.
    """
    ucis, sans, fens, forcing = _walk_bounded_trace(start_fen, line_uci)
    start_quiet = is_quiescent(chess.Board(start_fen))
    decisions: list[dict] = []

    if fens and chess.Board(fens[-1]).is_game_over():
        line = EnvisionedLine(
            start_fen=start_fen, line_uci=ucis, line_san=sans, fens=fens,
            leaf_fen=fens[-1], root_eval_cp=root_eval_cp, depth=depth,
            trimmed_plies=0, start_quiescent=start_quiet,
            leaf_quiescent=is_quiescent(chess.Board(fens[-1])), evidence_valid=True,
        )
        return line, {"version": ADAPTIVE_SELECTOR_VERSION, "reason": "terminal", "selected_ply": len(fens), "candidates": decisions}

    vectors = {fen: compute_feature_vector_fen(fen) for fen in dict.fromkeys([start_fen, *fens])}
    trace = [_group_deltas(vectors[start_fen], vectors[fen]) for fen in fens]
    eligible: list[tuple[int, dict[str, int]]] = []
    for idx, fen in enumerate(fens, 1):
        quiet = is_quiescent(chess.Board(fen)) and not forcing[idx - 1]
        probe_status = _probe_status(root_eval_cp, fen, prober)
        consistent = probe_status != "contradicted"
        reason = None if quiet and consistent and start_quiet else (
            "nonquiet_start" if not start_quiet else "nonquiet_endpoint" if not quiet else "probe_contradiction"
        )
        deltas = trace[idx - 1] if reason is None else {}
        decision = {"ply": idx, "eligible": reason is None, "excluded_reason": reason,
                    "concepts": sorted(deltas), "feature_groups": sorted(deltas),
                    "probe_status": probe_status, "persistence": "unknown", "score": None}
        decisions.append(decision)
        if reason is None:
            eligible.append((idx, deltas))

    max_relevance = max((sum(abs(v) for v in ds.values()) for _, ds in eligible), default=1) or 1
    previous: set[str] = set()
    scored: list[tuple[int, float]] = []
    for pos, (idx, deltas) in enumerate(eligible):
        later = list(enumerate(trace[idx:], idx + 1))
        if later and deltas:
            persistent = sum(
                1 for name, value in deltas.items()
                if all(name in later_ds and later_ds[name] * value > 0 for _, later_ds in later)
            ) / len(deltas)
            persistence_label = "measured"
        else:
            persistent = 0.0
            persistence_label = "unknown"
        concepts = set(deltas)
        redundancy = len(concepts & previous) / len(concepts) if concepts else 1.0
        relevance = min(1.0, sum(abs(v) for v in deltas.values()) / max_relevance)
        length = idx / max(1, len(fens))
        score = (
            ADAPTIVE_RELEVANCE_WEIGHT * relevance
            + ADAPTIVE_PERSISTENCE_WEIGHT * persistent
            - ADAPTIVE_LENGTH_WEIGHT * length
            - ADAPTIVE_REDUNDANCY_WEIGHT * redundancy
        )
        decisions[idx - 1].update(
            {"persistence": persistence_label, "persistence_score": round(persistent, 6),
             "relevance_score": round(relevance, 6), "redundancy_score": round(redundancy, 6),
             "score": round(score, 6)}
        )
        scored.append((idx, score))
        previous = concepts

    if scored:
        best_score = max(score for _, score in scored)
        selected = min(idx for idx, score in scored if score >= best_score - ADAPTIVE_NEAR_BEST_TOLERANCE)
        valid = True
        reason = "scored_quiet_endpoint"
    else:
        selected = len(fens)
        valid = False
        reason = "no_eligible_quiet_endpoint"
    kept_fens = fens[:selected]
    leaf = kept_fens[-1] if kept_fens else start_fen
    line = EnvisionedLine(
        start_fen=start_fen, line_uci=ucis[:selected], line_san=sans[:selected],
        fens=kept_fens, leaf_fen=leaf, root_eval_cp=root_eval_cp, depth=depth,
        trimmed_plies=len(fens) - selected, start_quiescent=start_quiet,
        leaf_quiescent=is_quiescent(chess.Board(leaf)), evidence_valid=valid,
    )
    return line, {"version": ADAPTIVE_SELECTOR_VERSION, "reason": reason,
                  "selected_ply": selected, "available_plies": len(fens), "candidates": decisions,
                  "evidence_kind": "heuristic_feature_groups",
                  "trajectories": {name: {
                      "first_appearance_ply": next(i for i, ds in enumerate(trace, 1) if name in ds),
                      "supporting_plies": [i for i, ds in enumerate(trace, 1) if name in ds],
                      "loss_plies": [i for i in range(2, len(trace) + 1) if name in trace[i - 2] and (name not in trace[i - 1] or trace[i - 2][name] * trace[i - 1][name] <= 0)],
                  } for name in sorted({name for ds in trace for name in ds})}}


_SQUARE = re.compile(r"\b[a-h][1-8]\b", re.IGNORECASE)


def claim_semantic_identity(claim: Claim) -> tuple:
    """Stable concept identity independent of stylistic sentence wording."""
    return (
        claim.rule_id,
        claim.beneficiary,
        tuple(sorted(claim.features_involved)),
        tuple(sorted(set(s.lower() for s in _SQUARE.findall(claim.text)))),
    )


def select_contrastive_claims(
    played: list[Claim], alternative: list[Claim], *, mover: str,
) -> tuple[list[Claim], list[Claim], dict]:
    """Prioritize branch-specific concepts without increasing claim budgets."""
    from app.core.commentary.rules.constants import MAX_CLAIMS_PER_MOVE, MAX_CONCESSIONS, MAX_ALTERNATIVE_MERITS

    played_ids = [claim_semantic_identity(c) for c in played]
    alt_ids = [claim_semantic_identity(c) for c in alternative]
    alt_set, played_set = set(alt_ids), set(played_ids)

    def rank(claim: Claim, other: set[tuple]) -> tuple[int, int]:
        return (0 if claim_semantic_identity(claim) not in other else 1, -abs(claim.delta_cp))

    material_ids = {"material_won", "material_standing", "tactical_material_loss"}
    ranked = sorted(played, key=lambda c: rank(c, alt_set))
    material = [c for c in ranked if c.rule_id in material_ids]
    # The baseline permits four positional claims at each of its two horizons.
    # Keep that same per-horizon allowance and the same concession ceiling.
    counts: dict[tuple, int] = {}
    selected = []
    concessions = 0
    for claim in material + [c for c in ranked if c.rule_id not in material_ids]:
        concession = claim.beneficiary not in (mover.lower(), None)
        horizon = (claim.support_ply, claim.support_fen)
        if concession and concessions >= MAX_CONCESSIONS:
            continue
        if claim.rule_id not in material_ids:
            if counts.get(horizon, 0) >= MAX_CLAIMS_PER_MOVE:
                continue
            counts[horizon] = counts.get(horizon, 0) + 1
        concessions += int(concession)
        selected.append(claim.model_copy(update={"is_concession": concession}))
    played_ranked = [c for c in selected if not c.is_concession] + [c for c in selected if c.is_concession]
    mover_key = mover.lower()
    eligible_alt = [c for c in alternative if c.beneficiary in (mover_key, None)]
    alt_ranked = sorted(eligible_alt, key=lambda c: (0 if c.rule_id in material_ids else 1, *rank(c, played_set)))[:MAX_ALTERNATIVE_MERITS]
    metadata = {
        "version": CONTRASTIVE_SELECTOR_VERSION,
        "played_only": sum(identity not in alt_set for identity in played_ids),
        "alternative_only": sum(identity not in played_set for identity in alt_ids),
        "shared": len(set(played_ids) & set(alt_ids)),
    }
    return played_ranked, alt_ranked, metadata


def horizons_match(played: EnvisionedLine, alternative: EnvisionedLine | None) -> bool | None:
    if alternative is None:
        return None
    return abs(len(played.fens) - len(alternative.fens)) <= CONTRASTIVE_HORIZON_TOLERANCE_PLIES
