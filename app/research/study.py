"""Validated study/corpus loading for reproducible thesis experiments.

The returned cases contain only the recorded history through the target move.
Reference annotations may live beside corpus entries, but are deliberately not
copied into either generator inputs or the public corpus digest.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from app.evaluation import parse_game


SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
PRIVATE_KEYS = {
    "reference", "reference_comment", "reference_notes", "gold_label",
    "gold_labels", "expected_concepts", "expert_annotation",
}
IMPLEMENTED_VARIANTS = {
    "B00": {"horizon": "current_quiet_bounded", "claims": "current", "renderer": "template"},
    "B10": {"horizon": "adaptive_feature", "claims": "current", "renderer": "template"},
    "B01": {"horizon": "current_quiet_bounded", "claims": "contrastive", "renderer": "template"},
    "B11": {"horizon": "adaptive_feature", "claims": "contrastive", "renderer": "template"},
}
ENGINE_KEYS = {"depth", "probe_depth", "max_plies", "base_plies", "multipv", "threads"}


def _safe_id(value: Any, label: str) -> str:
    text = str(value)
    if not text or SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"unsafe {label}: {text!r}")
    return text


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _public(value: Any) -> Any:
    """Recursively remove fields reserved for blinded reference material."""
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _position_key(board: chess.Board) -> str:
    """Position identity without clocks, canonicalized with its colour mirror."""
    def key(candidate: chess.Board) -> str:
        return " ".join(candidate.fen(en_passant="fen").split()[:4])
    return min(key(board), key(board.mirror()))


def _target_key(board: chess.Board, move: chess.Move) -> tuple[str, str]:
    direct = (" ".join(board.fen(en_passant="fen").split()[:4]), move.uci())
    mirrored = board.mirror()
    mirror_move = chess.Move(
        chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
        promotion=move.promotion, drop=move.drop,
    )
    reverse = (" ".join(mirrored.fen(en_passant="fen").split()[:4]), mirror_move.uci())
    return min(direct, reverse)


def _truncate(game: chess.pgn.Game, target_ply: int) -> tuple[str, chess.Board, chess.Move, chess.Board]:
    moves = list(game.mainline_moves())
    if not 1 <= target_ply <= len(moves):
        raise ValueError(f"target_ply {target_ply} is outside the recorded game (1..{len(moves)})")
    before = game.board()
    for move in moves[: target_ply - 1]:
        before.push(move)
    target = moves[target_ply - 1]
    after = before.copy()
    after.push(target)

    clean = chess.pgn.Game()
    clean.setup(game.board())
    for header in ("Event", "Site", "Date", "Round", "White", "Black"):
        if game.headers.get(header) not in (None, "?"):
            clean.headers[header] = game.headers[header]
    clean.headers["Result"] = "*"
    node = clean
    for move in moves[:target_ply]:
        node = node.add_main_variation(move)
    rendered = clean.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
    # Parse our own artifact so malformed setup headers or illegal reconstruction fail here.
    parse_game(rendered)
    return rendered, before, target, after


def _validate_engine(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != ENGINE_KEYS:
        raise ValueError(f"engine must contain exactly {sorted(ENGINE_KEYS)}")
    if any(type(value[key]) is not int or value[key] < 1 for key in ENGINE_KEYS):
        raise ValueError("all engine settings must be positive integers")
    if value["base_plies"] > value["max_plies"]:
        raise ValueError("engine base_plies cannot exceed max_plies")
    if value["multipv"] < 2:
        raise ValueError("the contrastive variants require engine multipv >= 2")
    if value["threads"] != 1:
        raise ValueError("only deterministic single-thread study execution is implemented")
    return {key: value[key] for key in ENGINE_KEYS}


def _validate_variants(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(IMPLEMENTED_VARIANTS):
        raise ValueError("variants must define B00, B10, B01 and B11 exactly once")
    found: dict[str, dict[str, str]] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"id", "horizon", "claims", "renderer"}:
            raise ValueError("each variant must contain only id, horizon, claims and renderer")
        variant_id = _safe_id(raw["id"], "variant id")
        if variant_id in found:
            raise ValueError(f"duplicate variant id: {variant_id}")
        settings = {key: str(raw[key]) for key in ("horizon", "claims", "renderer")}
        if IMPLEMENTED_VARIANTS.get(variant_id) != settings:
            raise ValueError(f"unimplemented variant settings: {variant_id}")
        found[variant_id] = {"id": variant_id, **settings}
    if set(found) != set(IMPLEMENTED_VARIANTS):
        raise ValueError("variants must define B00, B10, B01 and B11 exactly once")
    return [found[variant_id] for variant_id in ("B00", "B10", "B01", "B11")]


def _validate_evaluation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("evaluation must be an object")
    raters = value.get("rater_ids")
    if not isinstance(raters, list) or len(raters) != 2 or len(set(map(str, raters))) != 2:
        raise ValueError("evaluation.rater_ids must contain two distinct anonymous ids")
    raters = [_safe_id(item, "rater id") for item in raters]
    if value.get("anonymous") is not True or value.get("assignment") != "balanced":
        raise ValueError("evaluation must request anonymous, balanced assignment")
    return {"rater_ids": raters, "anonymous": True, "assignment": "balanced"}


def load_study(config_path: Path) -> dict[str, Any]:
    """Load and validate an executable study configuration and its corpus."""
    config_path = Path(config_path).resolve()
    config = _read_object(config_path)
    required = {"protocol", "experiment_id", "corpus", "dataset_status", "audience", "seed",
                "replicates", "rubric_version", "engine", "variants", "evaluation"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"study config is missing: {', '.join(missing)}")
    protocol = _safe_id(config["protocol"], "protocol")
    experiment_id = _safe_id(config["experiment_id"], "experiment id")
    dataset_status = str(config["dataset_status"])
    if dataset_status not in {"development", "pilot", "diagnostic_not_held_out", "frozen_held_out"}:
        raise ValueError(f"unsupported dataset_status: {dataset_status}")
    audience = str(config["audience"])
    if audience not in {"beginner", "intermediate", "expert"}:
        raise ValueError(f"unsupported audience: {audience}")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if config["replicates"] != 1:
        raise ValueError("only one deterministic core replicate is implemented")
    rubric_version = _safe_id(config["rubric_version"], "rubric version")
    engine = _validate_engine(config["engine"])
    variants = _validate_variants(config["variants"])
    evaluation = _validate_evaluation(config["evaluation"])

    corpus_path = (config_path.parent / str(config["corpus"])).resolve()
    corpus = _read_object(corpus_path)
    raw_cases = corpus.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("corpus must contain a nonempty cases list")
    if "held_out" in {str(item.get("split")) for item in raw_cases if isinstance(item, dict)}:
        if dataset_status != "frozen_held_out" or corpus.get("frozen") is not True:
            raise ValueError("held-out data must be explicitly frozen; diagnostic data cannot masquerade as held-out")

    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    targets: dict[tuple[str, str], str] = {}
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    position_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"corpus case {index} must be an object")
        for field in ("id", "pgn", "target_ply", "source_group", "family_group", "split",
                      "category", "teaching_target", "provenance"):
            if field not in raw:
                raise ValueError(f"corpus case {index} is missing {field}")
        case_id = _safe_id(raw["id"], "case id")
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        source_group = _safe_id(raw["source_group"], "source_group")
        family_group = _safe_id(raw["family_group"], "family_group")
        split = _safe_id(raw["split"], "split")
        category = _safe_id(raw["category"], "category")
        if type(raw["target_ply"]) is not int:
            raise ValueError(f"target_ply must be an integer: {case_id}")
        if type(raw["teaching_target"]) is not bool:
            raise ValueError(f"teaching_target must be boolean: {case_id}")
        if not isinstance(raw["provenance"], dict):
            raise ValueError(f"provenance must be an object: {case_id}")
        game = parse_game(str(raw["pgn"]))
        pgn, before, target, after = _truncate(game, raw["target_ply"])
        target_key = _target_key(before, target)
        if target_key in targets:
            raise ValueError(f"duplicate target: {targets[target_key]} and {case_id}")
        targets[target_key] = case_id
        group_splits[("source", source_group)].add(split)
        group_splits[("family", family_group)].add(split)
        position_splits[_position_key(before)].add(split)
        position_splits[_position_key(after)].add(split)
        split_counts[split] += 1
        cases.append({
            "id": case_id,
            "source_group": source_group,
            "family_group": family_group,
            "split": split,
            "category": category,
            "initial_fen": game.board().fen(),
            "target_ply": raw["target_ply"],
            "fen_before": before.fen(),
            "target_uci": target.uci(),
            "pgn": pgn,
            "input_hash": _digest(pgn),
            "teaching_target": raw["teaching_target"],
            "provenance": _public(raw["provenance"]),
        })

    leaks = [f"{kind}:{name}" for (kind, name), splits in group_splits.items() if len(splits) > 1]
    leaks += [f"position:{key}" for key, splits in position_splits.items() if len(splits) > 1]
    if leaks:
        raise ValueError("train/test split leakage: " + ", ".join(sorted(leaks)))

    public_corpus = {
        "version": corpus.get("version"),
        "dataset_status": dataset_status,
        "cases": cases,
    }
    return {
        "protocol": protocol,
        "experiment_id": experiment_id,
        "dataset_status": dataset_status,
        "audience": audience,
        "seed": config["seed"],
        "replicates": 1,
        "rubric_version": rubric_version,
        "engine": engine,
        "variants": variants,
        "evaluation": evaluation,
        "corpus_hash": _digest(public_corpus),
        "split_audit": {
            "passed": True,
            "case_count": len(cases),
            "split_counts": dict(sorted(split_counts.items())),
            "checks": ["unique_ids", "unique_targets", "source_groups", "family_groups",
                       "pre_and_post_positions", "colour_mirrors", "future_moves_hidden"],
        },
        "cases": cases,
    }
