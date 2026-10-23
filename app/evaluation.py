"""Independent, deterministic audits of exported annotation artifacts.

These checks establish structural/evidence consistency, not semantic chess
precision or educational value. Missing evidence fails rather than passing.
"""
from __future__ import annotations

import io
import re
import statistics
from collections import Counter

import chess
import chess.pgn

from app.core.commentary.features.envisioned import is_quiescent
from app.core.commentary.features.guid_features import compute_feature_vector_fen


def parse_game(pgn: str) -> chess.pgn.Game:
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or not list(game.mainline_moves()):
        raise ValueError("Expected one nonempty, legal PGN game")
    if chess.pgn.read_game(stream) is not None:
        raise ValueError("Expected exactly one game")
    if not game.board().is_valid():
        raise ValueError("Invalid starting position")
    return game


def audit_game(pgn: str, data: dict, exported_pgn: str, required_plies: list[int]) -> dict:
    source = parse_game(pgn)
    board = source.board()
    expected = list(source.mainline_moves())
    checks: dict[str, dict] = {}
    issues = []
    words = []

    def check(name: str, ok: bool, detail: str = ""):
        c = checks.setdefault(name, {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += int(bool(ok))
        if not ok:
            issues.append({"check": name, "detail": detail})

    def line_checks(line: dict, origin: str, label: str):
        sans, fens = line.get("san", []), line.get("fens", [])
        valid = line.get("start_fen") == origin and len(sans) == len(fens) and bool(sans)
        b = chess.Board(origin)
        for i, san in enumerate(sans):
            try:
                b.push_san(san)
                valid = valid and i < len(fens) and b.fen() == fens[i]
            except ValueError:
                valid = False
                break
        check("legal_evidence_lines", valid, label)
        check("quietness_flags", line.get("start_quiescent") == is_quiescent(chess.Board(origin))
              and line.get("leaf_quiescent") == is_quiescent(b), label)
        check("evidence_endpoint", line.get("leaf_fen") == b.fen(), label)
        series = line.get("feature_series", {})
        aligned = bool(series) and all(len(v) == len(fens) + 1 for v in series.values())
        if aligned and valid:
            for idx, fen in enumerate([origin, *fens]):
                vec = compute_feature_vector_fen(fen)
                if any(name not in vec or values[idx] != vec[name].value_cp for name, values in series.items()):
                    aligned = False
                    break
        check("feature_series_consistency", aligned, label)
        return b

    check("commentary_complete", data.get("commentary_complete") is True)
    moves = data.get("moves", [])
    check("mainline_length", len(moves) == len(expected))
    for ply in required_plies:
        check("required_moment_coverage", 1 <= ply <= len(moves) and moves[ply - 1].get("final_comment") is True,
              f"ply {ply}")
    for ply, expected_move in enumerate(expected, 1):
        if ply > len(moves):
            break
        gm = moves[ply - 1]
        origin = board.fen()
        expected_san = board.san(expected_move)
        check("move_identity", gm.get("uci") == expected_move.uci() and gm.get("san") == expected_san
              and gm.get("mn") == board.fullmove_number
              and gm.get("color") == ("w" if board.turn else "b"), f"ply {ply}")
        board.push(expected_move)
        check("mainline_positions", gm.get("fen") == board.fen(), f"ply {ply}")
        text = gm.get("comment") or ""
        if gm.get("final_comment"):
            check("usable_final_comment", bool(text.strip()) and "unavailable" not in text.lower(), f"ply {ply}")
            words.append(len(re.findall(r"\S+", re.sub(r"\[pv:[^\]]*\]", "", text))))
        facts = gm.get("comment_facts")
        if board.is_game_over():
            result = board.result()
            check("terminal_score", gm.get("score") is None, f"ply {ply}: terminal positions use the result")
            check("terminal_result", bool(facts) and facts.get("terminal_result") == result, f"ply {ply}: {result}")
            if board.is_checkmate():
                winner = "White" if not board.turn else "Black"
                check("checkmate_winner_text", winner in text and "checkmate" in text.lower()
                      and f"for {'Black' if winner == 'White' else 'White'}" not in text, f"ply {ply}")
        if not facts:
            continue
        available = []
        for branch, record in (("main", facts), ("alternative", facts.get("better_alternative") or {})):
            line = record.get("display_line")
            if not line:
                continue
            available.append(line.get("san", []))
            leaf = line_checks(line, origin, f"ply {ply} {branch}")
            for claim in record.get("claims", []):
                i = claim.get("support_ply")
                fens = line.get("fens", [])
                valid = (type(i) is int and 1 <= i <= len(fens)
                         and claim.get("support_fen") == fens[i - 1]
                         and claim.get("baseline_fen") == origin
                         and claim.get("realization") == ("immediate" if i == 1 else "envisioned"))
                check("claim_provenance", valid, f"ply {ply} {claim.get('rule_id')}")
                if valid:
                    support = chess.Board(fens[i - 1])
                    check("quiet_claim_evidence", is_quiescent(chess.Board(origin)) and is_quiescent(support)
                          and line.get("evidence_valid") is True and not leaf.is_game_over(), f"ply {ply}")
        for token in re.findall(r"\[pv:([^\]]+)\]", text):
            check("shown_line_matches_evidence", token.split() in available, f"ply {ply}: {token}")
    try:
        exported = parse_game(exported_pgn)
        ok = (exported.board().fen() == source.board().fen()
              and list(exported.mainline_moves()) == expected
              and exported.headers.get("Result") == source.headers.get("Result"))
        for node in exported.mainline():
            if node.board().is_game_over():
                check("terminal_pgn_evaluation", node.eval() is None and not (node.nags & set(range(10, 20))),
                      "Terminal PGN nodes must use the result, not an ambiguous mate-zero evaluation")
    except ValueError:
        ok = False
    check("pgn_roundtrip", ok)
    # Explicit counts, not one misleading overall 'quality score'.
    return {"checks": checks, "issues": issues, "plies": len(moves),
            "final_comments": sum(bool(m.get("final_comment")) for m in moves),
            "prose_word_counts": words,
            "renderings": dict(Counter(v for m in moves for v in ((m.get("debug") or {}).get("renderings") or {}).values()))}


def aggregate(reports: list[dict]) -> dict:
    totals = {}
    for report in reports:
        for name, c in report.get("audit", {}).get("checks", {}).items():
            dest = totals.setdefault(name, {"passed": 0, "total": 0})
            for field in ("passed", "total"):
                dest[field] += c[field]
    for c in totals.values():
        c["rate"] = c["passed"] / c["total"] if c["total"] else None
    seconds = [r["runtime"]["seconds"] for r in reports if "runtime" in r]
    words = [w for r in reports for w in r.get("audit", {}).get("prose_word_counts", [])]
    renderings = Counter()
    for r in reports:
        renderings.update(r.get("audit", {}).get("renderings", {}))
    return {"runs": len(reports), "failed_runs": sum(r.get("error") is not None for r in reports),
            "checks": totals,
            "renderings": dict(renderings),
            "runtime_seconds": {"n": len(seconds), "mean": statistics.mean(seconds) if seconds else None,
                                "median": statistics.median(seconds) if seconds else None,
                                "max": max(seconds) if seconds else None},
            "llm_calls": sum(r.get("runtime", {}).get("llm_calls", 0) for r in reports),
            "llm_failed_calls": sum(r.get("runtime", {}).get("llm_failed_calls", 0) for r in reports),
            "tokens": sum(r.get("runtime", {}).get("tokens", 0) for r in reports),
            "prose_words": {"n": len(words), "mean": statistics.mean(words) if words else None,
                            "max": max(words) if words else None},
            "final_comments": sum(r.get("audit", {}).get("final_comments", 0) for r in reports)}
