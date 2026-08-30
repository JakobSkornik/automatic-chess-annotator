"""Evaluation & performance harness for the annotation backend.

Runs the full pipeline (engine pass + rule engine + template composer, no LLM)
over PGNs from ``data/pgn/manual/`` (or a custom path) and reports:

Correctness metrics (the Guid contract):
  - coverage            % of mid/end plies that received any comment
  - claim_precision_proxy  % of claims that clear thresholds AND reference
                          features present in the stored diff vector
  - silence_discipline  comments with zero claims but a dubious move quality
                        (should be ~0: fallback claim or silence is expected)
  - forbidden_phrases   hits in template output (must be 0)
  - contract_integrity  every CommentFacts renders through the template
                        without raising; PV tokens parse as legal SAN
  - refutation_findings count of plausible-but-refuted moves found

Performance metrics:
  - wall time per phase (engine pass / facts+rules / assembly)
  - engine searches per ply (counted via a wrapper)
  - rules fired per commented move, claims per move

Usage:
    python scripts/eval_harness.py [--pgn-dir data/pgn/manual] [--limit N]
                                   [--json out.json]

Offline-safe: skips games if no stockfish binary is found next to the repo
root and says so explicitly.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.disable(logging.WARNING)  # keep harness output readable

import chess  # noqa: E402

from app.core.commentary.forbidden_phrases import scrub_forbidden  # noqa: E402
from app.core.commentary.phases.composer import (  # noqa: E402
    render_facts_template,
)
from app.core.engine.analysis.constants import DEFAULT_ANALYSIS_DEPTH  # noqa: E402
from app.core.io.pgn_reader import PGNReader  # noqa: E402
from app.models.chess_events import AnalyzedMoveData, MoveEvent  # noqa: E402


@dataclass
class GameReport:
    path: str
    plies_total: int = 0
    plies_commentable: int = 0
    plies_commented: int = 0
    claims_total: int = 0
    claims_with_features_in_diff: int = 0
    muted_claims: int = 0
    fallback_silence_moves: int = 0
    forbidden_hits: int = 0
    render_errors: int = 0
    refutations: int = 0
    engine_calls: int = 0
    engine_seconds: float = 0.0
    facts_seconds: float = 0.0
    key_moments: int = 0
    errors: list[str] = field(default_factory=list)


class CountingEngineWrapper:
    """Wraps EngineConnector.analyse to count calls and time spent.

    Explicitly declares the engine-facing surface (analyse,
    analyse_with_depth_snapshots, evaluate_position) so ``__getattr__``
    delegation can never mask a missing method as a silent no-op."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.seconds = 0.0

    def _wrap(self, fn, *args, **kwargs):
        self.calls += 1
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - t0

    def analyse(self, board, *args, **kwargs):
        return self._wrap(self._inner.analyse, board, *args, **kwargs)

    def evaluate_position(self, fen: str, *, depth: int):
        return self._wrap(self._inner.evaluate_position, fen, depth=depth)

    def analyse_with_depth_snapshots(self, board, *args, **kwargs):
        return self._wrap(
            self._inner.analyse_with_depth_snapshots, board, *args, **kwargs
        )

    def __getattr__(self, name):  # delegate the rest
        return getattr(self._inner, name)


def _iter_pgn_games(path: str) -> list[str]:
    """Each file may hold multiple games; split on game-terminator lines."""
    out: list[str] = []
    files = (
        sorted(glob.glob(os.path.join(path, "*.pgn")))
        if os.path.isdir(path)
        else [path]
    )
    for fp in files:
        try:
            text = open(fp, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        # A new game starts where headers begin again after a result line.
        chunks = []
        current: list[str] = []
        seen_result = False
        for line in text.splitlines():
            if line.startswith("[Event ") and seen_result and current:
                chunks.append("\n".join(current))
                current = []
                seen_result = False
            current.append(line)
            if line.strip() in ("1-0", "0-1", "1/2-1/2", "*"):
                seen_result = True
        if current:
            chunks.append("\n".join(current))
        out.extend(c.strip() for c in chunks if c.strip())
    return out


def evaluate_game(
    pgn_string: str, path: str, connector
) -> tuple[GameReport, dict | None]:
    from app.core.commentary.event_extractor import ChessEventExtractor
    from app.core.commentary.openings.eco_book import ECOBook
    from app.core.commentary.phase_classifier import PhaseClassifier
    from app.core.engine.analysis.engine_pass import AnalysisRetriever
    from app.core.engine.analysis.key_moments import _promote_key_moment
    from app.core.commentary.rules import build_comment_facts
    from app.core.commentary.features.envisioned import trim_envisioned_line_by_probe

    report = GameReport(path=path)
    wrapped = CountingEngineWrapper(connector)
    try:
        game = PGNReader().read_game_from_string(pgn_string)
    except Exception as e:
        report.errors.append(f"parse: {e}")
        return report, None
    if game is None:
        report.errors.append("parse: no game")
        return report, None

    eco_book = ECOBook()
    retriever = AnalysisRetriever(wrapped, game, eco_book)
    moves_list = retriever.get_move_list()
    report.plies_total = len(moves_list)
    phase_classifier = PhaseClassifier(eco_book)

    rows: list[AnalyzedMoveData] = []
    prev_obj = None
    for idx, move_obj in enumerate(moves_list):
        board_after = chess.Board(move_obj.position)
        uci_prefix = retriever._uci_prefix_for_depth(move_obj.depth)
        move_obj.phase = phase_classifier.classify(board_after, uci_prefix)
        t0 = time.perf_counter()
        try:
            if move_obj.phase == "early":
                analyzed, pvs = retriever.analyze_book_move(move_obj)
            else:
                analyzed, pvs = retriever.analyze_move(
                    move_obj, stage=DEFAULT_ANALYSIS_DEPTH
                )
        except Exception as e:
            report.errors.append(f"engine ply {idx}: {e}")
            break
        report.engine_seconds += time.perf_counter() - t0
        engine_meta = {}
        if isinstance(analyzed.hiddenFeatures, dict):
            engine_meta = analyzed.hiddenFeatures.get("_engine") or {}
        row = AnalyzedMoveData(
            index=idx,
            ply=move_obj.depth,
            san=_san_for(game, idx, move_obj.move),
            uci=move_obj.move,
            fen_before=_fen_before(game, idx),
            fen_after=move_obj.position,
            score_cp=analyzed.score,
            phase_raw=str(move_obj.phase),
            pvs=pvs or [],
            hidden_features=analyzed.hiddenFeatures or {},
            analyzed_move=analyzed,
            eval_at_depth=dict(engine_meta.get("eval_at_depth") or {}),
            pv1_change_count=int(engine_meta.get("pv1_change_count", 0)),
        )
        rows.append(row)
        prev_obj = analyzed
    report.engine_calls = wrapped.calls

    extractor = ChessEventExtractor(eco_book=eco_book, key_moment_detector=retriever.key_moment_detector)
    t0 = time.perf_counter()
    try:
        events = extractor.extract_events(game, rows)
    except Exception as e:
        report.errors.append(f"extract: {e}")
        return report, None

    prober = (lambda f: wrapped.evaluate_position(f, depth=6)) if True else None
    for mi, me in enumerate(events):
        if (me.phase == "opening") or not (0 <= me.move_index < len(rows)):
            continue
        row = rows[me.move_index]
        report.plies_commentable += 1
        tf0 = time.perf_counter()
        try:
            facts = build_comment_facts(row, me, depth=DEFAULT_ANALYSIS_DEPTH)
            if facts is not None and facts.display_line is not None:
                _assert_pv_parses(
                    "[pv:" + " ".join(facts.display_line.line_san) + "]",
                    facts,
                    report,
                    tag="raw",
                )
            if facts is not None and prober and facts.display_line is not None and facts.claims:
                trimmed = trim_envisioned_line_by_probe(facts.display_line, prober)
                if trimmed is not facts.display_line:
                    facts = facts.model_copy(update={"display_line": trimmed})
                    if facts.display_line is not None:
                        _assert_pv_parses(
                            "[pv:" + " ".join(facts.display_line.line_san) + "]",
                            facts,
                            report,
                            tag="trimmed",
                        )
        except Exception as e:
            report.render_errors += 1
            report.errors.append(f"facts ply {me.ply}: {e}")
            continue
        report.facts_seconds += time.perf_counter() - tf0
        if facts is None:
            continue
        km_new = _promote_key_moment(me, facts, facts.claims)
        if km_new:
            report.key_moments += 1
        # Template rendering must always succeed (deterministic fallback).
        try:
            text = render_facts_template(facts)
            _, hits = scrub_forbidden(text)
            report.forbidden_hits += len(hits)
            if "[pv:" in text:
                _assert_pv_parses(text, facts, report)
        except Exception as e:
            report.render_errors += 1
            report.errors.append(f"render ply {me.ply}: {e}")
            continue
        if facts.claims:
            report.plies_commented += 1
            report.claims_total += len(facts.claims)
            diff_names = set()
            if facts.feature_diff is not None:
                diff_names = {
                    d.name for d in facts.feature_diff.positive + facts.feature_diff.negative
                }
            report.muted_claims += len(facts.muted_claims)
            for c in facts.claims:
                if c.features_involved and diff_names and set(c.features_involved) & diff_names:
                    report.claims_with_features_in_diff += 1
            quality = me.move_quality.value if me.move_quality else ""
            if quality in ("inaccuracy", "mistake", "blunder") and not facts.claims:
                report.fallback_silence_moves += 1
        report.refutations += len(me.refutations or [])
    return report, {"rows": len(rows), "events": len(events)}


def _san_for(game, idx: int, uci: str) -> str:
    board = chess.Board(_fen_before(game, idx))
    try:
        return board.san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def _fen_before(game, idx: int) -> str:
    board = game.board()
    for i, gm in enumerate(game.mainline_moves()):
        if i >= idx:
            break
        board.push(gm)
    return board.fen()


def _pv_start_fens(facts) -> list[str]:
    """Every position a [pv:] token in this comment may be rooted at: the played
    line's start, and the alternative's (its line starts from the same pre-move
    position but with a different first move)."""
    fens: list[str] = []
    if facts.display_line is not None:
        fens.append(facts.display_line.start_fen)
    alt = facts.better_alternative
    if alt is not None and alt.display_line is not None:
        fens.append(alt.display_line.start_fen)
    return [f for f in fens if f]


def _replay_failure(sans: list[str], start_fen: str) -> str | None:
    """The first SAN that does not replay from ``start_fen``, or None if all do."""
    board = chess.Board(start_fen)
    for san in sans:
        try:
            board.push(board.parse_san(san))
        except Exception as e:
            return f"'{san}': {e}"
    return None


def _assert_pv_parses(text: str, facts, report: GameReport, tag: str = "") -> None:
    """A [pv:] token must replay as legal SAN from one of the comment's lines.

    Each move has to be *pushed* before parsing the next, and a token may belong
    to the alternative rather than the played line — checking every move against
    one static board reports the whole line as illegal from its second move on.
    """
    import re

    candidates = _pv_start_fens(facts)
    for m in re.finditer(r"\[pv:([^\]]+)\]", text):
        sans = m.group(1).split()
        if not sans or not candidates:
            continue
        failures = [_replay_failure(sans, fen) for fen in candidates]
        if any(f is None for f in failures):
            continue
        report.errors.append(
            f"pv token[{tag}] illegal from any line start: {failures[0]}"
        )
        report.render_errors += 1
        return


def summarize(reports: list[GameReport]) -> dict:
    total_commentable = sum(r.plies_commentable for r in reports)
    total_claimed = sum(r.claims_total for r in reports)
    return {
        "games": len(reports),
        "plies_total": sum(r.plies_total for r in reports),
        "commentable_plies": total_commentable,
        "commented_plies": sum(r.plies_commented for r in reports),
        "coverage_pct": round(
            100.0
            * sum(r.plies_commented for r in reports)
            / max(total_commentable, 1),
            2,
        ),
        "claims_total": total_claimed,
        "claims_per_commented_move": round(
            total_claimed / max(sum(r.plies_commented for r in reports), 1), 2
        ),
        "claim_grounding_pct": round(
            100.0
            * sum(r.claims_with_features_in_diff for r in reports)
            / max(total_claimed, 1),
            2,
        ),
        "forbidden_phrase_hits": sum(r.forbidden_hits for r in reports),
        "render_errors": sum(r.render_errors for r in reports),
        "muted_claims": sum(r.muted_claims for r in reports),
        "refutation_findings": sum(r.refutations for r in reports),
        "key_moment_promotions": sum(r.key_moments for r in reports),
        "engine_calls_per_ply": round(
            sum(r.engine_calls for r in reports)
            / max(sum(r.plies_total for r in reports), 1),
            2,
        ),
        "avg_engine_seconds_per_game": round(
            statistics.mean([r.engine_seconds for r in reports]) if reports else 0, 2
        ),
        "avg_facts_ms_per_move": round(
            1000.0
            * sum(r.facts_seconds for r in reports)
            / max(total_commentable, 1),
            3,
        ),
        "games_with_errors": sum(1 for r in reports if r.errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn-dir", default=os.path.join("data", "pgn", "manual"))
    ap.add_argument("--limit", type=int, default=10, help="max games to analyze")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    from app.core.engine.engine_connector import get_global_engine_connector

    try:
        connector = get_global_engine_connector()
    except Exception as e:
        print(f"OFFLINE: no engine available ({e}); nothing to measure.")
        return 1

    games = _iter_pgn_games(os.path.abspath(args.pgn_dir))[: args.limit]
    if not games:
        print(f"No PGN games found under {args.pgn_dir}")
        return 1

    reports: list[GameReport] = []
    for i, g in enumerate(games, 1):
        rep, _extra = evaluate_game(g, f"game_{i}", connector)
        reports.append(rep)
        status = "ok" if not rep.errors else f"{len(rep.errors)} errors"
        print(
            f"[{i}/{len(games)}] {rep.plies_total} plies, "
            f"commented {rep.plies_commented}/{rep.plies_commentable}, "
            f"claims {rep.claims_total}, engine {rep.engine_seconds:.1f}s "
            f"({status})"
        )

    summary = summarize(reports)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    err_sample = [e for r in reports for e in r.errors][:20]
    if err_sample:
        print("\nFirst errors:")
        for e in err_sample:
            print(f"  - {e}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "errors": err_sample}, f, indent=2)
        print(f"\nWrote {args.json_out}")
    try:
        connector.close()
    except Exception:
        pass
    # os._exit skips interpreter shutdown, so buffered output would be lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # Stockfish child can linger on Windows; exit hard after output


if __name__ == "__main__":
    main()

