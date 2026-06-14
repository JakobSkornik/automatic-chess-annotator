"""Semantic chess event extraction from engine-analyzed moves."""

from __future__ import annotations

from typing import Any

import chess
import chess.pgn

from app.core.commentary.features.delta_to_motif import infer_motifs_from_deltas
from app.core.commentary.features.move_category import classify_move_event
from app.core.commentary.features.opponent_threats import detect_opponent_threats
from app.core.commentary.features.plan_extractor import build_plan_comparison
from app.core.commentary.features.pv_motif_scan import (
    merge_pv_motifs_into_strategic,
    merge_pv_motifs_into_tactical,
    scan_pv_motifs,
)
from app.core.commentary.features.strategic_motifs import detect_strategic_motifs
from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.openings.eco_book import ECOBook
from app.models.chess_events import (
    AnalyzedMoveData,
    MoveEvent,
    MoveEventType,
    MoveQuality,
    PlanComparison,
)
from app.models.Move import Move


def _map_phase(raw: str) -> str:
    if raw == "early":
        return "opening"
    if raw == "end":
        return "endgame"
    return "middlegame"


def _move_quality_from_loss(eval_loss_cp: int, played_is_best: bool) -> MoveQuality:
    if played_is_best or eval_loss_cp <= 5:
        return MoveQuality.BEST
    if eval_loss_cp <= 20:
        return MoveQuality.EXCELLENT
    if eval_loss_cp <= 50:
        return MoveQuality.GOOD
    if eval_loss_cp <= 100:
        return MoveQuality.INACCURACY
    if eval_loss_cp <= 200:
        return MoveQuality.MISTAKE
    return MoveQuality.BLUNDER


def _eval_loss_white_pov(
    board_before: chess.Board, best_white: int, played_white: int
) -> int:
    if board_before.turn == chess.WHITE:
        return max(0, best_white - played_white)
    return max(0, played_white - best_white)


class ChessEventExtractor:
    EVAL_SWING_CRITICAL = 80
    SCORE_TREND_WINDOW = 6

    def __init__(
        self,
        eco_book: ECOBook | None = None,
        key_moment_detector: KeyMomentDetector | None = None,
    ) -> None:
        self._eco = eco_book or ECOBook()
        self._key_moment_detector = key_moment_detector or KeyMomentDetector()
        self._prev_pawn_center: str | None = None

    def extract_events(
        self,
        game: chess.pgn.Game,
        analyzed_rows: list[AnalyzedMoveData],
        *,
        previous_engine_move: Move | None = None,
    ) -> list[MoveEvent]:
        mainline = list(game.mainline_moves())
        events: list[MoveEvent] = []
        prev_move_obj: Move | None = previous_engine_move
        score_history: list[int] = []

        for i, row in enumerate(analyzed_rows):
            idx = row.index
            board_before = chess.Board(row.fen_before)
            board_after = chess.Board(row.fen_after)
            uci = row.uci
            try:
                ch_move = chess.Move.from_uci(uci)
            except Exception:
                ch_move = mainline[idx] if idx < len(mainline) else None
            if ch_move is None:
                continue

            analyzed = row.analyzed_move
            pvs = row.pvs if isinstance(row.pvs, list) else []
            prev_score = (
                prev_move_obj.score
                if prev_move_obj and prev_move_obj.score is not None
                else None
            )
            cur_score = (
                analyzed.score if analyzed and analyzed.score is not None else None
            )

            best_uci: str | None = None
            best_san: str | None = None
            best_eval: int | None = None
            pv_lines: list[dict[str, Any]] = []
            if pvs and pvs[0]:
                first = pvs[0][0]
                if getattr(first, "move", None):
                    best_uci = str(first.move)
                    best_eval = first.score if first.score is not None else None
                    try:
                        b = board_before.copy()
                        bm = chess.Move.from_uci(best_uci)
                        if bm in b.legal_moves:
                            best_san = b.san(bm)
                    except Exception:
                        best_san = best_uci
            for rank, pv_seq in enumerate(pvs):
                if not pv_seq:
                    continue
                fm = pv_seq[0]
                sc = fm.score if fm.score is not None else None
                uci_moves = [
                    str(m.move) for m in pv_seq[:8] if getattr(m, "move", None)
                ]
                san_line: list[str] = []
                bb = board_before.copy()
                for u in uci_moves:
                    try:
                        mm = chess.Move.from_uci(u)
                        san_line.append(bb.san(mm))
                        bb.push(mm)
                    except Exception:
                        break
                pv_lines.append({"rank": rank + 1, "line_san": san_line, "eval_cp": sc})

            played_is_best = bool(best_uci and best_uci == uci)
            loss = 0
            if best_eval is not None and cur_score is not None:
                loss = _eval_loss_white_pov(board_before, best_eval, cur_score)
            mq = _move_quality_from_loss(loss, played_is_best)

            swing: int | None = None
            if prev_score is not None and cur_score is not None:
                swing = cur_score - prev_score

            motifs = detect_tactical_motifs(
                board_before,
                board_after,
                ch_move,
                prev_score,
                cur_score,
            )

            hf = row.hidden_features if isinstance(row.hidden_features, dict) else {}
            ps = (
                (hf.get("pawnStructure") or {})
                if isinstance(hf.get("pawnStructure"), dict)
                else {}
            )
            pawn_type = ps.get("centerType") if isinstance(ps, dict) else None
            mat = hf.get("material") if isinstance(hf.get("material"), dict) else None
            wk = (
                (hf.get("white") or {}).get("kingExposure")
                if isinstance(hf.get("white"), dict)
                else None
            )
            bk = (
                (hf.get("black") or {}).get("kingExposure")
                if isinstance(hf.get("black"), dict)
                else None
            )
            king_safety = None
            if isinstance(wk, (int, float)) and isinstance(bk, (int, float)):
                king_safety = {"white": float(wk), "black": float(bk)}

            seq_uci: list[str] = []
            bb2 = game.board()
            for j, gm in enumerate(mainline):
                if j > idx:
                    break
                seq_uci.append(bb2.uci(gm))
                bb2.push(gm)
            opening_name: str | None = None
            opening_eco: str | None = None
            if seq_uci:
                info, _matched_ply = self._eco.match(seq_uci)
                if info:
                    opening_eco = info.code
                    opening_name = info.name

            km = self._key_moment_detector.detect(
                analyzed, prev_move_obj, pvs, pv1_change_count=row.pv1_change_count
            )
            in_book_ply = (row.phase_raw or "") == "early"
            if in_book_ply:
                # Book plies carry no engine data and are commented by the
                # opening commenter — never as key moments.
                km = None

            eval_instability_cp: int | None = None
            ead = row.eval_at_depth or {}
            if len(ead) >= 2:
                vals = list(ead.values())
                eval_instability_cp = max(vals) - min(vals)
            if (
                not km
                and eval_instability_cp is not None
                and eval_instability_cp >= 100
                and loss < 100
            ):
                km = "hidden_inflection"

            if not km and mq == MoveQuality.BLUNDER:
                km = "blunder"
            elif not km and mq == MoveQuality.MISTAKE:
                km = "mistake"

            analyzed_rows[i] = analyzed_rows[i].model_copy(
                update={"key_moment_type": km}
            )

            phase = _map_phase(row.phase_raw or (analyzed.phase or "mid"))

            pv_ucis: list[str] = []
            if pvs and pvs[0]:
                for pm in pvs[0]:
                    u = getattr(pm, "move", None)
                    if u:
                        pv_ucis.append(str(u))

            pt, bt, ps, bs, pchain, bchain, ptags, btags = build_plan_comparison(
                row.fen_before, pvs, uci
            )
            plan_cmp = PlanComparison(
                played_target_squares=pt,
                best_target_squares=bt,
                played_plan_seed=ps,
                best_plan_seed=bs,
                played_recurring_destinations=pchain,
                best_recurring_destinations=bchain,
                played_plan_tags=ptags,
                best_plan_tags=btags,
            )

            strat_motifs = detect_strategic_motifs(
                board_before,
                board_after,
                ch_move,
                hf,
                eval_after_cp=int(cur_score) if cur_score is not None else None,
                eval_before_cp=int(prev_score) if prev_score is not None else None,
                eval_swing_cp=int(swing) if swing is not None else None,
                phase=phase,
                best_pv_ucis=pv_ucis or None,
                plan_comparison=plan_cmp,
            )

            # PV motif scan along engine PV1 from root
            pv_motif_scans = scan_pv_motifs(
                board_before,
                pv_ucis,
                max_plies=4,
                phase=phase,
            )
            motifs = merge_pv_motifs_into_tactical(motifs, pv_motif_scans)
            strat_motifs = merge_pv_motifs_into_strategic(strat_motifs, pv_motif_scans)

            # Map PV horizon feature deltas to strategic motifs
            if row.pv_horizon_diff is not None:
                mover_color = board_before.turn
                delta_motifs = infer_motifs_from_deltas(
                    row.pv_horizon_diff,
                    mover=mover_color,
                    phase=phase,
                )
                seen_strat = set(strat_motifs)
                for dm in delta_motifs:
                    if dm not in seen_strat:
                        seen_strat.add(dm)
                        strat_motifs.append(dm)

            # Opponent threat scan from fen_after
            opp_threats = detect_opponent_threats(
                board_after,
                best_pv_ucis=pv_ucis or None,
                played_matches_best=played_is_best,
            )

            event_type = MoveEventType.QUIET
            if km == "missed_opportunity":
                event_type = MoveEventType.MISSED_TACTIC
            elif km == "critical_decision":
                event_type = MoveEventType.CRITICAL_DECISION
            elif (
                pawn_type
                and self._prev_pawn_center
                and pawn_type != self._prev_pawn_center
            ):
                event_type = MoveEventType.STRUCTURAL_CHANGE
            elif played_is_best and loss <= 20:
                event_type = MoveEventType.BEST_MOVE_PLAYED
            elif loss >= 100 and not played_is_best:
                event_type = MoveEventType.POSITIONAL_CONCESSION
            elif (
                swing is not None and abs(swing) >= self.EVAL_SWING_CRITICAL
            ) or km in ("blunder", "mistake", "inaccuracy"):
                event_type = MoveEventType.EVAL_SWING
            self._prev_pawn_center = pawn_type or self._prev_pawn_center

            if cur_score is not None:
                score_history.append(int(cur_score))
            trend = score_history[-self.SCORE_TREND_WINDOW :]

            is_critical = (not in_book_ply) and bool(
                km
                or (swing is not None and abs(swing) >= self.EVAL_SWING_CRITICAL)
                or bool(motifs)
                or loss >= 50
                or event_type != MoveEventType.QUIET
                or (
                    eval_instability_cp is not None
                    and eval_instability_cp >= 100
                    and loss < 80
                )
            )

            ev = MoveEvent(
                move_index=idx,
                ply=row.ply,
                san=row.san,
                uci=uci,
                fen_before=row.fen_before,
                fen_after=row.fen_after,
                phase=phase,
                eval_before_cp=int(prev_score) if prev_score is not None else None,
                eval_after_cp=int(cur_score) if cur_score is not None else None,
                eval_swing_cp=int(swing) if swing is not None else None,
                move_quality=mq,
                event_type=event_type,
                tactical_motifs=motifs,
                strategic_motifs=strat_motifs,
                plan_comparison=plan_cmp,
                is_critical=is_critical,
                best_move_san=best_san,
                best_move_uci=best_uci,
                best_move_eval_cp=int(best_eval) if best_eval is not None else None,
                pv_lines=pv_lines,
                material_balance=mat,
                king_safety=king_safety,
                pawn_structure_type=str(pawn_type) if pawn_type else None,
                score_trend=[int(x) for x in trend],
                opening_name=opening_name,
                opening_eco=opening_eco,
                key_moment_type=km,
                eval_instability_cp=eval_instability_cp,
                pv_horizon_diff=row.pv_horizon_diff,
                pv_motifs=pv_motif_scans,
                opponent_threats=opp_threats,
            )
            ev = ev.model_copy(
                update={"move_category": classify_move_event(ev, future_delta=None)}
            )
            events.append(ev)
            prev_move_obj = analyzed

        return events
