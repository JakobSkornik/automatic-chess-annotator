"""Semantic chess event extraction from engine-analyzed moves."""

from __future__ import annotations

import chess
import chess.pgn
from typing import Any, Dict, List, Optional

from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.openings.eco_book import ECOBook
from app.models.Move import Move
from app.models.chess_events import (
    AnalyzedMoveData,
    MoveEvent,
    MoveEventType,
    MoveQuality,
)


def board_before_mainline_index(game: chess.pgn.Game, move_index: int) -> chess.Board:
    board = game.board()
    for i, gm in enumerate(game.mainline_moves()):
        if i >= move_index:
            break
        board.push(gm)
    return board


def build_analyzed_row_from_interactive(
    game: chess.pgn.Game,
    move_idx: int,
    analyzed_move: Move,
    pvs: Optional[List[List[Move]]],
) -> AnalyzedMoveData:
    """Build AnalyzedMoveData for a single WebSocket-analyzed move (same shape as batch pipeline)."""
    board_before = board_before_mainline_index(game, move_idx)
    fen_before = board_before.fen()
    try:
        uci = str(analyzed_move.move)
        ch = chess.Move.from_uci(uci)
        san = board_before.san(ch)
    except Exception:
        san = str(analyzed_move.move)
        uci = str(analyzed_move.move)
    return AnalyzedMoveData(
        index=move_idx,
        ply=analyzed_move.depth,
        san=san,
        uci=uci,
        fen_before=fen_before,
        fen_after=analyzed_move.position,
        score_cp=analyzed_move.score,
        phase_raw=str(analyzed_move.phase or "mid"),
        pvs=list(pvs) if pvs else [],
        hidden_features=analyzed_move.hiddenFeatures or {},
        trace=analyzed_move.trace,
        captured_by_white=analyzed_move.capturedByWhite or {},
        captured_by_black=analyzed_move.capturedByBlack or {},
        analyzed_move=analyzed_move,
    )


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

    def __init__(self, eco_book: Optional[ECOBook] = None) -> None:
        self._eco = eco_book or ECOBook()
        self._key_moment_detector = KeyMomentDetector()
        self._prev_pawn_center: Optional[str] = None

    def extract_events(
        self,
        game: chess.pgn.Game,
        analyzed_rows: List[AnalyzedMoveData],
        *,
        previous_engine_move: Optional[Move] = None,
    ) -> List[MoveEvent]:
        mainline = list(game.mainline_moves())
        events: List[MoveEvent] = []
        prev_move_obj: Optional[Move] = previous_engine_move
        score_history: List[int] = []

        for row in analyzed_rows:
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
            prev_score = prev_move_obj.score if prev_move_obj and prev_move_obj.score is not None else None
            cur_score = analyzed.score if analyzed and analyzed.score is not None else None

            best_uci: Optional[str] = None
            best_san: Optional[str] = None
            best_eval: Optional[int] = None
            pv_lines: List[Dict[str, Any]] = []
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
                uci_moves = [str(m.move) for m in pv_seq[:8] if getattr(m, "move", None)]
                san_line: List[str] = []
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

            swing: Optional[int] = None
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
            ps = (hf.get("pawnStructure") or {}) if isinstance(hf.get("pawnStructure"), dict) else {}
            pawn_type = ps.get("centerType") if isinstance(ps, dict) else None
            mat = hf.get("material") if isinstance(hf.get("material"), dict) else None
            wk = (hf.get("white") or {}).get("kingExposure") if isinstance(hf.get("white"), dict) else None
            bk = (hf.get("black") or {}).get("kingExposure") if isinstance(hf.get("black"), dict) else None
            king_safety = None
            if isinstance(wk, (int, float)) and isinstance(bk, (int, float)):
                king_safety = {"white": float(wk), "black": float(bk)}

            seq_uci: List[str] = []
            bb2 = game.board()
            for j, gm in enumerate(mainline):
                if j > idx:
                    break
                seq_uci.append(bb2.uci(gm))
                bb2.push(gm)
            opening_name: Optional[str] = None
            opening_eco: Optional[str] = None
            if seq_uci:
                info = self._eco.match(seq_uci)
                if info:
                    opening_eco = getattr(info, "code", None)
                    opening_name = getattr(info, "name", None)

            km = self._key_moment_detector.detect(analyzed, prev_move_obj, pvs)

            phase = _map_phase(row.phase_raw or (analyzed.phase or "mid"))

            event_type = MoveEventType.QUIET
            if km == "missed_opportunity":
                event_type = MoveEventType.MISSED_TACTIC
            elif km == "critical_decision":
                event_type = MoveEventType.CRITICAL_DECISION
            elif pawn_type and self._prev_pawn_center and pawn_type != self._prev_pawn_center:
                event_type = MoveEventType.STRUCTURAL_CHANGE
            elif played_is_best and loss <= 20:
                event_type = MoveEventType.BEST_MOVE_PLAYED
            elif loss >= 100 and not played_is_best:
                event_type = MoveEventType.POSITIONAL_CONCESSION
            elif swing is not None and abs(swing) >= self.EVAL_SWING_CRITICAL:
                event_type = MoveEventType.EVAL_SWING
            elif km in ("blunder", "mistake", "inaccuracy"):
                event_type = MoveEventType.EVAL_SWING
            self._prev_pawn_center = pawn_type or self._prev_pawn_center

            if cur_score is not None:
                score_history.append(int(cur_score))
            trend = score_history[-self.SCORE_TREND_WINDOW :]

            is_critical = bool(
                km
                or (swing is not None and abs(swing) >= self.EVAL_SWING_CRITICAL)
                or bool(motifs)
                or loss >= 50
                or event_type != MoveEventType.QUIET
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
            )
            events.append(ev)
            prev_move_obj = analyzed

        return events
