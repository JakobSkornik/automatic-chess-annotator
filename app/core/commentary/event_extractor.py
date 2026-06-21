"""Semantic chess event extraction from engine-analyzed moves."""

from __future__ import annotations

import chess
import chess.pgn

from app.core.commentary.features.move_category import classify_move_event
from app.core.commentary.features.plan_extractor import build_plan_comparison
from app.core.commentary.features.pv_motif_scan import (
    merge_pv_motifs_into_strategic,
    merge_pv_motifs_into_tactical,
    scan_pv_motifs,
)
from app.core.commentary.features.strategic_motifs import detect_strategic_motifs
from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.core.commentary.key_moment_detector import (
    KeyMomentDetector,
    decisive_eval_cp,
)
from app.core.commentary.openings.eco_book import ECOBook
from app.models.chess_events import (
    AnalyzedMoveData,
    MoveEvent,
    MoveEventType,
    MoveQuality,
    PlanComparison,
)
from app.models.Move import Move

# --- Move-quality eval-loss bands (centipawns, mover-POV) ---
BEST_MOVE_MAX_LOSS_CP = 5
EXCELLENT_MAX_LOSS_CP = 20
GOOD_MAX_LOSS_CP = 50
INACCURACY_MAX_LOSS_CP = 100
MISTAKE_MAX_LOSS_CP = 200

# --- Criticality / event-classification thresholds (centipawns) ---
CRITICAL_MOVE_LOSS_CP = 50  # eval loss that alone marks a move critical
PV_MOTIF_SCAN_PLIES = 4  # plies of engine PV1 scanned for motifs
HIDDEN_INFLECTION_SPREAD_CP = (
    100  # multi-depth eval spread flagging a hidden inflection
)
HIDDEN_INFLECTION_MAX_LOSS_CP = (
    80  # instability flags critical only below this own-loss
)


def _map_phase(raw: str) -> str:
    if raw == "early":
        return "opening"
    if raw == "end":
        return "endgame"
    return "middlegame"


def _move_quality_from_loss(eval_loss_cp: int, played_is_best: bool) -> MoveQuality:
    if played_is_best or eval_loss_cp <= BEST_MOVE_MAX_LOSS_CP:
        return MoveQuality.BEST
    if eval_loss_cp <= EXCELLENT_MAX_LOSS_CP:
        return MoveQuality.EXCELLENT
    if eval_loss_cp <= GOOD_MAX_LOSS_CP:
        return MoveQuality.GOOD
    if eval_loss_cp <= INACCURACY_MAX_LOSS_CP:
        return MoveQuality.INACCURACY
    if eval_loss_cp <= MISTAKE_MAX_LOSS_CP:
        return MoveQuality.MISTAKE
    return MoveQuality.BLUNDER


def _eval_loss_white_pov(
    board_before: chess.Board, best_white: int, played_white: int
) -> int:
    if board_before.turn == chess.WHITE:
        return max(0, best_white - played_white)
    return max(0, played_white - best_white)


def _resolve_move(
    row: AnalyzedMoveData, mainline: list[chess.Move]
) -> chess.Move | None:
    try:
        return chess.Move.from_uci(row.uci)
    except Exception:
        return mainline[row.index] if row.index < len(mainline) else None


def _best_move_and_eval(
    board_before: chess.Board, pvs: list
) -> tuple[str | None, str | None, int | None]:
    """(uci, san, white-POV cp) of the engine's top move from the PVs."""
    if not pvs or not pvs[0]:
        return None, None, None
    first = pvs[0][0]
    if not getattr(first, "move", None):
        return None, None, None
    best_uci = str(first.move)
    best_eval = first.score if first.score is not None else None
    best_san: str | None = None
    try:
        bm = chess.Move.from_uci(best_uci)
        if bm in board_before.legal_moves:
            best_san = board_before.san(bm)
    except Exception:
        best_san = best_uci
    return best_uci, best_san, best_eval


def _move_quality_for(
    loss: int, best_eval: int | None, cur_score: int | None, played_is_best: bool
) -> MoveQuality:
    """Quality from eval loss; demoted to GOOD in already-decided positions (Guid)."""
    move_quality = _move_quality_from_loss(loss, played_is_best)
    negative = (MoveQuality.INACCURACY, MoveQuality.MISTAKE, MoveQuality.BLUNDER)
    if move_quality in negative and best_eval is not None and cur_score is not None:
        decisive = decisive_eval_cp()
        if abs(best_eval) > decisive and abs(cur_score) > decisive:
            move_quality = MoveQuality.GOOD
    return move_quality


def _pawn_and_material(hidden_features: dict) -> tuple[str | None, dict | None]:
    ps = (
        hidden_features.get("pawnStructure")
        if isinstance(hidden_features.get("pawnStructure"), dict)
        else {}
    )
    pawn_type = ps.get("centerType") if isinstance(ps, dict) else None
    mat = (
        hidden_features.get("material")
        if isinstance(hidden_features.get("material"), dict)
        else None
    )
    return pawn_type, mat


def _pv1_ucis(pvs: list) -> list[str]:
    if not pvs or not pvs[0]:
        return []
    return [str(u) for pm in pvs[0] if (u := getattr(pm, "move", None))]


def _eval_instability(row: AnalyzedMoveData) -> int | None:
    """Spread between the shallowest and deepest re-search evals (search noise)."""
    ead = row.eval_at_depth or {}
    if len(ead) < 2:
        return None
    vals = list(ead.values())
    return max(vals) - min(vals)


def _plan_comparison_for(fen_before: str, pvs: list, uci: str) -> PlanComparison:
    pt, bt, ps, bs, pchain, bchain, ptags, btags = build_plan_comparison(
        fen_before, pvs, uci
    )
    return PlanComparison(
        played_target_squares=pt,
        best_target_squares=bt,
        played_plan_seed=ps,
        best_plan_seed=bs,
        played_recurring_destinations=pchain,
        best_recurring_destinations=bchain,
        played_plan_tags=ptags,
        best_plan_tags=btags,
    )


def _is_critical_move(
    key_moment: str | None,
    swing: int | None,
    motifs: list,
    loss: int,
    event_type: MoveEventType,
    instability: int | None,
    in_book_ply: bool,
    swing_critical: int,
) -> bool:
    if in_book_ply:
        return False
    return bool(
        key_moment
        or (swing is not None and abs(swing) >= swing_critical)
        or bool(motifs)
        or loss >= CRITICAL_MOVE_LOSS_CP
        or event_type != MoveEventType.QUIET
        or (
            instability is not None
            and instability >= HIDDEN_INFLECTION_SPREAD_CP
            and loss < HIDDEN_INFLECTION_MAX_LOSS_CP
        )
    )


def _to_move_event(
    row: AnalyzedMoveData,
    *,
    phase: str,
    prev_score: int | None,
    cur_score: int | None,
    swing: int | None,
    move_quality: MoveQuality,
    event_type: MoveEventType,
    motifs: list,
    strat_motifs: list,
    is_critical: bool,
    best_san: str | None,
    best_uci: str | None,
    best_eval: int | None,
    mat: dict | None,
    pawn_type: str | None,
    opening_name: str | None,
    opening_eco: str | None,
    key_moment: str | None,
) -> MoveEvent:
    """Map the computed per-move signals onto a categorized MoveEvent."""
    event = MoveEvent(
        move_index=row.index,
        ply=row.ply,
        san=row.san,
        uci=row.uci,
        fen_before=row.fen_before,
        fen_after=row.fen_after,
        phase=phase,
        eval_before_cp=int(prev_score) if prev_score is not None else None,
        eval_after_cp=int(cur_score) if cur_score is not None else None,
        eval_swing_cp=int(swing) if swing is not None else None,
        move_quality=move_quality,
        event_type=event_type,
        tactical_motifs=motifs,
        strategic_motifs=strat_motifs,
        is_critical=is_critical,
        best_move_san=best_san,
        best_move_uci=best_uci,
        best_move_eval_cp=int(best_eval) if best_eval is not None else None,
        material_balance=mat,
        pawn_structure_type=str(pawn_type) if pawn_type else None,
        opening_name=opening_name,
        opening_eco=opening_eco,
        key_moment_type=key_moment,
    )
    return event.model_copy(update={"move_category": classify_move_event(event)})


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
        """Build the per-move MoveEvent list (quality, motifs, key moments, opening) from engine-analyzed rows."""
        mainline = list(game.mainline_moves())
        events: list[MoveEvent] = []
        prev_move_obj: Move | None = previous_engine_move
        for i, row in enumerate(analyzed_rows):
            ch_move = _resolve_move(row, mainline)
            if ch_move is None:
                continue
            event = self._event_for_row(game, mainline, row, ch_move, prev_move_obj)
            analyzed_rows[i] = row.model_copy(
                update={"key_moment_type": event.key_moment_type}
            )
            events.append(event)
            prev_move_obj = row.analyzed_move
        return events

    def _event_for_row(
        self,
        game: chess.pgn.Game,
        mainline: list[chess.Move],
        row: AnalyzedMoveData,
        ch_move: chess.Move,
        prev_move_obj: Move | None,
    ) -> MoveEvent:
        analyzed = row.analyzed_move
        pvs = row.pvs if isinstance(row.pvs, list) else []
        board_before = chess.Board(row.fen_before)
        board_after = chess.Board(row.fen_after)
        prev_score = (
            prev_move_obj.score
            if prev_move_obj and prev_move_obj.score is not None
            else None
        )
        cur_score = analyzed.score if analyzed and analyzed.score is not None else None

        best_uci, best_san, best_eval = _best_move_and_eval(board_before, pvs)
        played_is_best = bool(best_uci and best_uci == row.uci)
        loss = (
            _eval_loss_white_pov(board_before, best_eval, cur_score)
            if best_eval is not None and cur_score is not None
            else 0
        )
        move_quality = _move_quality_for(loss, best_eval, cur_score, played_is_best)
        swing = (
            cur_score - prev_score
            if prev_score is not None and cur_score is not None
            else None
        )

        hidden_features = (
            row.hidden_features if isinstance(row.hidden_features, dict) else {}
        )
        pawn_type, mat = _pawn_and_material(hidden_features)
        in_book_ply = (row.phase_raw or "") == "early"
        key_moment = self._detect_key_moment(
            analyzed, prev_move_obj, pvs, row, loss, move_quality, in_book_ply
        )
        opening_name, opening_eco = self._opening_for(game, mainline, row.index)
        phase = _map_phase(row.phase_raw or (analyzed.phase or "mid"))

        motifs, strat_motifs = self._collect_motifs(
            board_before,
            board_after,
            ch_move,
            hidden_features,
            row,
            pvs,
            prev_score,
            cur_score,
            swing,
            phase,
        )
        event_type = self._event_type_for(
            key_moment, played_is_best, loss, swing, pawn_type
        )
        is_critical = _is_critical_move(
            key_moment,
            swing,
            motifs,
            loss,
            event_type,
            _eval_instability(row),
            in_book_ply,
            self.EVAL_SWING_CRITICAL,
        )
        return _to_move_event(
            row,
            phase=phase,
            prev_score=prev_score,
            cur_score=cur_score,
            swing=swing,
            move_quality=move_quality,
            event_type=event_type,
            motifs=motifs,
            strat_motifs=strat_motifs,
            is_critical=is_critical,
            best_san=best_san,
            best_uci=best_uci,
            best_eval=best_eval,
            mat=mat,
            pawn_type=pawn_type,
            opening_name=opening_name,
            opening_eco=opening_eco,
            key_moment=key_moment,
        )

    def _collect_motifs(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        ch_move: chess.Move,
        hidden_features: dict,
        row: AnalyzedMoveData,
        pvs: list,
        prev_score: int | None,
        cur_score: int | None,
        swing: int | None,
        phase: str,
    ) -> tuple[list, list]:
        """Tactical + strategic motifs, merged with the PV1 motif scan."""
        pv_ucis = _pv1_ucis(pvs)
        motifs = detect_tactical_motifs(
            board_before, board_after, ch_move, prev_score, cur_score
        )
        strat_motifs = detect_strategic_motifs(
            board_before,
            board_after,
            ch_move,
            hidden_features,
            eval_after_cp=int(cur_score) if cur_score is not None else None,
            eval_before_cp=int(prev_score) if prev_score is not None else None,
            eval_swing_cp=int(swing) if swing is not None else None,
            phase=phase,
            best_pv_ucis=pv_ucis or None,
            plan_comparison=_plan_comparison_for(row.fen_before, pvs, row.uci),
        )
        scans = scan_pv_motifs(
            board_before, pv_ucis, max_plies=PV_MOTIF_SCAN_PLIES, phase=phase
        )
        return (
            merge_pv_motifs_into_tactical(motifs, scans),
            merge_pv_motifs_into_strategic(strat_motifs, scans),
        )

    def _detect_key_moment(
        self,
        analyzed: Move,
        prev_move_obj: Move | None,
        pvs: list,
        row: AnalyzedMoveData,
        loss: int,
        move_quality: MoveQuality,
        in_book_ply: bool,
    ) -> str | None:
        # Book plies carry no engine data and are never key moments.
        key_moment = (
            None
            if in_book_ply
            else self._key_moment_detector.detect(
                analyzed, prev_move_obj, pvs, pv1_change_count=row.pv1_change_count
            )
        )
        if not key_moment:
            instability = _eval_instability(row)
            if (
                instability is not None
                and instability >= HIDDEN_INFLECTION_SPREAD_CP
                and loss < INACCURACY_MAX_LOSS_CP
            ):
                key_moment = "hidden_inflection"
        if not key_moment and move_quality == MoveQuality.BLUNDER:
            key_moment = "blunder"
        elif not key_moment and move_quality == MoveQuality.MISTAKE:
            key_moment = "mistake"
        return key_moment

    def _opening_for(
        self, game: chess.pgn.Game, mainline: list[chess.Move], idx: int
    ) -> tuple[str | None, str | None]:
        seq_uci: list[str] = []
        board = game.board()
        for j, gm in enumerate(mainline):
            if j > idx:
                break
            seq_uci.append(board.uci(gm))
            board.push(gm)
        if not seq_uci:
            return None, None
        info, _matched_ply = self._eco.match(seq_uci)
        return (info.name, info.code) if info else (None, None)

    def _event_type_for(
        self,
        key_moment: str | None,
        played_is_best: bool,
        loss: int,
        swing: int | None,
        pawn_type: str | None,
    ) -> MoveEventType:
        if key_moment == "missed_opportunity":
            event_type = MoveEventType.MISSED_TACTIC
        elif key_moment == "critical_decision":
            event_type = MoveEventType.CRITICAL_DECISION
        elif (
            pawn_type and self._prev_pawn_center and pawn_type != self._prev_pawn_center
        ):
            event_type = MoveEventType.STRUCTURAL_CHANGE
        elif played_is_best and loss <= EXCELLENT_MAX_LOSS_CP:
            event_type = MoveEventType.BEST_MOVE_PLAYED
        elif loss >= INACCURACY_MAX_LOSS_CP and not played_is_best:
            event_type = MoveEventType.POSITIONAL_CONCESSION
        elif (
            swing is not None and abs(swing) >= self.EVAL_SWING_CRITICAL
        ) or key_moment in (
            "blunder",
            "mistake",
            "inaccuracy",
        ):
            event_type = MoveEventType.EVAL_SWING
        else:
            event_type = MoveEventType.QUIET
        self._prev_pawn_center = pawn_type or self._prev_pawn_center
        return event_type
