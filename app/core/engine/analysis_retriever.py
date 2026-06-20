from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import chess
from chess.pgn import Game

from app.core.commentary.event_extractor import ChessEventExtractor
from app.core.commentary.features.guid_features import (
    CHART_FEATURES,
    compute_feature_vector,
    vector_to_plain,
)
from app.core.commentary.features.positional_features import compute_hidden_features
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.openings.eco_book import (
    ECOBook,
    detect_opening,
    is_absent_opening_header,
    merge_opening_with_headers,
    parse_pgn_eco_tag,
)
from app.core.commentary.phase_classifier import PhaseClassifier
from app.core.engine.engine_connector import EngineConnector
from app.core.io.pgn_reader import PGNReader
from app.models.chess_events import (
    AnalyzedMoveData,
    GameAnalysisContext,
    MoveEvent,
)
from app.models.GameJson import (
    AnalysisInfo,
    FeatureRef,
    FeatureSeries,
    GameJson,
    GameMetadata,
    GameMove,
    MoveScore,
    Variation,
)
from app.models.Move import AnalysisStage, Move
from app.models.PgnMetadata import PgnMetadata

# ANALYSIS_STAGES = [0.05, 0.1, 0.2, 0.4]
ANALYSIS_STAGES = [4, 8, 16]
DEFAULT_ANALYSIS_DEPTH = ANALYSIS_STAGES[-1]
DEFAULT_PV_COUNT = 3
MATE_SCORE = 1000000
FEATURE_DIFF_MAX_ENTRIES = 10
logger = logging.getLogger(__name__)


def _apply_back_to_back_key_moment_suppression(
    move_events: list[MoveEvent],
    context: GameAnalysisContext,
) -> None:
    """Second of two same-type key moments within 2 plies loses LLM pass; gets stub reference."""
    last_ply: int | None = None
    last_type: str | None = None
    for mi, me in enumerate(move_events):
        km = me.key_moment_type
        if not km:
            continue
        if km in ("blunder", "mistake", "critical_decision"):
            last_ply, last_type = me.ply, km
            continue
        if last_type == km and last_ply is not None and (me.ply - last_ply) <= 2:
            updated = me.model_copy(
                update={
                    "key_moment_type": None,
                    "brief_commentary": True,
                    "commentary_stub_ref_ply": last_ply,
                    "is_critical": True,
                }
            )
            move_events[mi] = updated
            context.move_events[mi] = updated
        else:
            last_ply, last_type = me.ply, km


def _cp_and_pv1(info_any: Any) -> tuple[int | None, str | None]:
    """White-POV centipawns and the first PV move (UCI) from an engine info blob."""
    inf = info_any[0] if isinstance(info_any, list) and info_any else info_any
    if not isinstance(inf, dict):
        return None, None
    sc = inf.get("score")
    if sc is None:
        return None, None
    try:
        cp = int(sc.white().score(mate_score=MATE_SCORE))
    except Exception:
        return None, None
    pv = inf.get("pv") or []
    uci = pv[0].uci() if pv else None
    return cp, uci


def _compute_features_delta(before: dict, after: dict) -> dict:
    """Strategic before/after deltas (open files, pawn counts, boolean flags)."""
    delta: dict = {"white": {}, "black": {}, "openFiles": {}}
    try:
        # Open files count deltas
        if isinstance(before.get("openFiles"), dict) and isinstance(
            after.get("openFiles"), dict
        ):
            try:
                delta["openFiles"]["openCount"] = len(
                    after["openFiles"].get("open", [])
                ) - len(before["openFiles"].get("open", []))
                delta["openFiles"]["semiOpenWhiteCount"] = len(
                    after["openFiles"].get("semiOpenWhite", [])
                ) - len(before["openFiles"].get("semiOpenWhite", []))
                delta["openFiles"]["semiOpenBlackCount"] = len(
                    after["openFiles"].get("semiOpenBlack", [])
                ) - len(before["openFiles"].get("semiOpenBlack", []))
            except Exception:
                pass

        def _list_or_int_len(d: dict, key: str) -> int:
            v = d.get(key)
            if isinstance(v, list):
                return len(v)
            if isinstance(v, int):
                return v
            return 0

        for side in ("white", "black"):
            b = before.get(side, {}) if isinstance(before.get(side, {}), dict) else {}
            a = after.get(side, {}) if isinstance(after.get(side, {}), dict) else {}

            def diff_num(key: str, b=b, a=a, side=side):
                if isinstance(b.get(key), int) and isinstance(a.get(key), int):
                    delta[side][key] = a[key] - b[key]

            for k in ("doubledPawns", "isolatedPawns", "passedPawns"):
                delta[side][k] = _list_or_int_len(a, k) - _list_or_int_len(b, k)
            for k in (
                "attackedPieces",
                "attackingPieces",
                "rooksOnOpenFiles",
                "rooksOnSemiOpenFiles",
            ):
                diff_num(k)
            # Booleans as changed flags
            for k in (
                "hasBishopPair",
                "canCastleKingSide",
                "canCastleQueenSide",
                "connectedRooks",
            ):
                if isinstance(b.get(k), bool) and isinstance(a.get(k), bool):
                    if a[k] != b[k]:
                        delta[side][f"{k}Changed"] = True
        return delta
    except Exception:
        return {}


class AnalysisRetriever:
    def __init__(
        self,
        engine_connector: EngineConnector,
        game: Game,
        eco_book: ECOBook | None = None,
    ):
        self.analysis_stages = ANALYSIS_STAGES
        self.engine_connector = engine_connector
        self.game = game
        self._eco_book = eco_book or ECOBook()
        self.id_counter = 0
        self.analyzed_game: list[Move] = self.get_move_list()
        self.key_moment_detector = KeyMomentDetector()

    def _uci_prefix_for_depth(self, depth: int) -> list[str]:
        """First ``depth`` mainline half-moves as UCI (same order as ``Move.depth``)."""
        out: list[str] = []
        board = self.game.board()
        for i, gm in enumerate(self.game.mainline_moves()):
            if i >= depth:
                break
            out.append(board.uci(gm))
            board.push(gm)
        return out

    def get_pgn_headers(self) -> PgnMetadata:
        """Extract game metadata from PGN headers."""
        headers = self.game.headers

        return PgnMetadata(
            whiteName=headers.get("White", ""),
            blackName=headers.get("Black", ""),
            whiteElo=(
                int(headers.get("WhiteElo", 0))
                if headers.get("WhiteElo", "").isdigit()
                else None
            ),
            blackElo=(
                int(headers.get("BlackElo", 0))
                if headers.get("BlackElo", "").isdigit()
                else None
            ),
            event=headers.get("Event", ""),
            opening=headers.get("Opening", ""),
            eco=parse_pgn_eco_tag(headers),
            result=headers.get("Result", ""),
        )

    def get_game_id(self) -> int:
        """
        Returns a unique ID for the game.
        This is used to identify the game in the database.
        """
        self.id_counter += 1
        return self.id_counter

    def get_move_list(self) -> list[Move]:
        """
        Returns a list of moves in the game.
        If PGN parsing failed during init, this will return an empty list.
        """
        moves = []
        board = self.game.board()

        depth = 1
        for chess_move in self.game.mainline_moves():
            board_before_move = board.copy()
            san_representation = board.uci(chess_move)
            board.push(chess_move)
            fen_after_move = board.fen()

            move_obj = Move(
                id=self.get_game_id(),
                position=fen_after_move,
                move=san_representation,
                context="mainline",
                isAnalyzed=False,
                depth=depth,
                piece=self._get_piece_for_move(
                    board_before_move=board_before_move, san_move=san_representation
                ),
            )
            moves.append(move_obj)
            depth += 1

        return moves

    def analyze_book_move(self, main_move_obj: Move) -> tuple[Move, list[list[Move]]]:
        """Opening-book ply: static features and metadata only — no engine calls at all."""
        board_after_move = chess.Board(main_move_obj.position)
        uci_prefix = self._uci_prefix_for_depth(main_move_obj.depth)
        _info_book, matched_ply = self._eco_book.match(uci_prefix)

        main_move_obj.phase = "early"
        main_move_obj.score = None
        try:
            after_features = compute_hidden_features(
                board_after_move, in_opening_book_phase=True
            )
        except Exception as e:
            logger.error(
                f"Hidden features (book) failed at depth {main_move_obj.depth} "
                f"FEN={board_after_move.fen()}: {e}"
            )
            after_features = {"error": str(e)}
        if isinstance(after_features, dict):
            after_features.setdefault("_engine", {})
            after_features["_engine"]["opening_book_hit"] = True
            after_features["_engine"]["opening_matched_ply"] = int(matched_ply)
        main_move_obj.hiddenFeatures = after_features
        (
            main_move_obj.capturedByWhite,
            main_move_obj.capturedByBlack,
        ) = self._get_all_captured_pieces(board_after_move)
        main_move_obj.isAnalyzed = True
        main_move_obj.analysisStage = AnalysisStage.FINAL
        main_move_obj.analysisVersion = 1
        return main_move_obj, []

    def evaluate_position(self, fen: str, *, depth: int) -> int | None:
        """One-off White-POV cp eval of a position (used to seed the book-exit baseline)."""
        try:
            board = chess.Board(fen)
            info = self.engine_connector.analyse(board, depth=depth, multiPv=1)
            if isinstance(info, list) and info:
                info = info[0]
            if not isinstance(info, dict):
                return None
            sc = info.get("score")
            if sc is None:
                return None
            return int(sc.white().score(mate_score=MATE_SCORE))
        except Exception as e:
            logger.warning("Seed eval failed for FEN %s: %s", fen, e)
            return None

    def analyze_move(
        self, main_move_obj: Move, stage: float
    ) -> tuple[Move, list[list[Move]]]:
        """Compute main move eval on the after-move position, and PVs on the position before the move."""
        # Board AFTER the move (stored in Move.position) – used for the move's own eval/metadata
        board_after_move = chess.Board(main_move_obj.position)

        # Evaluate the position after the move for the move's score and trace
        after_results = self.engine_connector.analyse(
            board_after_move, depth=stage, multiPv=1
        )
        if isinstance(after_results, list):
            after_primary = after_results[0]
        else:
            after_primary = after_results

        # Analyze main move (score refers to the position after the move)
        main_move_obj.score = (
            after_primary.get("score").white().score(mate_score=MATE_SCORE)
        )
        uci_prefix = self._uci_prefix_for_depth(main_move_obj.depth)
        _info_book, matched_ply = self._eco_book.match(uci_prefix)
        in_book = (
            len(uci_prefix) > 0
            and _info_book is not None
            and matched_ply >= len(uci_prefix)
        )
        # Phase is decided by PhaseClassifier before the engine is invoked;
        # this path only runs for out-of-book (mid/end) plies.
        if not main_move_obj.phase:
            main_move_obj.phase = "mid"

        # Multi-depth instability on the after-move position (search swings / PV changes)
        eval_at_depth: dict[int, int] = {}
        pv1_ucis: list[str | None] = []
        stage_i = int(stage)
        for d in (8, 12, 16):
            if d == stage_i:
                inf = after_primary
            else:
                try:
                    inf = self.engine_connector.analyse(
                        board_after_move, depth=d, multiPv=1
                    )
                except Exception:
                    inf = None
            cp, uci = _cp_and_pv1(inf)
            if cp is not None:
                eval_at_depth[d] = cp
            pv1_ucis.append(uci)
        pv1_change_count = sum(
            1
            for i in range(len(pv1_ucis) - 1)
            if pv1_ucis[i] and pv1_ucis[i + 1] and pv1_ucis[i] != pv1_ucis[i + 1]
        )

        # Hidden features on the after-move position + before/after delta for strategic context
        try:
            after_features = compute_hidden_features(
                board_after_move, in_opening_book_phase=in_book
            )
        except Exception as e:
            logger.error(
                f"Hidden features (after) failed at depth {main_move_obj.depth} FEN={board_after_move.fen()}: {e}"
            )
            after_features = {"error": str(e)}
        if isinstance(after_features, dict):
            after_features.setdefault("_engine", {})
            after_features["_engine"]["eval_at_depth"] = eval_at_depth
            after_features["_engine"]["pv1_change_count"] = pv1_change_count
            after_features["_engine"]["opening_book_hit"] = bool(in_book)
            after_features["_engine"]["opening_matched_ply"] = int(matched_ply)
            # Engine continuation from the position after the played move —
            # the played move's "envisioned" line (no extra search needed).
            try:
                after_pv = after_primary.get("pv") or []
                after_features["_engine"]["after_pv_uci"] = [m.uci() for m in after_pv]
            except Exception:
                after_features["_engine"]["after_pv_uci"] = []
        (
            main_move_obj.capturedByWhite,
            main_move_obj.capturedByBlack,
        ) = self._get_all_captured_pieces(board_after_move)
        main_move_obj.isAnalyzed = True
        # Mark stage as final if this is the last configured stage
        main_move_obj.analysisStage = (
            AnalysisStage.FINAL
            if stage == max(self.analysis_stages)
            else AnalysisStage.DEEP
        )
        main_move_obj.analysisVersion = 1

        all_pvs_as_list_of_moves: list[list[Move]] = []

        # Compute PVs from the position BEFORE the move
        # Reconstruct the board before this move using PGN and the move depth
        board_before_move = self.game.board()
        try:
            target_depth = max(0, int(main_move_obj.depth) - 1)
        except Exception:
            target_depth = 0

        for idx, game_move in enumerate(self.game.mainline_moves()):
            if idx >= target_depth:
                break
            board_before_move.push(game_move)

        # Compute features for BEFORE position and a small delta map for AI context
        prefix_before = self._uci_prefix_for_depth(max(0, main_move_obj.depth - 1))
        _ib_bef, _mb_bef = self._eco_book.match(prefix_before)
        in_book_before = (
            len(prefix_before) > 0
            and _ib_bef is not None
            and _mb_bef >= len(prefix_before)
        )
        try:
            before_features = compute_hidden_features(
                board_before_move, in_opening_book_phase=in_book_before
            )
        except Exception as e:
            logger.error(
                f"Hidden features (before) failed at depth {main_move_obj.depth} FEN={board_before_move.fen()}: {e}"
            )
            before_features = {"error": str(e)}

        try:
            positional_delta = _compute_features_delta(before_features, after_features)
        except Exception:
            positional_delta = {}

        # Attach features in a backward-compatible way: keep after at top-level, nest before/delta under _ai
        main_move_obj.hiddenFeatures = after_features
        try:
            if isinstance(main_move_obj.hiddenFeatures, dict):
                main_move_obj.hiddenFeatures.setdefault("_ai", {})
                main_move_obj.hiddenFeatures["_ai"]["before"] = before_features
                main_move_obj.hiddenFeatures["_ai"]["delta"] = positional_delta
        except Exception:
            pass

        pv_results = self.engine_connector.analyse(
            board_before_move, depth=stage, multiPv=DEFAULT_PV_COUNT
        )

        # Process each PV variation (from the BEFORE-move position)
        for pv_idx, pv_data_from_engine in enumerate(pv_results):
            current_pv_as_moves_list: list[Move] = []
            board_for_this_pv = chess.Board(board_before_move.fen())

            pv_score_for_first_step = None
            if pv_data_from_engine.get("score"):
                pv_score_for_first_step = (
                    pv_data_from_engine["score"].white().score(mate_score=MATE_SCORE)
                )

            engine_pv_moves = pv_data_from_engine.get("pv")
            if not engine_pv_moves:
                if pv_score_for_first_step is not None:
                    all_pvs_as_list_of_moves.append([])
                continue

            # Process each move in this PV
            for pv_move_idx, pv_chess_move in enumerate(engine_pv_moves):
                uci_for_pv_move = pv_chess_move.uci()
                board_before_move_for_pv = board_for_this_pv.copy()
                board_for_this_pv.push(pv_chess_move)
                fen_after_pv_move = board_for_this_pv.fen()

                pv_step_move_obj = Move(
                    id=self.get_game_id(),
                    position=fen_after_pv_move,
                    move=uci_for_pv_move,
                    context=f"pv_{pv_idx}_step_{pv_move_idx}",
                    isAnalyzed=False,
                    piece=self._get_piece_for_move(
                        board_before_move=board_before_move_for_pv,
                        san_move=uci_for_pv_move,
                    ),
                    depth=main_move_obj.depth + pv_move_idx + 1,
                )
                current_pv_as_moves_list.append(pv_step_move_obj)

            # Assign the PV score to the first move in this PV
            if pv_score_for_first_step is not None and current_pv_as_moves_list:
                current_pv_as_moves_list[0].score = pv_score_for_first_step

            all_pvs_as_list_of_moves.append(current_pv_as_moves_list)

        return main_move_obj, all_pvs_as_list_of_moves

    def _get_piece_for_move(
        self, board_before_move: chess.Board, san_move: str
    ) -> str | None:
        """Helper to determine the piece (e.g., wP, bN) that made a move.

        Tries SAN first, then falls back to UCI to be robust with input formats.
        """
        move_obj = None
        try:
            move_obj = board_before_move.parse_san(san_move)
        except Exception:
            try:
                move_obj = chess.Move.from_uci(san_move)
                if move_obj not in board_before_move.legal_moves:
                    move_obj = None
            except Exception:
                move_obj = None

        if move_obj is None:
            return None

        piece = board_before_move.piece_at(move_obj.from_square)
        if piece:
            color_char = "w" if piece.color == chess.WHITE else "b"
            return f"{color_char}{piece.symbol().upper()}"
        return None

    def _get_all_captured_pieces(
        self, board: chess.Board
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Counts the missing original pieces for both players.
        The first dictionary returned contains Black pieces captured by White.
        The second dictionary returned contains White pieces captured by Black.
        Keys are piece symbols (p, n, b, r, q), values are counts.
        """
        initial_piece_counts = {
            chess.PAWN: 8,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.ROOK: 2,
            chess.QUEEN: 1,
        }

        # Standard algebraic notation for pieces (lowercase)
        piece_to_symbol = {
            chess.PAWN: "p",
            chess.KNIGHT: "n",
            chess.BISHOP: "b",
            chess.ROOK: "r",
            chess.QUEEN: "q",
        }

        # Stores Black's pieces that White has captured
        black_pieces_captured_by_white: dict[str, int] = {}
        # Stores White's pieces that Black has captured
        white_pieces_captured_by_black: dict[str, int] = {}

        for piece_type, initial_count in initial_piece_counts.items():
            symbol = piece_to_symbol[piece_type]

            # Count White's current pieces of this type
            current_white_pieces_on_board = len(board.pieces(piece_type, chess.WHITE))
            # Difference is the number of White pieces of this type captured by Black
            num_white_captured = initial_count - current_white_pieces_on_board
            if num_white_captured > 0:
                white_pieces_captured_by_black[symbol] = num_white_captured

            # Count Black's current pieces of this type
            current_black_pieces_on_board = len(board.pieces(piece_type, chess.BLACK))
            # Difference is the number of Black pieces of this type captured by White
            num_black_captured = initial_count - current_black_pieces_on_board
            if num_black_captured > 0:
                black_pieces_captured_by_white[symbol] = num_black_captured

        # The function is expected to return (pieces_captured_by_white, pieces_captured_by_black)
        return black_pieces_captured_by_white, white_pieces_captured_by_black


def _board_before_mainline_move(game: Game, move_index: int) -> chess.Board:
    board = game.board()
    for i, gm in enumerate(game.mainline_moves()):
        if i >= move_index:
            break
        board.push(gm)
    return board


def _score_to_move_score(score_val: float | None) -> MoveScore:
    if score_val is None:
        return MoveScore(cp=None, mate=None)
    if abs(score_val) > MATE_SCORE - 1000:
        moves_to_mate = MATE_SCORE - abs(score_val)
        if score_val < 0:
            mate = -moves_to_mate
        else:
            mate = moves_to_mate
        return MoveScore(cp=None, mate=mate)
    return MoveScore(cp=round(score_val), mate=None)


@dataclass
class EnginePipelineState:
    """Mutable state shared between engine phase and LLM commentary phase."""

    game: Game
    retriever: AnalysisRetriever
    moves_list: list[Move]
    analyzed_rows: list[AnalyzedMoveData]
    move_events: list[MoveEvent]
    context: GameAnalysisContext
    metadata: GameMetadata
    # Set by GameAnnotationPipeline when the commentary sweep finished.
    llm_done: bool = False


# Audience levels for comment renderings (kept in sync with phases/composer.py)
COMMENT_LEVELS = ("expert", "intermediate", "beginner")
DEFAULT_COMMENT_LEVEL = "intermediate"


def _facts_to_json(facts: Any) -> dict[str, Any]:
    """Trimmed CommentFacts for the structured comment renderer in the UI."""

    def _line(dl: Any) -> dict[str, Any] | None:
        if dl is None or not getattr(dl, "line_san", None):
            return None
        return {
            "start_fen": dl.start_fen,
            "san": list(dl.line_san),
            "fens": list(dl.fens),
            "feature_series": dict(getattr(dl, "feature_series", {}) or {}),
        }

    def _claims(claims: Any) -> list[dict[str, Any]]:
        return [
            {
                "text": c.text,
                "text_state": c.text_state,
                "features": list(c.features_involved),
                "delta_cp": c.delta_cp,
                "flag_note": c.flag_note,
                "beneficiary": c.beneficiary,
                "is_concession": c.is_concession,
            }
            for c in (claims or [])
        ]

    out: dict[str, Any] = {
        "verdict": facts.verdict,
        "eval_cp": facts.eval_cp,
        "eval_mate": facts.eval_mate,
        "depth": facts.depth,
        "engine": facts.engine,
        "display_line": _line(facts.display_line),
        "claims": _claims(facts.claims),
        "concession_mode": facts.concession_mode,
    }
    alt = facts.better_alternative
    if alt is not None:
        out["better_alternative"] = {
            "san": alt.san,
            "uci": alt.uci,
            "verdict": alt.verdict,
            "eval_cp": alt.eval_cp,
            "display_line": _line(alt.display_line),
            "claims": _claims(alt.claims),
        }
    return out


def _build_feature_series(analyzed_rows: list[AnalyzedMoveData]) -> FeatureSeries:
    """Aligned per-ply arrays of the charted Guid features (White-POV cp)."""
    plies: list[int] = []
    by_name: dict[str, list[int | None]] = {name: [] for name in CHART_FEATURES}
    for row in analyzed_rows:
        plies.append(row.ply)
        guid = (row.hidden_features or {}).get("_guid") or {}
        for name in CHART_FEATURES:
            d = guid.get(name)
            if isinstance(d, dict) and d.get("v") is not None:
                by_name[name].append(int(d["v"]))
            else:
                by_name[name].append(None)
    return FeatureSeries(plies=plies, features=by_name)


@dataclass
class _MoveComment:
    """A move's resolved comment, its per-audience-level texts, and its motifs."""

    comment: str | None = None
    comments_by_level: dict[str, str] = field(default_factory=dict)


def _opening_comment(analyzed_move: Move) -> str | None:
    hidden = analyzed_move.hiddenFeatures or {}
    opening = hidden.get("_opening") if isinstance(hidden, dict) else None
    if isinstance(opening, dict) and opening.get("comment"):
        return str(opening["comment"])
    return None


def _apply_llm_comment(mc: _MoveComment, analyzed_move: Move) -> None:
    """Overlay the LLM-authored comment, per-level texts, and motifs onto ``mc``."""
    hidden = analyzed_move.hiddenFeatures or {}
    llm = hidden.get("_llm") if isinstance(hidden, dict) else None
    if not isinstance(llm, dict):
        return
    if llm.get("comment"):
        mc.comment = str(llm["comment"])
    levels = llm.get("comments")
    if isinstance(levels, dict):
        mc.comments_by_level = {str(k): str(v) for k, v in levels.items() if v}


def _facts_floor_comment(move_event: MoveEvent | None) -> str | None:
    """Deterministic Guid template (verdict + line + eval) for a move with facts."""
    if not move_event or move_event.comment_facts is None:
        return None
    try:
        from app.core.commentary.phases.composer import render_facts_template

        return render_facts_template(move_event.comment_facts)
    except Exception:
        return None


def _key_moment_comment(
    move_event: MoveEvent | None, key_moment: str | None
) -> str | None:
    if not (key_moment and move_event):
        return None
    label = key_moment.replace("_", " ").capitalize()
    swing = move_event.eval_swing_cp
    if swing is None:
        return label
    return f"{label} (Eval swing: {swing / 100:+.2f} pawns, White POV step)"


def _fallback_comment(
    move_event: MoveEvent | None, key_moment: str | None
) -> str | None:
    """Comment of last resort: brief stub, else facts floor, else key-moment line."""
    me = move_event
    if me and me.brief_commentary and me.commentary_stub_ref_ply is not None:
        return (
            f"Continues the same theme as around ply {me.commentary_stub_ref_ply} "
            f"(see that move's commentary)."
        )
    return _facts_floor_comment(me) or _key_moment_comment(me, key_moment)


def _resolve_move_comment(
    row: AnalyzedMoveData,
    analyzed_move: Move,
    move_event: MoveEvent | None,
    key_moment: str | None,
    *,
    side_ok: bool,
) -> _MoveComment:
    """Resolve the comment cascade: opening -> LLM -> stub/facts/key-moment floor.

    A non-selected commentary side stays fully silent (no fallback either).
    """
    mc = _MoveComment()
    if (row.phase_raw or "") == "early":
        mc.comment = _opening_comment(analyzed_move)
    if move_event and move_event.is_critical and side_ok:
        _apply_llm_comment(mc, analyzed_move)
    if side_ok and not mc.comment:
        mc.comment = _fallback_comment(move_event, key_moment)
    if mc.comment and not mc.comments_by_level:
        mc.comments_by_level = dict.fromkeys(COMMENT_LEVELS, mc.comment)
    if mc.comments_by_level:
        mc.comment = mc.comments_by_level.get(DEFAULT_COMMENT_LEVEL) or mc.comment
    return mc


def _trace_pv_line(
    board_start: chess.Board, pv_sequence: list
) -> tuple[list[str], list[str]]:
    """SAN and post-move FEN for each ply of a PV, walked from ``board_start``."""
    san_line: list[str] = []
    fen_line: list[str] = []
    board = board_start.copy()
    for pm in pv_sequence:
        try:
            mv = chess.Move.from_uci(pm.move)
            san_line.append(board.san(mv))
            board.push(mv)
            fen_line.append(board.fen())
        except Exception:
            san_line.append(str(pm.move))
            fen_line.append("")
    return san_line, fen_line


def _build_variations(game: Game, move_index: int, pvs: list) -> list[Variation]:
    """Engine PVs -> Variation list (SAN + per-ply FENs) from the pre-move board."""
    board_start = _board_before_mainline_move(game, move_index)
    variations: list[Variation] = []
    for rank, pv_sequence in enumerate(pvs):
        if not pv_sequence:
            continue
        san_line, fen_line = _trace_pv_line(board_start, pv_sequence)
        variations.append(
            Variation(
                rank=rank + 1,
                move_san=san_line[0] if san_line else "",
                score=_score_to_move_score(pv_sequence[0].score),
                line=san_line,
                fens=fen_line,
                depth=DEFAULT_ANALYSIS_DEPTH,
            )
        )
    return variations


def _llm_rendering_debug(analyzed_move: Move) -> tuple[Any, Any]:
    """(facts_renderings, facts_contract_ok) the LLM stored under hiddenFeatures._llm."""
    hidden = analyzed_move.hiddenFeatures or {}
    llm = hidden.get("_llm") if isinstance(hidden, dict) else None
    if not isinstance(llm, dict):
        return None, None
    return llm.get("facts_renderings"), llm.get("facts_contract_ok")


def _facts_debug(
    facts: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    """(envisioned-line stats, fired-rule claims, muted claims) from CommentFacts."""
    if facts is None:
        return None, [], []
    envisioned_stats = None
    dl = facts.display_line
    if dl is not None:
        envisioned_stats = {
            "kept_plies": len(dl.line_san),
            "trimmed_plies": dl.trimmed_plies,
            "start_quiescent": dl.start_quiescent,
            "leaf_quiescent": dl.leaf_quiescent,
        }
    fired_rules = [
        {
            "rule_id": c.rule_id,
            "text": c.text,
            "delta_cp": c.delta_cp,
            "features": list(c.features_involved),
            "flag_note": c.flag_note,
        }
        for c in facts.claims
    ]
    return envisioned_stats, fired_rules, list(facts.muted_claims)


def _build_move_debug(
    analyzed_move: Move,
    move_event: MoveEvent | None,
    facts: Any,
    comment: str | None,
    phase_raw: str,
) -> dict[str, Any] | None:
    """Academic reasoning trace for a commented mid/endgame move (None otherwise)."""
    me = move_event
    if not (comment and me and phase_raw != "early"):
        return None
    renderings, contract_ok = _llm_rendering_debug(analyzed_move)
    envisioned_stats, fired_rules, muted = _facts_debug(facts)
    return {
        "eval_before_cp": me.eval_before_cp,
        "eval_after_cp": me.eval_after_cp,
        "eval_swing_cp": me.eval_swing_cp,
        "best_move_san": me.best_move_san,
        "best_move_eval_cp": me.best_move_eval_cp,
        "key_moment_type": me.key_moment_type,
        "move_quality": me.move_quality.value if me.move_quality else None,
        "envisioned": envisioned_stats,
        "fired_rules": fired_rules,
        "muted_claims": muted,
        "renderings": renderings,
        "contract_ok": contract_ok,
    }


def _build_feature_refs(
    facts: Any, comment: str | None
) -> tuple[list[FeatureRef], dict[str, Any] | None]:
    """Chart-highlight feature refs (and the diff tables) grounding this comment."""
    if facts is None:
        return [], None
    delta_by_name: dict[str, int] = {}
    if facts.feature_diff:
        for fd in list(facts.feature_diff.positive) + list(facts.feature_diff.negative):
            delta_by_name[fd.name] = fd.delta_cp
    feature_refs = [
        FeatureRef(name=name, delta_cp=int(delta_by_name.get(name, 0)))
        for name in facts.feature_refs()
    ]
    feature_diff_out: dict[str, Any] | None = None
    if comment and facts.feature_diff:
        feature_diff_out = {
            "positive": [
                fd.model_dump()
                for fd in facts.feature_diff.positive[:FEATURE_DIFF_MAX_ENTRIES]
            ],
            "negative": [
                fd.model_dump()
                for fd in facts.feature_diff.negative[:FEATURE_DIFF_MAX_ENTRIES]
            ],
        }
    return feature_refs, feature_diff_out


def _resolve_comment_tokens(
    comment: str | None,
    comments_by_level: dict[str, str],
    move_event: MoveEvent | None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Resolve interactive tokens for the comment and each per-audience-level text."""
    if not (comment and move_event):
        return [], {}
    try:
        from app.core.commentary.annotation_tokens import resolve_tokens_for_comment

        tokens = resolve_tokens_for_comment(
            comment, move_event.fen_before, move_event.fen_after
        )
        by_text: dict[str, list[dict[str, Any]]] = {comment: tokens}
        by_level: dict[str, list[dict[str, Any]]] = {}
        for level, text in comments_by_level.items():
            if text not in by_text:
                by_text[text] = resolve_tokens_for_comment(
                    text, move_event.fen_before, move_event.fen_after
                )
            by_level[level] = by_text[text]
        return tokens, by_level
    except Exception:
        return [], {}


def assemble_game_json(state: EnginePipelineState) -> GameJson:
    """Build GameJson from pipeline state (engine + optional LLM fields)."""
    game = state.game
    moves_list = state.moves_list
    analyzed_rows = state.analyzed_rows
    move_events = state.move_events

    game_moves: list[GameMove] = []
    for idx, row in enumerate(analyzed_rows):
        move_obj = moves_list[idx]
        analyzed_move = row.analyzed_move
        pvs = row.pvs if isinstance(row.pvs, list) else []

        me = move_events[idx] if idx < len(move_events) else None
        key_moment = (me.key_moment_type if me else None) or getattr(
            analyzed_rows[idx], "key_moment_type", None
        )

        # Commentary side gate: when a side is selected, the other side's moves
        # get NO commentary apparatus at all (no comment, facts, dot, charts).
        side_sel = (state.metadata.comment_side or "both").lower()
        mover_is_white = move_obj.depth % 2 == 1
        _side_ok = side_sel == "both" or (side_sel == "white") == mover_is_white

        mc = _resolve_move_comment(row, analyzed_move, me, key_moment, side_ok=_side_ok)
        comment = mc.comment
        comments_by_level = mc.comments_by_level

        variations = _build_variations(game, idx, pvs)

        board_before = _board_before_mainline_move(game, idx)
        try:
            chess_move_obj = chess.Move.from_uci(move_obj.move)
            san_main = board_before.san(chess_move_obj)
        except Exception:
            san_main = move_obj.move

        # Guid Expert Module outputs: which features ground this comment
        # (chart highlights).
        facts = me.comment_facts if (me and _side_ok) else None
        feature_refs, _ = _build_feature_refs(facts, comment)
        resolved_tokens, resolved_by_level = _resolve_comment_tokens(
            comment, comments_by_level, me
        )
        comment_facts_out = (
            _facts_to_json(facts) if (comment and facts is not None) else None
        )

        # Academic reasoning trace ("how did we reach this conclusion").
        debug_out = _build_move_debug(
            analyzed_move, me, facts, comment, (row.phase_raw or "")
        )

        game_moves.append(
            GameMove(
                mn=board_before.fullmove_number,
                color="w" if board_before.turn == chess.WHITE else "b",
                san=san_main,
                uci=move_obj.move,
                fen=move_obj.position,
                phase=str(row.phase_raw or analyzed_move.phase or "mid"),
                score=_score_to_move_score(analyzed_move.score),
                variations=variations,
                comment=comment,
                classification=key_moment,
                move_quality=me.move_quality.value if me else None,
                is_key_moment=bool(me and me.key_moment_type and _side_ok),
                feature_refs=feature_refs,
                resolved_tokens=resolved_tokens,
                comments=comments_by_level,
                resolved_tokens_by_level=resolved_by_level,
                comment_facts=comment_facts_out,
                debug=debug_out,
            )
        )

    return GameJson(
        metadata=state.metadata,
        moves=game_moves,
        feature_series=_build_feature_series(analyzed_rows),
        commentary_complete=state.llm_done,
        analysis_info=AnalysisInfo(
            engine="Stockfish",
            depth=16,
            multipv=DEFAULT_PV_COUNT,
            timestamp=time.time(),
        ),
    )


async def run_engine_analysis_to_json(
    pgn_string: str,
    engine_connector: EngineConnector,
    progress_callback: Callable[[float, str], Awaitable[None]],
    *,
    metadata_id: str | None = None,
) -> tuple[GameJson, EnginePipelineState]:
    """Passes 1–3: engine analysis + event extraction. Heuristic comments only."""
    PGNReader.validate_single_game(pgn_string)
    pgn_reader = PGNReader()
    game = pgn_reader.read_game_from_string(pgn_string)
    if not game:
        raise ValueError("Invalid PGN")

    eco_book = ECOBook()
    retriever = AnalysisRetriever(engine_connector, game, eco_book)
    headers = retriever.get_pgn_headers()
    detected_opening, _det_ply = detect_opening(game, eco_book)
    header_opening = (
        "" if is_absent_opening_header(headers.opening) else headers.opening.strip()
    )
    merged_opening = merge_opening_with_headers(
        detected_opening,
        header_opening,
        headers.eco,
    )
    meta_opening_name = (
        merged_opening.name if merged_opening else (header_opening or None)
    )
    meta_opening_eco = merged_opening.code if merged_opening else headers.eco

    metadata = GameMetadata(
        id=metadata_id or str(uuid.uuid4()),
        white=headers.whiteName,
        black=headers.blackName,
        result=headers.result,
        date=game.headers.get("Date"),
        eventId=headers.event,
        whiteElo=headers.whiteElo,
        blackElo=headers.blackElo,
        opening=meta_opening_name,
        opening_eco=meta_opening_eco,
    )

    moves_list = retriever.get_move_list()
    total_moves = len(moves_list)
    analyzed_rows: list[AnalyzedMoveData] = []
    previous_move_obj: Move | None = None
    phase_classifier = PhaseClassifier(eco_book)
    seed_depth = int(os.environ.get("BOOK_EXIT_SEED_DEPTH", "16"))

    for idx, move_obj in enumerate(moves_list):
        progress = (idx / max(total_moves, 1)) * 90.0

        board_before = _board_before_mainline_move(game, idx)
        fen_before = board_before.fen()
        try:
            chess_move_obj = chess.Move.from_uci(move_obj.move)
            san_main = board_before.san(chess_move_obj)
        except Exception:
            san_main = move_obj.move

        board_after = chess.Board(move_obj.position)
        uci_prefix = retriever._uci_prefix_for_depth(move_obj.depth)
        phase = phase_classifier.classify(board_after, uci_prefix)
        move_obj.phase = phase

        if phase == "early":
            # In-book ply: no engine work at all.
            await progress_callback(progress, f"Book: move {idx + 1}/{total_moves}")
            analyzed_move, pvs = retriever.analyze_book_move(move_obj)
        else:
            await progress_callback(progress, f"Engine: move {idx + 1}/{total_moves}")
            if previous_move_obj is not None and previous_move_obj.score is None:
                # First move out of book: evaluate the book-exit position once so
                # eval-swing detection has a baseline for this novelty.
                seed = retriever.evaluate_position(fen_before, depth=seed_depth)
                if seed is not None:
                    previous_move_obj.score = seed
                    if analyzed_rows:
                        analyzed_rows[-1] = analyzed_rows[-1].model_copy(
                            update={"score_cp": seed}
                        )
            analyzed_move, pvs = retriever.analyze_move(move_obj, stage=16)

        # Guid feature vector (engine-free): every mainline ply, all phases —
        # feeds the per-feature progression charts and the rule engine.
        try:
            guid_vec = compute_feature_vector(board_after)
            if isinstance(analyzed_move.hiddenFeatures, dict):
                analyzed_move.hiddenFeatures["_guid"] = vector_to_plain(guid_vec)
        except Exception as e:
            logger.error("Guid feature vector failed at ply %s: %s", move_obj.depth, e)

        try:
            if analyzed_move.hiddenFeatures is None:
                analyzed_move.hiddenFeatures = {}
            if isinstance(analyzed_move.hiddenFeatures, dict):
                analyzed_move.hiddenFeatures.setdefault("_ai", {})
                ai_meta = analyzed_move.hiddenFeatures["_ai"]
                ai_meta["prevScore"] = (
                    previous_move_obj.score if previous_move_obj else None
                )
                ai_meta["scoreNow"] = analyzed_move.score
                if (
                    ai_meta.get("prevScore") is not None
                    and ai_meta.get("scoreNow") is not None
                ):
                    ai_meta["scoreDelta"] = ai_meta["scoreNow"] - ai_meta["prevScore"]
        except Exception:
            pass

        _eng = {}
        if isinstance(analyzed_move.hiddenFeatures, dict):
            _eng = (
                (analyzed_move.hiddenFeatures.get("_engine") or {})
                if analyzed_move.hiddenFeatures
                else {}
            )
        analyzed_rows.append(
            AnalyzedMoveData(
                index=idx,
                ply=move_obj.depth,
                san=san_main,
                uci=move_obj.move,
                fen_before=fen_before,
                fen_after=move_obj.position,
                score_cp=analyzed_move.score,
                phase_raw=str(analyzed_move.phase or "mid"),
                pvs=list(pvs) if pvs else [],
                hidden_features=analyzed_move.hiddenFeatures or {},
                captured_by_white=analyzed_move.capturedByWhite or {},
                captured_by_black=analyzed_move.capturedByBlack or {},
                analyzed_move=analyzed_move,
                eval_at_depth=dict(_eng.get("eval_at_depth") or {}),
                pv1_change_count=int(_eng.get("pv1_change_count", 0)),
            )
        )
        previous_move_obj = analyzed_move

    # Deterministic opening comments for in-book plies (no engine, no LLM)
    from app.core.commentary.phases.early import attach_opening_comments

    attach_opening_comments(analyzed_rows, eco_book)

    await progress_callback(92.0, "Extracting move events...")
    extractor = ChessEventExtractor(
        eco_book=eco_book,
        key_moment_detector=retriever.key_moment_detector,
    )
    move_events = extractor.extract_events(game, analyzed_rows)

    # Guid Expert Module: fire the rule engine for every out-of-book move and
    # attach the resulting CommentFacts (the comment's inviolable content).
    from app.core.commentary.rules import build_comment_facts

    CLAIM_DEDUP_WINDOW_PLIES = int(os.environ.get("CLAIM_DEDUP_WINDOW_PLIES", "6"))
    # Structural facts describe a state (a lost bishop pair, a weak back rank),
    # not an event — they should be stated once, not re-announced whenever best
    # play recreates them. Dedup them over a much wider window, keyed by rule.
    STRUCTURAL_RULE_IDS = {
        "bishop_pair_eliminated",
        "back_rank_weakness",
        "bad_bishop_created",
        "bad_bishop_solved",
        "strong_knight_established",
        "strong_knight_lost",
        "rook_reaches_seventh",
        "rooks_connected",
        "passed_pawn_created",
        "outside_passer",
    }
    STRUCTURAL_WINDOW_PLIES = int(os.environ.get("STRUCTURAL_DEDUP_WINDOW_PLIES", "24"))
    last_claim_ply: dict[tuple[str | None, str | None], int] = {}
    for mi, me in enumerate(move_events):
        row = (
            analyzed_rows[me.move_index]
            if 0 <= me.move_index < len(analyzed_rows)
            else None
        )
        if row is None:
            continue
        try:
            facts = build_comment_facts(row, me, depth=16)
        except Exception as e:
            logger.warning("comment facts failed at ply %s: %s", me.ply, e)
            facts = None
        if facts is not None:
            # A persistent feature change (e.g. an unsolved bad bishop) fires on
            # every envisioned line; keep the first occurrence, mute repeats.
            kept = []
            muted: list[str] = []
            for c in facts.claims:
                key = (c.rule_id, c.beneficiary)
                window = (
                    STRUCTURAL_WINDOW_PLIES
                    if c.rule_id in STRUCTURAL_RULE_IDS
                    else CLAIM_DEDUP_WINDOW_PLIES
                )
                prev = last_claim_ply.get(key)
                if prev is not None and (me.ply - prev) <= window:
                    muted.append(c.text)
                    continue
                last_claim_ply[key] = me.ply
                kept.append(c)
            facts = facts.model_copy(update={"claims": kept, "muted_claims": muted})
            move_events[mi] = me.model_copy(update={"comment_facts": facts})

    opening_name = meta_opening_name
    opening_eco_ctx = meta_opening_eco
    for me in move_events:
        if me.opening_name:
            opening_name = me.opening_name
        if me.opening_eco:
            opening_eco_ctx = me.opening_eco

    context = GameAnalysisContext(
        metadata={
            "white": headers.whiteName,
            "black": headers.blackName,
            "result": headers.result,
            "whiteElo": headers.whiteElo,
            "blackElo": headers.blackElo,
        },
        move_events=move_events,
        critical_moments=[e for e in move_events if e.is_critical],
        opening_name=opening_name,
        opening_eco=opening_eco_ctx,
    )

    await progress_callback(95.0, "Engine analysis complete.")
    state = EnginePipelineState(
        game=game,
        retriever=retriever,
        moves_list=moves_list,
        analyzed_rows=analyzed_rows,
        move_events=move_events,
        context=context,
        metadata=metadata,
    )
    return assemble_game_json(state), state


def _pv_line_for_ai_payload(row: AnalyzedMoveData | None) -> list[dict[str, str]]:
    """SAN + FEN after each ply of PV1 for interactive commentary hover boards."""
    if not row or not row.pvs or not row.pvs[0]:
        return []
    out: list[dict[str, str]] = []
    board = chess.Board(row.fen_before)
    for pm in row.pvs[0]:
        uci = getattr(pm, "move", None)
        if not uci:
            break
        try:
            chm = chess.Move.from_uci(str(uci))
            san = board.san(chm)
            board.push(chm)
            out.append({"san": san, "fen": board.fen()})
        except Exception:
            break
    return out
