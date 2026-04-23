from __future__ import annotations

import chess
import logging
import os
import time
import uuid
from dataclasses import dataclass
from chess.pgn import Game
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.core.engine.engine_connector import EngineConnector
from app.models.Move import Move
from app.models.PgnMetadata import PgnMetadata
from app.models.Move import AnalysisStage
from app.core.commentary.features.positional_features import compute_hidden_features
from app.models.GameJson import (
    GameJson,
    GameMetadata,
    GameMove,
    AnalysisInfo,
    Variation,
    MoveScore,
    EpisodeSummary,
)
from app.core.io.pgn_reader import PGNReader
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.openings.eco_book import (
    ECOBook,
    detect_opening,
    is_absent_opening_header,
    merge_opening_with_headers,
    parse_pgn_eco_tag,
)
from app.core.commentary.tantivy_positional_retriever import get_default_retriever
from app.core.commentary.event_extractor import ChessEventExtractor
from app.core.commentary.episode_segmenter import EpisodeSegmenter
from app.core.commentary.features.future_line_compare import (
    compare_played_vs_best_future_lines,
)
from app.core.commentary.features.move_category import classify_move_event
from app.models.chess_events import (
    AnalyzedMoveData,
    Episode,
    GameAnalysisContext,
    MoveEvent,
    MoveQuality,
)

# ANALYSIS_STAGES = [0.05, 0.1, 0.2, 0.4]
ANALYSIS_STAGES = [4, 8, 16]
DEFAULT_PV_COUNT = 3
MATE_SCORE = 1000000
logger = logging.getLogger(__name__)


def _heuristic_llm_fallback_comment(me: MoveEvent) -> str:
    """Same shape as assemble_game_json heuristic when LLM returns empty."""
    key_moment = me.key_moment_type or ""
    if key_moment:
        swing = me.eval_swing_cp
        if swing is not None:
            return (
                f"{key_moment.replace('_', ' ').capitalize()} "
                f"(Eval swing: {swing / 100:+.2f} pawns, White POV step)"
            )
        return key_moment.replace("_", " ").capitalize()
    if me.teaching_moment:
        tact = ", ".join(m.value for m in me.tactical_motifs[:3])
        strat = ", ".join(m.value for m in me.strategic_motifs[:3])
        parts = [p for p in (tact, strat) if p]
        return "Teaching highlight" + (f": {', '.join(parts)}" if parts else "")
    return ""


def _game_winner_from_result(result: Optional[str]) -> Optional[str]:
    r = (result or "").strip()
    if r == "1-0":
        return "white"
    if r == "0-1":
        return "black"
    return None


def _apply_back_to_back_key_moment_suppression(
    move_events: List[MoveEvent],
    episodes: List[Episode],
    context: GameAnalysisContext,
) -> None:
    """Second of two same-type key moments within 2 plies loses LLM pass; gets stub reference."""
    last_ply: Optional[int] = None
    last_type: Optional[str] = None
    for mi, me in enumerate(move_events):
        km = me.key_moment_type
        if not km:
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
            for ep in episodes:
                for ej, ev in enumerate(ep.move_events):
                    if ev.ply == updated.ply:
                        ep.move_events[ej] = updated
                        break
        else:
            last_ply, last_type = me.ply, km


def _mark_teaching_moments_per_episode(
    move_events: List[MoveEvent],
    episodes: List[Episode],
    context: GameAnalysisContext,
    result: Optional[str],
) -> None:
    """One teaching highlight per episode: strong quiet move by eventual winner with strategy."""
    winner = _game_winner_from_result(result)
    if winner is None:
        return
    taken_plys: set[int] = set()
    ply_to_mi = {move_events[i].ply: i for i in range(len(move_events))}
    for ep in episodes:
        candidates: List[Tuple[int, int, int]] = []
        for ev in ep.move_events:
            if ev.key_moment_type or ev.teaching_moment or ev.brief_commentary:
                continue
            if ev.move_quality not in (MoveQuality.BEST, MoveQuality.EXCELLENT):
                continue
            if not ev.strategic_motifs:
                continue
            is_white = ev.ply % 2 == 1
            if winner == "white" and not is_white:
                continue
            if winner == "black" and is_white:
                continue
            mi = ply_to_mi.get(ev.ply)
            if mi is None:
                continue
            richness = len(ev.strategic_motifs) * 3 + len(ev.tactical_motifs)
            candidates.append((richness, ev.ply, mi))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        _score, ply, mi = candidates[0]
        if ply in taken_plys:
            continue
        taken_plys.add(ply)
        promoted = move_events[mi].model_copy(
            update={"teaching_moment": True, "is_critical": True}
        )
        move_events[mi] = promoted
        context.move_events[mi] = promoted
        for ep2 in episodes:
            for ej, ev2 in enumerate(ep2.move_events):
                if ev2.ply == promoted.ply:
                    ep2.move_events[ej] = promoted
                    break


def _bm25_pv_san_from_fen_after(engine: EngineConnector, fen_after: str, depth: int) -> List[str]:
    """SAN plies of engine PV1 from the position after the move (matches BM25 corpus indexing)."""
    try:
        board = chess.Board(fen_after)
        info = engine.analyse(board, depth=depth, multiPv=1)
        if isinstance(info, list) and info:
            info = info[0]
        if not isinstance(info, dict):
            return []
        pv = info.get("pv") or []
        b2 = board.copy()
        out: List[str] = []
        for m in pv[:8]:
            if m not in b2.legal_moves:
                break
            out.append(b2.san(m))
            b2.push(m)
        return out[:5]
    except Exception as e:
        logger.debug("BM25 PV from fen_after failed: %s", e)
        return []


class AnalysisRetriever:
    def __init__(self, engine_connector: EngineConnector, game: Game):
        self.analysis_stages = ANALYSIS_STAGES
        self.engine_connector = engine_connector
        self.game = game
        self.id_counter = 0
        self.analyzed_game: List[Move] = self.get_move_list()
        self.key_moment_detector = KeyMomentDetector()

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

    def get_move_list(self) -> List[Move]:
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

    def get_analysis_stages(self) -> List[float]:  # Corrected type hint
        """
        Returns the analysis stages for the engine.
        """
        return self.analysis_stages

    def analyze_move(
        self, main_move_obj: Move, stage: float
    ) -> Tuple[Move, List[List[Move]]]:
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
            after_primary
            .get("score")
            .white()
            .score(mate_score=MATE_SCORE)
        )
        # Capture trace for the after-move position (before any PV queries overwrite it)
        try:
            main_move_obj.trace = self.engine_connector.trace()
        except Exception:
            main_move_obj.trace = None
        main_move_obj.phase = self._determine_game_phase(board_after_move)
        # Multi-depth instability on the after-move position (search swings / PV changes)
        def _cp_and_pv1(info_any: Any) -> Tuple[Optional[int], Optional[str]]:
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

        eval_at_depth: Dict[int, int] = {}
        pv1_ucis: List[Optional[str]] = []
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
            after_features = compute_hidden_features(board_after_move)
        except Exception as e:
            logger.error(
                f"Hidden features (after) failed at depth {main_move_obj.depth} FEN={board_after_move.fen()}: {e}"
            )
            after_features = {"error": str(e)}
        if isinstance(after_features, dict):
            after_features.setdefault("_engine", {})
            after_features["_engine"]["eval_at_depth"] = eval_at_depth
            after_features["_engine"]["pv1_change_count"] = pv1_change_count
        (
            main_move_obj.capturedByWhite,
            main_move_obj.capturedByBlack,
        ) = self._get_all_captured_pieces(board_after_move)
        main_move_obj.isAnalyzed = True
        # Mark stage as final if this is the last configured stage
        main_move_obj.analysisStage = (
            AnalysisStage.FINAL if stage == max(self.analysis_stages) else AnalysisStage.DEEP
        )
        main_move_obj.analysisVersion = 1

        all_pvs_as_list_of_moves: List[List[Move]] = []

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
        try:
            before_features = compute_hidden_features(board_before_move)
        except Exception as e:
            logger.error(
                f"Hidden features (before) failed at depth {main_move_obj.depth} FEN={board_before_move.fen()}: {e}"
            )
            before_features = {"error": str(e)}

        def _compute_features_delta(before: dict, after: dict) -> dict:
            # Focus on numeric and boolean keys that matter strategically
            delta: dict = {"white": {}, "black": {}, "openFiles": {}}
            try:
                # Open files count deltas
                if isinstance(before.get("openFiles"), dict) and isinstance(after.get("openFiles"), dict):
                    try:
                        delta["openFiles"]["openCount"] = len(after["openFiles"].get("open", [])) - len(before["openFiles"].get("open", []))
                        delta["openFiles"]["semiOpenWhiteCount"] = len(after["openFiles"].get("semiOpenWhite", [])) - len(before["openFiles"].get("semiOpenWhite", []))
                        delta["openFiles"]["semiOpenBlackCount"] = len(after["openFiles"].get("semiOpenBlack", [])) - len(before["openFiles"].get("semiOpenBlack", []))
                    except Exception:
                        pass
                for side in ("white", "black"):
                    b = before.get(side, {}) if isinstance(before.get(side, {}), dict) else {}
                    a = after.get(side, {}) if isinstance(after.get(side, {}), dict) else {}
                    def diff_num(key: str):
                        if isinstance(b.get(key), int) and isinstance(a.get(key), int):
                            delta[side][key] = a[key] - b[key]
                    for k in ("doubledPawns", "isolatedPawns", "passedPawns", "attackedPieces", "attackingPieces", "rooksOnOpenFiles", "rooksOnSemiOpenFiles"):
                        diff_num(k)
                    # Booleans as changed flags
                    for k in ("hasBishopPair", "canCastleKingSide", "canCastleQueenSide", "connectedRooks"):
                        if isinstance(b.get(k), bool) and isinstance(a.get(k), bool):
                            if a[k] != b[k]:
                                delta[side][f"{k}Changed"] = True
                return delta
            except Exception:
                return {}

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
            current_pv_as_moves_list: List[Move] = []
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

                # Calculate trace for this PV move (as requested)
                trace_for_pv_move_pos: Optional[Dict] = None
                try:
                    self.engine_connector.analyse(
                        board_for_this_pv, time_limit=0.01, multiPv=1
                    )
                    trace_for_pv_move_pos = self.engine_connector.trace()
                except Exception as e_trace:
                    logger.error(
                        f"Error getting trace for PV move {uci_for_pv_move} "
                        f"at FEN {fen_after_pv_move}: {e_trace}"
                    )

                pv_step_move_obj = Move(
                    id=self.get_game_id(),
                    position=fen_after_pv_move,
                    move=uci_for_pv_move,
                    context=f"pv_{pv_idx}_step_{pv_move_idx}",
                    isAnalyzed=False,
                    trace=trace_for_pv_move_pos,
                    piece=self._get_piece_for_move(
                        board_before_move=board_before_move_for_pv, san_move=uci_for_pv_move
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
    ) -> tuple[Dict[str, int], Dict[str, int]]:
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
        black_pieces_captured_by_white: Dict[str, int] = {}
        # Stores White's pieces that Black has captured
        white_pieces_captured_by_black: Dict[str, int] = {}

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

    def _determine_game_phase(self, board: chess.Board) -> str:
        """
        Determine the game phase based on piece count and move number.
        Returns "early", "mid", or "end".

        Source: https://lichess.org/forum/general-chess-discussion/opening--middle--end-what-defines-the-phase
        """
        # --- Piece Counts ---
        white_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
        white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))

        black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
        black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))

        # Sum of all minor (Knights, Bishops) and major (Rooks, Queens) pieces on the board
        current_minor_major_pieces_count = (
            white_knights
            + white_bishops
            + white_rooks
            + white_queens
            + black_knights
            + black_bishops
            + black_rooks
            + black_queens
        )

        # --- Phase Determination ---

        # 1. Early Game (Opening)
        # The game starts in the "early" phase.
        # Transition out of early game after a certain number of moves, e.g., 10 full moves.
        # This also implies that pieces are somewhat developed.
        if board.fullmove_number <= 10:  # Threshold for early game
            return "early"

        # 2. End Game
        # "when there are less than 7 minor and major pieces on the board the end-game has begun"
        if current_minor_major_pieces_count < 7:
            return "end"

        # 3. Mid Game
        # If the game is not in the early phase and not yet in the end game, it's considered mid-game.
        # This covers scenarios where the position is complex, pieces are developed,
        # and potentially 2 or more sets of minor/major pieces have been exchanged.
        # (Initial minor/major pieces = 14. If >=4 are off, count <= 10.
        # If count is between 7 and 10 (inclusive) and not early, it's mid).
        return "mid"


def _board_before_mainline_move(game: Game, move_index: int) -> chess.Board:
    board = game.board()
    for i, gm in enumerate(game.mainline_moves()):
        if i >= move_index:
            break
        board.push(gm)
    return board


def _score_to_move_score(score_val: Optional[float]) -> MoveScore:
    if score_val is None:
        return MoveScore(cp=None, mate=None)
    if abs(score_val) > MATE_SCORE - 1000:
        moves_to_mate = MATE_SCORE - abs(score_val)
        if score_val < 0:
            mate = -moves_to_mate
        else:
            mate = moves_to_mate
        return MoveScore(cp=None, mate=mate)
    return MoveScore(cp=int(round(score_val)), mate=None)


@dataclass
class EnginePipelineState:
    """Mutable state shared between engine phase and LLM commentary phase."""

    game: Game
    retriever: AnalysisRetriever
    moves_list: List[Move]
    analyzed_rows: List[AnalyzedMoveData]
    move_events: List[MoveEvent]
    episodes: List[Episode]
    context: GameAnalysisContext
    metadata: GameMetadata
    ply_to_episode: Dict[int, int]


def assemble_game_json(state: EnginePipelineState) -> GameJson:
    """Build GameJson from pipeline state (engine + optional LLM fields)."""
    game = state.game
    retriever = state.retriever
    moves_list = state.moves_list
    analyzed_rows = state.analyzed_rows
    move_events = state.move_events
    episodes = state.episodes
    context = state.context
    ply_to_episode = state.ply_to_episode

    episode_summaries: List[EpisodeSummary] = []
    for ep in episodes:
        start_mn = (ep.start_ply + 1) // 2
        end_mn = (ep.end_ply + 1) // 2
        episode_summaries.append(
            EpisodeSummary(
                episode_index=ep.episode_index,
                title=ep.title,
                start_move=start_mn,
                end_move=end_mn,
                narrative=ep.narrative_summary,
                dominant_theme=ep.dominant_theme,
            )
        )

    game_moves: List[GameMove] = []
    for idx, row in enumerate(analyzed_rows):
        move_obj = moves_list[idx]
        analyzed_move = row.analyzed_move
        pvs = row.pvs if isinstance(row.pvs, list) else []

        me = move_events[idx] if idx < len(move_events) else None
        key_moment = (me.key_moment_type if me else None) or getattr(
            analyzed_rows[idx], "key_moment_type", None
        )

        comment: Optional[str] = None
        named_motifs: List[str] = []
        primary_motif_label: Optional[str] = None
        if me and me.is_critical:
            try:
                hf = analyzed_move.hiddenFeatures or {}
                llm = hf.get("_llm") if isinstance(hf, dict) else None
                if isinstance(llm, dict) and llm.get("comment"):
                    comment = str(llm["comment"])
                if isinstance(llm, dict):
                    nm = llm.get("named_motifs")
                    if isinstance(nm, list):
                        named_motifs = [str(x) for x in nm if x]
                    pm = llm.get("primary_motif_label")
                    if pm:
                        primary_motif_label = str(pm)
            except Exception:
                comment = None
        if (
            not comment
            and me
            and me.brief_commentary
            and me.commentary_stub_ref_ply is not None
        ):
            comment = (
                f"Continues the same theme as around ply {me.commentary_stub_ref_ply} "
                f"(see that move's commentary)."
            )
        if not comment and key_moment and me:
            swing = me.eval_swing_cp
            if swing is not None:
                comment = (
                    f"{key_moment.replace('_', ' ').capitalize()} "
                    f"(Eval swing: {swing / 100:+.2f} pawns, White POV step)"
                )
            else:
                comment = key_moment.replace("_", " ").capitalize()

        variations: List[Variation] = []
        board_pv_start = _board_before_mainline_move(game, idx)
        for rank, pv_sequence in enumerate(pvs):
            if not pv_sequence:
                continue
            first_move = pv_sequence[0]
            san_line: List[str] = []
            board_trace = board_pv_start.copy()
            for pm in pv_sequence:
                try:
                    m_uci = chess.Move.from_uci(pm.move)
                    san_line.append(board_trace.san(m_uci))
                    board_trace.push(m_uci)
                except Exception:
                    san_line.append(str(pm.move))

            score_val = first_move.score
            variations.append(
                Variation(
                    rank=rank + 1,
                    move_san=san_line[0] if san_line else "",
                    score=_score_to_move_score(score_val),
                    line=san_line,
                )
            )

        board_before = _board_before_mainline_move(game, idx)
        try:
            chess_move_obj = chess.Move.from_uci(move_obj.move)
            san_main = board_before.san(chess_move_obj)
        except Exception:
            san_main = move_obj.move

        ep_idx = ply_to_episode.get(move_obj.depth)

        game_moves.append(
            GameMove(
                mn=board_before.fullmove_number,
                color="w" if board_before.turn == chess.WHITE else "b",
                san=san_main,
                uci=move_obj.move,
                fen=move_obj.position,
                score=_score_to_move_score(analyzed_move.score),
                variations=variations,
                comment=comment,
                classification=key_moment,
                move_quality=me.move_quality.value if me else None,
                event_type=me.event_type.value if me else None,
                tactical_motifs=[m.value for m in (me.tactical_motifs if me else [])],
                strategic_motifs=[m.value for m in (me.strategic_motifs if me else [])],
                move_category=me.move_category.value if me and me.move_category else None,
                plan_comparison=me.plan_comparison.model_dump() if me and me.plan_comparison else None,
                is_critical=bool(me.is_critical if me else False),
                episode_index=ep_idx,
                named_motifs=named_motifs,
                primary_motif_label=primary_motif_label,
            )
        )

    return GameJson(
        metadata=state.metadata,
        moves=game_moves,
        episodes=episode_summaries,
        game_narrative=context.game_narrative,
        game_summary=context.game_digest or None,
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
    metadata_id: Optional[str] = None,
) -> Tuple[GameJson, EnginePipelineState]:
    """Passes 1–3: engine analysis, event extraction, episode segmentation. Heuristic comments only."""
    PGNReader.validate_single_game(pgn_string)
    pgn_reader = PGNReader()
    game = pgn_reader.read_game_from_string(pgn_string)
    if not game:
        raise ValueError("Invalid PGN")

    retriever = AnalysisRetriever(engine_connector, game)
    headers = retriever.get_pgn_headers()
    eco_book = ECOBook()
    detected_opening, _det_ply = detect_opening(game, eco_book)
    header_opening = "" if is_absent_opening_header(headers.opening) else headers.opening.strip()
    merged_opening = merge_opening_with_headers(
        detected_opening,
        header_opening,
        headers.eco,
    )
    meta_opening_name = merged_opening.name if merged_opening else (header_opening or None)
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
    analyzed_rows: List[AnalyzedMoveData] = []
    previous_move_obj: Optional[Move] = None

    for idx, move_obj in enumerate(moves_list):
        progress = (idx / max(total_moves, 1)) * 90.0
        await progress_callback(progress, f"Engine: move {idx + 1}/{total_moves}")

        board_before = _board_before_mainline_move(game, idx)
        fen_before = board_before.fen()
        try:
            chess_move_obj = chess.Move.from_uci(move_obj.move)
            san_main = board_before.san(chess_move_obj)
        except Exception:
            san_main = move_obj.move

        analyzed_move, pvs = retriever.analyze_move(move_obj, stage=16)

        try:
            if analyzed_move.hiddenFeatures is None:
                analyzed_move.hiddenFeatures = {}
            if isinstance(analyzed_move.hiddenFeatures, dict):
                analyzed_move.hiddenFeatures.setdefault("_ai", {})
                ai_meta = analyzed_move.hiddenFeatures["_ai"]
                ai_meta["prevScore"] = previous_move_obj.score if previous_move_obj else None
                ai_meta["scoreNow"] = analyzed_move.score
                if ai_meta.get("prevScore") is not None and ai_meta.get("scoreNow") is not None:
                    ai_meta["scoreDelta"] = ai_meta["scoreNow"] - ai_meta["prevScore"]
        except Exception:
            pass

        _eng = {}
        if isinstance(analyzed_move.hiddenFeatures, dict):
            _eng = (analyzed_move.hiddenFeatures.get("_engine") or {}) if analyzed_move.hiddenFeatures else {}
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
                trace=analyzed_move.trace,
                captured_by_white=analyzed_move.capturedByWhite or {},
                captured_by_black=analyzed_move.capturedByBlack or {},
                analyzed_move=analyzed_move,
                eval_at_depth=dict(_eng.get("eval_at_depth") or {}),
                pv1_change_count=int(_eng.get("pv1_change_count", 0)),
            )
        )
        previous_move_obj = analyzed_move

    await progress_callback(92.0, "Extracting events and episodes...")
    extractor = ChessEventExtractor(
        eco_book=eco_book,
        key_moment_detector=retriever.key_moment_detector,
    )
    move_events = extractor.extract_events(game, analyzed_rows)
    segmenter = EpisodeSegmenter()
    episodes = segmenter.segment(move_events)

    ply_to_episode: Dict[int, int] = {}
    for ep in episodes:
        for me in ep.move_events:
            ply_to_episode[me.ply] = ep.episode_index

    opening_name = meta_opening_name
    opening_eco_ctx = meta_opening_eco
    for me in move_events:
        if me.opening_name:
            opening_name = me.opening_name
        if me.opening_eco:
            opening_eco_ctx = me.opening_eco
        if opening_name and opening_eco_ctx:
            break

    context = GameAnalysisContext(
        metadata={
            "white": headers.whiteName,
            "black": headers.blackName,
            "result": headers.result,
            "whiteElo": headers.whiteElo,
            "blackElo": headers.blackElo,
        },
        move_events=move_events,
        episodes=episodes,
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
        episodes=episodes,
        context=context,
        metadata=metadata,
        ply_to_episode=ply_to_episode,
    )
    return assemble_game_json(state), state


def _pv_line_for_ai_payload(row: Optional[AnalyzedMoveData]) -> List[Dict[str, str]]:
    """SAN + FEN after each ply of PV1 for interactive commentary hover boards."""
    if not row or not row.pvs or not row.pvs[0]:
        return []
    out: List[Dict[str, str]] = []
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
