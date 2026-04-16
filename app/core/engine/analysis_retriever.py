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
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.chroma_rag_retriever import get_default_retriever
from app.core.commentary.rag_retriever import rag_results_to_ws_refs
from app.core.commentary.annotation_tokens import resolve_tokens_for_comment
from app.core.commentary.event_extractor import ChessEventExtractor
from app.core.commentary.episode_segmenter import EpisodeSegmenter
from app.models.chess_events import (
    AnalyzedMoveData,
    Episode,
    GameAnalysisContext,
    MoveEvent,
)

# ANALYSIS_STAGES = [0.05, 0.1, 0.2, 0.4]
ANALYSIS_STAGES = [4, 8, 16]
DEFAULT_PV_COUNT = 3
MATE_SCORE = 1000000
logger = logging.getLogger(__name__)


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

    def get_move_by_depth(self, depth: int) -> Optional[Move]:
        """
        Returns a move from the analyzed game by its depth.
        """
        for move in self.analyzed_game:
            if move.depth == depth:
                return move
        return None

    def get_analysis_stages(self) -> List[float]:  # Corrected type hint
        """
        Returns the analysis stages for the engine.
        """
        return self.analysis_stages

    async def get_full_game_analysis_with_progress(
        self, 
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> Tuple[List[Move], Dict[int, List[List[Move]]]]:
        """
        Returns a full analysis of the game with progress updates.
        
        Args:
            progress_callback: Function called with (current_move, total_moves, phase, move_san)
                             where phase is either 'analyzing_move' or 'analyzing_pvs'
        
        Returns:
            Tuple of (analyzed_moves, all_pvs_dict)
        """
        moves = self.get_move_list()
        total_moves = len(moves)
        analysis_time = 16
        
        analyzed_moves = []
        all_pvs_dict = {}
        
        for move_idx, chess_move in enumerate(moves):
            try:
                # Update progress for move analysis
                if progress_callback:
                    await progress_callback(
                        move_idx, 
                        total_moves, 
                        "analyzing_move", 
                        chess_move.move
                    )
                
                # Analyze the main move
                analyzed_move, all_pvs = self.analyze_move(chess_move, analysis_time)
                analyzed_moves.append(analyzed_move)
                
                # Update progress for PV analysis
                if progress_callback:
                    await progress_callback(
                        move_idx, 
                        total_moves, 
                        "analyzing_pvs", 
                        chess_move.move
                    )
                
                # Store PVs for this move
                if all_pvs:
                    all_pvs_dict[move_idx] = all_pvs
                
                # Brief pause to allow WebSocket to send progress
                # (Optional: depends on your threading model)
                
            except Exception as e:
                logger.error(f"Error analyzing move {chess_move.move}: {e}")
                # Continue with next move instead of stopping
                continue
        
        return analyzed_moves, all_pvs_dict

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
        # Hidden features on the after-move position + before/after delta for strategic context
        try:
            after_features = compute_hidden_features(board_after_move)
        except Exception as e:
            logger.error(
                f"Hidden features (after) failed at depth {main_move_obj.depth} FEN={board_after_move.fen()}: {e}"
            )
            after_features = {"error": str(e)}
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

    def move_trace(self, move: Move) -> Dict:
        """
        Returns the trace of a move.
        """
        board = chess.Board(move.position)
        self.engine_connector.analyse(board, time_limit=0.01, multiPv=1)
        return self.engine_connector.trace()

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

        key_moment = retriever.key_moment_detector.detect(
            analyzed_move,
            analyzed_rows[idx - 1].analyzed_move if idx > 0 else None,
            list(pvs) if pvs else None,
        )

        me = move_events[idx] if idx < len(move_events) else None
        comment: Optional[str] = None
        if me and me.is_critical:
            try:
                hf = analyzed_move.hiddenFeatures or {}
                llm = hf.get("_llm") if isinstance(hf, dict) else None
                if isinstance(llm, dict) and llm.get("comment"):
                    comment = str(llm["comment"])
            except Exception:
                comment = None
        if not comment and key_moment:
            ps = analyzed_move.score if analyzed_move.score is not None else 0
            prev_s = (
                analyzed_rows[idx - 1].analyzed_move.score
                if idx > 0 and analyzed_rows[idx - 1].analyzed_move.score is not None
                else 0
            )
            comment = f"{key_moment.replace('_', ' ').capitalize()} (Score change: {(ps - prev_s) / 100:.2f})"

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
                is_critical=bool(me.is_critical if me else False),
                episode_index=ep_idx,
            )
        )

    return GameJson(
        metadata=state.metadata,
        moves=game_moves,
        episodes=episode_summaries,
        game_narrative=context.game_narrative,
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

    metadata = GameMetadata(
        id=metadata_id or str(uuid.uuid4()),
        white=headers.whiteName,
        black=headers.blackName,
        result=headers.result,
        date=game.headers.get("Date"),
        eventId=headers.event,
        whiteElo=headers.whiteElo,
        blackElo=headers.blackElo,
        opening=headers.opening,
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
            )
        )
        previous_move_obj = analyzed_move

    await progress_callback(92.0, "Extracting events and episodes...")
    extractor = ChessEventExtractor()
    move_events = extractor.extract_events(game, analyzed_rows)
    segmenter = EpisodeSegmenter()
    episodes = segmenter.segment(move_events)

    ply_to_episode: Dict[int, int] = {}
    for ep in episodes:
        for me in ep.move_events:
            ply_to_episode[me.ply] = ep.episode_index

    opening_name = None
    opening_eco = None
    for me in move_events:
        if me.opening_name:
            opening_name = me.opening_name
        if me.opening_eco:
            opening_eco = me.opening_eco
        if opening_name and opening_eco:
            break

    context = GameAnalysisContext(
        metadata={"white": headers.whiteName, "black": headers.blackName, "result": headers.result},
        move_events=move_events,
        episodes=episodes,
        critical_moments=[e for e in move_events if e.is_critical],
        opening_name=opening_name,
        opening_eco=opening_eco,
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


async def run_llm_commentary(
    state: EnginePipelineState,
    advanced_commenter: AdvancedCommentService,
    *,
    progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
    commentary_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    llm_model: Optional[str] = None,
    llm_effort: Optional[str] = None,
) -> None:
    """Pass 4: LLM commentary for critical moves, episodes, and game narrative (streams via callback)."""
    mdl = llm_model or os.environ.get("LLM_DEFAULT_MODEL", "gpt-5-mini")
    eff = llm_effort or os.environ.get("LLM_DEFAULT_EFFORT", "low")
    analyzed_rows = state.analyzed_rows
    move_events = state.move_events
    episodes = state.episodes
    context = state.context
    ply_to_episode = state.ply_to_episode

    critical_list = [me for me in move_events if me.is_critical]
    n_crit = max(len(critical_list), 1)
    crit_idx = 0

    for me in move_events:
        if not me.is_critical:
            continue
        ep = next((e for e in episodes if e.episode_index == ply_to_episode.get(me.ply)), None)
        pct = 95.0 + (crit_idx / n_crit) * 3.0
        crit_idx += 1
        if progress_callback:
            await progress_callback(pct, f"LLM: critical move {me.san} (ply {me.ply})")
        row = analyzed_rows[me.move_index] if 0 <= me.move_index < len(analyzed_rows) else None
        try:
            text, rag_results = await advanced_commenter.analyze_and_compose_event(
                me,
                ep,
                context,
                model=mdl,
                effort=eff,
                key_moment_type=me.key_moment_type,
                analyzed_row=row,
            )
            if text:
                for r in analyzed_rows:
                    if r.ply == me.ply:
                        if isinstance(r.analyzed_move.hiddenFeatures, dict):
                            r.analyzed_move.hiddenFeatures.setdefault("_llm", {})
                            r.analyzed_move.hiddenFeatures["_llm"]["comment"] = text
                        break
            # Frontend GameStateManager uses move id = mainline index + 1 (see loadGameFromJson).
            move_id = me.move_index + 1
            if commentary_callback and text:
                resolved_tokens = resolve_tokens_for_comment(text, me.fen_before, me.fen_after)
                await commentary_callback(
                    "AI_COMMENT_UPDATE",
                    {
                        "moveId": move_id,
                        "context": "mainline",
                        "data": {
                            "summary": text,
                            "commentary": text,
                            "pv_line": _pv_line_for_ai_payload(row),
                            "resolved_tokens": resolved_tokens,
                            "rag_refs": rag_results_to_ws_refs(rag_results),
                        },
                    },
                )
        except Exception as e:
            logger.error(f"LLM move commentary failed at ply {me.ply}: {e}")

    if progress_callback:
        await progress_callback(98.5, "LLM: episode narratives...")
    for ep in episodes:
        try:
            ep.narrative_summary = await advanced_commenter.generate_episode_commentary(
                ep, model=mdl, effort=eff
            )
            if commentary_callback and ep.narrative_summary:
                await commentary_callback(
                    "EPISODE_NARRATIVE",
                    {
                        "episode_index": ep.episode_index,
                        "title": ep.title,
                        "narrative": ep.narrative_summary,
                    },
                )
        except Exception as e:
            logger.error(f"Episode commentary failed: {e}")

    try:
        context.game_narrative = await advanced_commenter.generate_game_narrative(
            context, model=mdl, effort=eff
        )
        if commentary_callback and context.game_narrative:
            await commentary_callback(
                "GAME_NARRATIVE",
                {"narrative": context.game_narrative},
            )
    except Exception as e:
        logger.error(f"Game narrative failed: {e}")

    if progress_callback:
        await progress_callback(99.0, "Commentary complete.")


async def run_full_analysis_to_json(
    pgn_string: str,
    engine_connector: EngineConnector,
    progress_callback: Callable[[float, str], Awaitable[None]],
) -> GameJson:
    """Run engine + LLM in one call (no WebSocket); useful for tests or batch."""
    game_json, state = await run_engine_analysis_to_json(
        pgn_string, engine_connector, progress_callback
    )
    advanced_commenter = AdvancedCommentService(rag_retriever=get_default_retriever())
    await run_llm_commentary(state, advanced_commenter, progress_callback=progress_callback)
    return assemble_game_json(state)
