import chess
import logging
from chess.pgn import Game
from typing import Dict, List, Optional, Tuple, Callable

from app.core.engine.engine_connector import EngineConnector
from app.models.Move import Move
from app.models.PgnMetadata import PgnMetadata

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
        self.analyzed_game = []

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
                    progress_callback(
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
        (
            main_move_obj.capturedByWhite,
            main_move_obj.capturedByBlack,
        ) = self._get_all_captured_pieces(board_after_move)
        main_move_obj.isAnalyzed = True

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
