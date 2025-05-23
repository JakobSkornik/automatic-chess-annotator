import asyncio
import chess
import logging
from chess.pgn import Game
from typing import AsyncGenerator, Dict, List

from app.core.engine.engine_connector import EngineConnector
from app.models.Move import Move, PV
from app.models.MoveAnalysisNode import MoveAnalysisNode
from app.models.PgnMetadata import PgnMetadata

ANALYSIS_STAGES = [0.05, 0.1, 0.2, 0.4]
DEFAULT_PV_COUNT = 3
MATE_SCORE = 1000000
logger = logging.getLogger(__name__)


class AnalysisRetriever:
    def __init__(self, engine_connector: EngineConnector, game: Game):
        self.analysis_stages = ANALYSIS_STAGES
        self.engine_connector = engine_connector
        self.game = game
        self._node_id_counter = 0  # For trace tree node generation

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

    def get_move_list(self) -> List[Move]:
        """
        Returns a list of moves in the game.
        If PGN parsing failed during init, this will return an empty list.
        """
        moves = [
            Move(
                position=self.game.board().fen(),
                move="Start",
                isAnalyzed=False,
                context="mainline",
            )
        ]
        board = self.game.board()

        for chess_move in self.game.mainline_moves():
            san_representation = board.san(chess_move)
            board.push(chess_move)
            fen_after_move = board.fen()

            move_obj = Move(
                position=fen_after_move,
                move=san_representation,
                context="mainline",
                isAnalyzed=False,
            )
            moves.append(move_obj)

        return moves

    def get_analysis_stages(self) -> List[int]:
        """
        Returns the analysis stages for the engine.
        """
        return self.analysis_stages

    def analyze_move(self, move: Move, stage: float) -> Move:
        """
        Analyzes a move using the engine and returns the analysis result.
        """

        # Set the position on the board
        board = chess.Board(move.position)

        # Analyze the move using the engine
        analysis_results = self.engine_connector.analyse(
            board, time_limit=stage, multiPv=DEFAULT_PV_COUNT
        )

        best_pv = analysis_results[0]
        move.score = best_pv.get("score").white().score()
        move.trace = self.engine_connector.trace()
        move.phase = self._determine_game_phase(board)
        move.capturedByWhite, move.capturedByBlack = self.get_all_captured_pieces(board)
        pvs = []

        for pv in analysis_results:
            board = chess.Board(move.position)
            moves = []
            for pv_move in pv.get("pv"):
                moves.append(str(pv_move))
                board.push_san(str(pv_move))

            pv_obj = PV(
                score=pv.get("score").white().score(mate_score=MATE_SCORE) or 0.0,
                moves=moves,
            )
            pvs.append(pv_obj)

        move.isAnalyzed = True
        move.pvs = pvs
        return move

    def move_trace(self, move: Move) -> Dict:
        """
        Returns the trace of a move.
        """
        board = chess.Board(move.position)
        self.engine_connector.analyse(board, time_limit=0.01, multiPv=1)
        trace = self.engine_connector.trace()
        return trace

    def _get_piece_for_move(
        self, board_before_move: chess.Board, san_move: str
    ) -> str | None:
        """Helper to determine the piece (e.g., wP, bN) that made a move."""
        try:
            move = board_before_move.parse_san(san_move)
            piece = board_before_move.piece_at(move.from_square)
            if piece:
                color_char = "w" if piece.color == chess.WHITE else "b"
                return f"{color_char}{piece.symbol().upper()}"
            return None
        except (
            chess.InvalidMoveError
        ):  # Handle cases where SAN might be slightly off or board state unexpected
            logger.warning(
                f"Could not parse SAN '{san_move}' on board FEN: {board_before_move.fen()}"
            )
            return None
        except Exception as e:
            logger.error(f"Error in _get_piece_for_move for SAN '{san_move}': {e}")
            return None

    def get_all_captured_pieces(
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

    async def stream_trace_tree_nodes(
        self, initial_mainline_moves: List[Move]
    ) -> AsyncGenerator[MoveAnalysisNode, None]:
        """
        Asynchronously generates MoveAnalysisNode objects for a trace tree.
        Each move in a PV will also become a node.
        """
        self._node_id_counter = 0  # Reset for each new tree generation
        # Queue stores: (current_san_in_pv: str,
        #                remaining_sans_in_pv_line: List[str],
        #                parent_node_id: int,
        #                board_fen_before_current_san: str,
        #                original_score_of_pv_line: float | None,
        #                is_first_move_in_this_pv_line: bool)
        nodes_to_process_queue = asyncio.Queue()
        quick_analysis_stage = self.analysis_stages[0] if self.analysis_stages else 0.05

        # 1. Process "Start" Node
        self._node_id_counter += 1
        start_node_id = self._node_id_counter
        start_board = chess.Board()  # ply() is 0 here
        start_move_obj_data = Move(
            position=start_board.fen(),
            move="Start",
            isAnalyzed=False,  # Will be set to True by analyze_move
            context="mainline",
        )

        analyzed_start_move = self.analyze_move(
            start_move_obj_data, quick_analysis_stage
        )
        # User's convention: parent -1 for root, depth from board.ply()
        start_node = MoveAnalysisNode(
            id=start_node_id,
            parent=-1,
            depth=start_board.ply(),
            move=analyzed_start_move,
            piece=None,
        )
        yield start_node

        if analyzed_start_move and analyzed_start_move.pvs:
            for pv_info in analyzed_start_move.pvs[
                :3
            ]:  # pv_info contains score and moves (list of SANs)
                if pv_info.moves:  # pv_info.moves is List[str]
                    first_pv_san = pv_info.moves[0]
                    remaining_sans_for_line = pv_info.moves[1:]
                    original_score_for_line = pv_info.score

                    await nodes_to_process_queue.put(
                        (
                            first_pv_san,
                            remaining_sans_for_line,
                            start_node_id,  # Parent is the start_node
                            analyzed_start_move.position,  # FEN of the board before this PV's first move
                            original_score_for_line,
                            True,  # is_first_move_in_this_pv_line
                        )
                    )

        # 2. Process Initial Mainline Moves
        current_parent_id = start_node_id
        current_board = chess.Board()  # ply() is 0, tracks mainline state
        for client_move in initial_mainline_moves:
            if client_move.move == "Start":
                if (
                    current_board.ply() == 0
                    and start_board.fen() == current_board.fen()
                ):
                    continue

            self._node_id_counter += 1
            mainline_node_id = self._node_id_counter

            board_before_this_move_fen = current_board.fen()
            piece_moved = self._get_piece_for_move(current_board, client_move.move)

            try:
                actual_chess_move = current_board.parse_san(client_move.move)
                current_board.push(actual_chess_move)  # ply() increments here
            except Exception as e:
                logger.error(
                    f"Failed to parse/push mainline move {client_move.move} on board {board_before_this_move_fen}: {e}"
                )
                continue
            fen_after_move = current_board.fen()
            current_depth = current_board.ply()

            # Construct Move object for mainline, preserving client's isAnalyzed and pvs if present
            mainline_move_obj = Move(
                position=fen_after_move,
                move=client_move.move,
                isAnalyzed=client_move.isAnalyzed,
                context="mainline",
                pvs=client_move.pvs,  # Carry over PVs if already analyzed by client/previous step
                # Score, trace, phase, captures will be added by analyze_move if not already analyzed
            )

            if (
                not mainline_move_obj.isAnalyzed or not mainline_move_obj.pvs
            ):  # Analyze if needed
                analyzed_mainline_move = self.analyze_move(
                    mainline_move_obj, quick_analysis_stage
                )
            else:
                # If already analyzed and has PVs, ensure other fields like phase, captures are present
                # For simplicity, we can re-set them based on the current board state if they are missing
                # or assume analyze_move would populate them if called.
                # If client_move already has all fields, this is fine.
                # Let's ensure phase and captures are set if we don't re-analyze.
                if not mainline_move_obj.phase:
                    mainline_move_obj.phase = self._determine_game_phase(current_board)
                if (
                    not mainline_move_obj.capturedByWhite
                    and not mainline_move_obj.capturedByBlack
                ):
                    (
                        mainline_move_obj.capturedByWhite,
                        mainline_move_obj.capturedByBlack,
                    ) = self.get_all_captured_pieces(current_board)
                # Trace might be missing if not analyzed by this backend.
                # If trace is critical and not present, a light analysis/trace call might be needed.
                # For now, we assume client_move.pvs implies sufficient prior analysis.
                analyzed_mainline_move = mainline_move_obj

            node = MoveAnalysisNode(
                id=mainline_node_id,
                parent=current_parent_id,
                depth=current_depth,
                move=analyzed_mainline_move,
                piece=piece_moved,
            )
            yield node

            if analyzed_mainline_move.pvs:
                for pv_info in analyzed_mainline_move.pvs[:3]:
                    if pv_info.moves:
                        first_pv_san = pv_info.moves[0]
                        remaining_sans_for_line = pv_info.moves[1:]
                        original_score_for_line = pv_info.score
                        await nodes_to_process_queue.put(
                            (
                                first_pv_san,
                                remaining_sans_for_line,
                                mainline_node_id,  # Parent is the current mainline_node
                                analyzed_mainline_move.position,  # FEN after mainline_move_obj
                                original_score_for_line,
                                True,  # is_first_move_in_this_pv_line
                            )
                        )
            current_parent_id = mainline_node_id

        # 3. Process PV Queue (now processes each move in a PV line)
        while not nodes_to_process_queue.empty():
            (
                current_san_to_process,
                remaining_sans_in_line,
                parent_id_for_current_san,
                board_fen_before_current_san,
                original_line_score,
                is_first_in_pv_line,
            ) = await nodes_to_process_queue.get()

            self._node_id_counter += 1
            current_pv_node_id = self._node_id_counter

            temp_board_for_pv = chess.Board(board_fen_before_current_san)
            piece_moved_in_pv = self._get_piece_for_move(
                temp_board_for_pv, current_san_to_process
            )

            try:
                actual_pv_chess_move = temp_board_for_pv.parse_san(
                    current_san_to_process
                )
                temp_board_for_pv.push(actual_pv_chess_move)  # ply() increments here
            except Exception as e:
                logger.error(
                    f"Failed to parse/push PV move {current_san_to_process} on board {board_fen_before_current_san}: {e}"
                )
                nodes_to_process_queue.task_done()
                continue

            fen_after_current_san = temp_board_for_pv.fen()
            current_pv_depth = temp_board_for_pv.ply()

            # For PV moves, we only want trace. Score and PV line are from parent's analysis.
            # Create a Move object for this specific step in the PV.

            # Score for this specific Move object in the PV line
            # Typically, only the first move of an engine's PV output has the direct evaluation score for that line.
            # Subsequent moves in the PV are just the sequence.
            move_obj_score = original_line_score if is_first_in_pv_line else None

            # The 'pvs' for this move object will be the continuation of its own line.
            move_obj_pvs = []
            if remaining_sans_in_line:  # If this is not the last move of the PV
                # The PV associated with this move is the rest of the line, with the original line's score
                move_obj_pvs.append(
                    PV(score=original_line_score, moves=remaining_sans_in_line)
                )

            # Get trace for the current PV move's resulting position
            # Create a minimal Move object just for the trace call, as move_trace uses move.position
            move_for_trace_call = Move(
                position=fen_after_current_san,
                move=current_san_to_process,
                context="variation",
                isAnalyzed=False,
            )
            trace_data = self.move_trace(move_for_trace_call)

            # Determine phase and captures for the board state after this PV move
            current_phase = self._determine_game_phase(temp_board_for_pv)
            captures_white, captures_black = self.get_all_captured_pieces(
                temp_board_for_pv
            )

            # Construct the final Move object for this PV step
            processed_pv_move_obj = Move(
                position=fen_after_current_san,
                move=current_san_to_process,
                score=move_obj_score,
                pvs=move_obj_pvs,
                trace=trace_data,
                isAnalyzed=True,  # Considered analyzed as it's part of an engine's PV
                context="variation",
                phase=current_phase,
                capturedByWhite=captures_white,
                capturedByBlack=captures_black,
            )

            node = MoveAnalysisNode(
                id=current_pv_node_id,
                parent=parent_id_for_current_san,
                depth=current_pv_depth,
                move=processed_pv_move_obj,
                piece=piece_moved_in_pv,
            )
            yield node

            # If there are more moves in this PV line, enqueue the next one
            if remaining_sans_in_line:
                next_san_in_line = remaining_sans_in_line[0]
                further_remaining_sans = remaining_sans_in_line[1:]
                await nodes_to_process_queue.put(
                    (
                        next_san_in_line,
                        further_remaining_sans,
                        current_pv_node_id,  # Parent is the node just created
                        fen_after_current_san,  # Board state after the current_san_to_process
                        original_line_score,  # Propagate the original score of the line
                        False,  # This is a continuation, not the first in its PV line
                    )
                )
            nodes_to_process_queue.task_done()
