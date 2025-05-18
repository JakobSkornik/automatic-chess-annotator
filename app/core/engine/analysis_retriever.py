import chess
from chess.pgn import Game
from typing import Optional, List, Dict
from collections import deque
import logging

from app.core.engine.engine_connector import EngineConnector
from app.models.AnalysisResponse import AnalysisResponse
from app.models.Move import Move
from app.models.MoveAnalysisNode import MoveAnalysisNode
from app.models.PgnMetadata import PgnMetadata

MATE_SCORE = 1000000
logger = logging.getLogger(__name__)


class AnalysisRetriever:
    def __init__(
        self,
        engine_connector: EngineConnector,
        shallow_depth: int = 5,
        deep_depth: int = 12,
    ):
        self.engine_connector = engine_connector
        self.shallow_depth = shallow_depth
        self.deep_depth = deep_depth
        self.node_counter = 0

    def retrieve_analysis(self, game: Game) -> AnalysisResponse:
        analysis_queue = deque()
        tree = {}

        metadata = self._extract_metadata(game)
        self._initialize_queue_with_mainline(game, analysis_queue, tree)
        self._process_analysis_queue(analysis_queue, tree)
        moves = self._generate_move_list_from_tree(tree)

        print(f"Analysis completed: {len(tree)} positions analyzed")
        serialized_tree = {node_id: node.__dict__ for node_id, node in tree.items()}
        return AnalysisResponse(
            metadata=metadata, moves=moves, move_tree=serialized_tree
        )

    def _extract_metadata(self, game: Game) -> PgnMetadata:
        """Extract game metadata from PGN headers."""
        headers = game.headers

        return PgnMetadata(
            white_name=headers.get("White", ""),
            black_name=headers.get("Black", ""),
            white_elo=(
                int(headers.get("WhiteElo", 0))
                if headers.get("WhiteElo", "").isdigit()
                else None
            ),
            black_elo=(
                int(headers.get("BlackElo", 0))
                if headers.get("BlackElo", "").isdigit()
                else None
            ),
            event=headers.get("Event", ""),
            opening=headers.get("Opening", ""),
            result=headers.get("Result", ""),
        )

    def _initialize_queue_with_mainline(self, game: Game, queue: deque, tree: Dict):
        board = chess.Board()
        parent_id = -1

        # For tracking mainline positions to avoid duplicate PV analysis
        self.mainline_positions = set()

        # Create root node
        phase = self._determine_game_phase(board)
        root_node = self._create_node(
            parent=parent_id,
            depth=0,
            move="start",
            fen=board.fen(),
            shallow_score=0,
            deep_score=0,
            trace={},
            context="mainline",
            phase=phase,
            piece=None,  # No piece for root
        )

        # Add root FEN to mainline positions
        self.mainline_positions.add(board.fen())

        # Add root node to tree
        tree[root_node.id] = root_node

        # Add starting position to queue
        queue.append(
            {
                "fen": board.fen(),
                "move": None,
                "context": "mainline",
                "parent_id": parent_id,
                "depth": 0,
                "node_id": root_node.id,
                "capturedByWhite": {k: 0 for k in "pnbrqk"},
                "capturedByBlack": {k: 0 for k in "pnbrqk"},
            }
        )

        # Add mainline moves to queue
        parent_id = root_node.id
        current_depth = 0

        for move in game.mainline_moves():
            move_uci = move.uci()

            # Track captures before making the move
            captures_white, captures_black = self._track_captures(board, move)

            board.push(move)
            current_depth = board.ply()

            # Add this position to mainline positions
            self.mainline_positions.add(board.fen())

            # Determine phase
            phase = self._determine_game_phase(board)

            # Create the node and add to tree
            piece_initial = self._get_piece_initial(board, move)
            node = self._create_node(
                parent=parent_id,
                depth=current_depth,
                move=move_uci,
                fen=board.fen(),
                shallow_score=0,  # Placeholder, will be filled during analysis
                deep_score=0,  # Placeholder, will be filled during analysis
                trace={},  # Placeholder, will be filled during analysis
                context="mainline",  # Changed from mainline=True
                phase=phase,
                capturedByWhite=captures_white,
                capturedByBlack=captures_black,
                piece=piece_initial,
            )

            # Add node to tree
            tree[node.id] = node

            queue.append(
                {
                    "fen": board.fen(),
                    "move": move_uci,
                    "context": "mainline",
                    "parent_id": parent_id,
                    "depth": current_depth,
                    "node_id": node.id,
                    "capturedByWhite": captures_white,
                    "capturedByBlack": captures_black,
                }
            )

            # Update parent ID for the next move
            parent_id = node.id

    def _track_captures(self, board: chess.Board, move: chess.Move) -> tuple:
        """Track captures for a move and return updated capture counters."""
        captures_white = {k: 0 for k in "pnbrqk"}
        captures_black = {k: 0 for k in "pnbrqk"}

        is_capture = board.is_capture(move)
        if is_capture:
            captured_piece_square = move.to_square
            captured_piece = board.piece_type_at(captured_piece_square)

            if captured_piece:
                piece_symbol = chess.piece_symbol(captured_piece).lower()
                if board.turn == chess.WHITE:
                    captures_white[piece_symbol] = 1
                else:
                    captures_black[piece_symbol] = 1

        return captures_white, captures_black

    def _process_analysis_queue(self, queue: deque, tree: Dict):
        while queue:
            print(f"Queue length: {len(queue)}")
            job = queue.popleft()
            fen = job["fen"]
            move = job["move"]
            parent_id = job["parent_id"]
            node_id = job.get("node_id")  # This could be None for alternative/PV moves
            context = job["context"]  # Get the context type

            # Handle both mainline and PV contexts
            is_mainline = context == "mainline"
            is_top_pv = context == "pv1" if not is_mainline else False

            # Check if node already exists in tree
            existing_node = None
            if node_id is not None and node_id in tree:
                existing_node = tree[node_id]
            else:
                # Look for a node with matching FEN and parent
                for n in tree.values():
                    if (
                        n.fen == fen
                        and n.parent == parent_id
                        and n.move == (move if move else "start")
                    ):
                        existing_node = n
                        break

            # Analyze the position (directly, no cache)
            board = chess.Board(fen)

            # Convert move to chess.Move if it's a string and not None
            move_obj = None
            if move and isinstance(move, str):
                try:
                    move_obj = chess.Move.from_uci(move)
                except Exception:
                    move_obj = None
            elif isinstance(move, chess.Move):
                move_obj = move

            # Run deep analysis only for mainline moves and top PV moves
            run_deep = is_mainline or is_top_pv
            analysis_result = self._analyze_position(board, run_deep_analysis=run_deep)

            # Determine the game phase for this position
            phase = self._determine_game_phase(board)

            if existing_node:
                # Update existing node with analysis results
                existing_node.shallow_score = analysis_result["shallow_score"]
                existing_node.deep_score = analysis_result["deep_score"]
                existing_node.trace = analysis_result["trace"]
                existing_node.phase = phase  # Update phase
                node = existing_node
            else:
                # Create new node and add to tree
                piece_initial = self._get_piece_initial(board, move_obj)
                node = self._create_node(
                    parent=parent_id,
                    depth=job["depth"],
                    move=move if move else "start",
                    fen=fen,
                    shallow_score=analysis_result["shallow_score"],
                    deep_score=analysis_result["deep_score"],
                    trace=analysis_result["trace"],
                    context=context,
                    phase=phase,
                    capturedByWhite=job["capturedByWhite"],
                    capturedByBlack=job["capturedByBlack"],
                    piece=piece_initial,
                )
                tree[node.id] = node

            # Only enqueue PVs for mainline nodes - not for PV nodes themselves
            if is_mainline:
                self._enqueue_alternative_moves(board, node.id, queue, tree)
                self._enqueue_principal_variation(
                    board, node.id, queue, analysis_result, tree
                )

    def _analyze_position(
        self, board: chess.Board, run_deep_analysis: bool = True
    ) -> Dict:
        """Analyze a position without caching."""
        # Define multipv parameter - use more for deep analysis
        shallow_multipv = 4  # Just one line for shallow analysis
        deep_multipv = 8  # Multiple lines for deep analysis

        # Always run shallow analysis
        shallow = self.engine_connector.analyse(
            board, depth=self.shallow_depth, multiPv=shallow_multipv
        )
        shallow_score = self._extract_score(shallow)

        # Only run deep analysis for mainline moves
        if run_deep_analysis:
            deep = self.engine_connector.analyse(
                board, depth=self.deep_depth, multiPv=deep_multipv
            )
            deep_score = self._extract_score(deep)
            pvs = self._extract_pv(deep)
        else:
            # For non-mainline moves, use shallow analysis results for deep score too
            deep_score = shallow_score
            pvs = self._extract_pv(shallow)

        trace = self.engine_connector.trace()

        # Handle case where trace is a string
        if isinstance(trace, str):
            trace = {"error": {"message": trace}}
        else:
            # Process trace to use only mg values and remove FinalEvaluation
            trace = self._process_trace(trace)

        return {
            "shallow_score": shallow_score,
            "deep_score": deep_score,
            "trace": trace,
            "pvs": pvs,  # Now returning multiple PVs
        }

    def _process_trace(self, trace: dict) -> dict:
        """
        Process the engine trace to:
        1. Keep both midgame (mg) and endgame (eg) values
        2. Remove FinalEvaluation key
        """
        if not isinstance(trace, dict):
            return trace

        processed_trace = {}

        # Process each key in the trace
        for key, value in trace.items():
            # Skip FinalEvaluation key
            if key == "FinalEvaluation":
                continue

            # Keep the original value structure with both mg and eg
            processed_trace[key] = value

        return processed_trace

    def _determine_game_phase(self, board: chess.Board) -> str:
        """
        Determine the game phase based on piece count and other factors.
        Returns "early", "mid", or "end".
        """
        # Count pieces
        pieces = board.piece_map()
        piece_count = len(pieces)

        # Count pawns
        pawn_count = sum(
            1 for piece in pieces.values() if piece.piece_type == chess.PAWN
        )

        # Count major pieces (rooks and queens)
        major_piece_count = sum(
            1
            for piece in pieces.values()
            if piece.piece_type == chess.ROOK or piece.piece_type == chess.QUEEN
        )

        # Simple rules
        if piece_count >= 24:  # Most pieces still on board
            return "early"
        elif piece_count <= 12 or (
            pawn_count <= 8 and major_piece_count <= 3
        ):  # Few pieces or few pawns and major pieces
            return "end"
        else:
            return "mid"

    def _enqueue_alternative_moves(
        self, board: chess.Board, parent_id: int, queue: deque, tree: Dict
    ):
        for move in board.legal_moves:
            move_uci = move.uci()

            # Track captures properly
            captures_white, captures_black = self._track_captures(board, move)

            new_board = board.copy()
            new_board.push(move)
            new_fen = new_board.fen()

            # Check if this position is already in the tree with this parent
            skip_enqueue = False
            for node in tree.values():
                if node.fen == new_fen and node.parent == parent_id:
                    skip_enqueue = True
                    break

            if skip_enqueue:
                continue

            queue.append(
                {
                    "fen": new_fen,
                    "move": move_uci,
                    "context": "alternative",
                    "parent_id": parent_id,
                    "depth": new_board.ply(),
                    "node_id": None,
                    "capturedByWhite": captures_white,
                    "capturedByBlack": captures_black,
                }
            )

    def _enqueue_principal_variation(
        self,
        board: chess.Board,
        parent_id: int,
        queue: deque,
        analysis_result: Dict,
        tree: Dict,
    ):
        """Enqueue all principal variations for analysis."""
        if "pvs" not in analysis_result or not analysis_result["pvs"]:
            return

        # Get all PVs from the analysis result
        all_pvs = analysis_result["pvs"]

        # Process each PV line (limited to top 2 for efficiency)
        for pv_index, pv_line in enumerate(all_pvs[:2]):  # Only process top 2 lines
            pv_board = board.copy()
            current_parent_id = parent_id

            # Add a tag for PV line number in the queue items
            pv_tag = f"pv{pv_index+1}"

            # Maximum number of moves to include from each PV
            max_pv_depth = 5 if pv_index == 0 else 3  # More moves for first PV

            # Process each move in the PV line
            for move_idx, move_uci in enumerate(pv_line[:max_pv_depth]):
                try:
                    move = chess.Move.from_uci(move_uci)

                    # Track captures properly for PV moves
                    captures_white, captures_black = self._track_captures(
                        pv_board, move
                    )

                    pv_board.push(move)
                    new_fen = pv_board.fen()

                    # Skip this PV move if it's in the mainline
                    if new_fen in self.mainline_positions:
                        break  # Skip the rest of this PV since it's following mainline

                    # Check if this position is already in the tree with this parent
                    skip_enqueue = False
                    existing_node_id = None

                    for node in tree.values():
                        if node.fen == new_fen and node.parent == current_parent_id:
                            skip_enqueue = True
                            existing_node_id = node.id
                            break

                    if skip_enqueue:
                        # If this position already exists, use it as parent for next PV move
                        current_parent_id = existing_node_id
                        continue

                    # Determine phase for this PV position
                    phase = self._determine_game_phase(pv_board)

                    queue.append(
                        {
                            "fen": new_fen,
                            "move": move_uci,
                            "context": pv_tag,  # Use PV tag in context
                            "parent_id": current_parent_id,
                            "depth": pv_board.ply(),
                            "node_id": None,  # Will be assigned when processed
                            "capturedByWhite": captures_white,
                            "capturedByBlack": captures_black,
                        }
                    )

                    # Since we don't know the node ID yet (it will be created when processed),
                    # we need to make sure we create a unique parent ID reference for next move
                    piece_initial = self._get_piece_initial(pv_board, move)
                    temp_node = self._create_node(
                        parent=current_parent_id,
                        depth=pv_board.ply(),
                        move=move_uci,
                        fen=new_fen,
                        shallow_score=0,  # Placeholder
                        deep_score=0,  # Placeholder
                        trace={},  # Placeholder
                        context=pv_tag,  # Changed from mainline=False, now using the pv1/pv2 tag
                        phase=phase,
                        capturedByWhite=captures_white,
                        capturedByBlack=captures_black,
                        piece=piece_initial,
                    )
                    tree[temp_node.id] = temp_node

                    # Update parent ID for the next move in sequence
                    current_parent_id = temp_node.id

                except chess.IllegalMoveError:
                    break

    def _generate_move_list_from_tree(
        self, tree: Dict[int, MoveAnalysisNode]
    ) -> List[Move]:
        moves = []
        cumulative_captures_white = {k: 0 for k in "pnbrqk"}
        cumulative_captures_black = {k: 0 for k in "pnbrqk"}

        # First collect and sort all mainline nodes
        mainline_nodes = [
            node for node in tree.values() if node.context == "mainline"
        ]  # Changed from node.mainline
        mainline_nodes.sort(key=lambda x: x.depth)  # Sort by depth

        # Process them in order to maintain running capture totals
        for node in mainline_nodes:
            # Add current node's captures to the running totals
            for piece, count in node.capturedByWhite.items():
                cumulative_captures_white[piece] += count

            for piece, count in node.capturedByBlack.items():
                cumulative_captures_black[piece] += count

            # Find best continuations from PV in the tree
            best_continuations = self._find_best_continuations(node, tree)

            move_entry = Move(
                position=node.fen,
                move=node.move if node.move != "start" else "",
                shallow_score=node.shallow_score,
                deep_score=node.deep_score,
                phase=node.phase,
                trace=node.trace,
                bestContinuations=best_continuations,
                # Use the cumulative captures rather than just this node's captures
                capturedByWhite=cumulative_captures_white.copy(),
                capturedByBlack=cumulative_captures_black.copy(),
            )
            moves.append(move_entry)

        return moves

    def _find_best_continuations(
        self, node: MoveAnalysisNode, tree: Dict[int, MoveAnalysisNode]
    ) -> List[str]:
        """Find the best continuations (PV) for a given node."""
        continuations = []
        seen_moves = set()  # Track moves we've already added

        # Find direct child nodes
        child_nodes = [n for n in tree.values() if n.parent == node.id]

        # Sort by score (best moves first)
        if child_nodes:
            # Sort based on whose turn it is
            board = chess.Board(node.fen)
            if board.turn == chess.WHITE:
                # White to move - highest score is best
                child_nodes.sort(key=lambda n: n.shallow_score, reverse=True)
            else:
                # Black to move - lowest score is best
                child_nodes.sort(key=lambda n: n.shallow_score)

            # Get moves from top 5 best child nodes, avoiding duplicates
            for child in child_nodes[:5]:
                if (
                    child.move
                    and child.move != "start"
                    and child.move not in seen_moves
                ):
                    continuations.append(
                        {"move": child.move, "score": child.shallow_score}
                    )
                    seen_moves.add(child.move)  # Mark this move as seen

        return continuations

    def _create_node(
        self,
        parent: int,
        depth: int,
        move: str,
        fen: str,
        shallow_score: int,
        deep_score: int,
        trace: Dict,
        context: str,  # Changed from mainline: bool
        phase: str = None,  # Add phase parameter with default
        capturedByWhite: Dict[str, int] = None,
        capturedByBlack: Dict[str, int] = None,
        piece: str = None,
    ) -> MoveAnalysisNode:
        if capturedByWhite is None:
            capturedByWhite = {k: 0 for k in "pnbrqk"}
        if capturedByBlack is None:
            capturedByBlack = {k: 0 for k in "pnbrqk"}

        # If phase wasn't provided, determine it now
        if phase is None:
            board = chess.Board(fen)
            phase = self._determine_game_phase(board)

        node = MoveAnalysisNode(
            id=self.node_counter,
            depth=depth,
            parent=parent,
            move=move,
            fen=fen,
            shallow_score=shallow_score,
            deep_score=deep_score,
            trace=trace,
            context=context,  # Changed from mainline
            phase=phase,  # Add phase
            capturedByWhite=capturedByWhite,
            capturedByBlack=capturedByBlack,
            piece=piece,
        )
        self.node_counter += 1
        return node

    @staticmethod
    def _extract_score(analysis_result: dict) -> int:
        score = analysis_result[0].get("score")
        return score.white().score(mate_score=MATE_SCORE)

    @staticmethod
    def _extract_pv(analysis_result: dict) -> Optional[List[List[str]]]:
        """Extract all principal variations from analysis result.

        Args:
            analysis_result: Analysis result from engine

        Returns:
            List of principal variations, where each PV is a list of move UCIs
            or None if no PVs found
        """
        all_pvs = []

        # Process all analysis entries (multiple PVs)
        for entry in analysis_result:
            if "pv" in entry and entry["pv"]:
                pv_line = [move.uci() for move in entry["pv"]]
                all_pvs.append(pv_line)

        return all_pvs if all_pvs else None

    def _get_piece_initial(self, board: chess.Board, move: chess.Move) -> str:
        """Return the piece initial (k, q, r, n, b, p) for the move."""
        if move is None:
            return None
        piece = board.piece_at(move.to_square)
        if piece is None:
            return None
        return chess.piece_symbol(piece.piece_type).lower()
