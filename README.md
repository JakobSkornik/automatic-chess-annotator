## Installation

### Windows

#### Requirements

- `Python 3.10` installed.
- `venv` package installed.
- `stockfish` executable present in root directory of this project named `stockfish.exe`.

#### Creating environment

- `python -m venv chess_v3.10`
- `.\chess_v3.10\Scripts\activate.bat`
- `pip install -r requirements.txt`
- `uvicorn app.main:app --reload`

Below is a detailed plan outlining the additional components, refactoring steps, and modularization strategies to move toward your full vision of an automatic chess annotator. This plan breaks the work into clear, manageable modules and outlines the functionality each should encapsulate:

---

### 1. `handle_get_initial_state(self) -> MoveAnalysisNode:`

- Set self.current_board to the initial FEN (chess.Board()).
- Perform deep analysis on self.current_board (depth=self.deep_depth, multiPv=self.\* default_pv_count).
- Extract score and PVs.
- Get trace using self.engine_connector.trace().
- Create a root MoveAnalysisNode (parent=-1, move="start", context="root").
- Store it in self.move_tree.
- Return the root node.

### 2. `handle_request_analysis(self, move_uci: str, parent_node_id: int, parent_fen: str | None) -> * Dict[int, MoveAnalysisNode]:`

- `updated_nodes_batch: Dict[int, MoveAnalysisNode] = {}`
- a. Setup Board from Parent:
  - Retrieve parent_node from self.move_tree[parent_node_id].
  - Set self.current_board.set_fen(parent_node.fen). (Verify with parent_fen if provided).
- b. Analyze Requested Move (Deep):
  - Parse move_uci to chess.Move. Check legality on self.current_board. If illegal, raise error.
  - `board_after_move = self.current_board.copy()`
  - `board_after_move.push(chess.Move.from_uci(move_uci))`
  - `analysis_main = self.engine_connector.analyse(board_after_move, depth=self.deep_depth, * multiPv=self.default_pv_count)`
  - `score_main = self._extract_score(analysis_main)`
  - `pvs_main = self._extract_pv(analysis_main)`
  - `trace_main = self._process_trace(self.engine_connector.trace())` (assuming trace is for last \* analysis)
  - Create m`ain_move_node = self._create_node(...)` for move_uci with parent_node_id, context="interactive", deep score, trace.
  - `self.move_tree[main_move_node.id] = main_move_node`
  - `updated_nodes_batch[main_move_node.id] = main_move_node`
- c. Process PVs of Requested Move (Trace Only):
- `current_pv_parent_id = main_move_node.id`
  - For each pv_line in pvs_main (up to self.default_pv_count):
  - `board_for_pv = board_after_move.copy(`) (board state where the PV starts)
  - `temp_pv_parent_id = current_pv_parent_id`
  - For pv_move_uci in pv_line:
  - board_for_pv.push(chess.Move.from_uci(pv_move_uci))
  - trace_pv = self.\_get_trace_for_board(board_for_pv) (helper method, see below)
  - Create pv_node = self.\_create_node(...) for pv_move_uci, parent=temp_pv_parent_id context=f"pv{idx+1}", trace=trace_pv, score=0 (or from shallow if used).
  - self.move_tree[pv_node.id] = pv_node
  - updated_nodes_batch[pv_node.id] = pv_node
  - temp_pv_parent_id = pv_node.id
- d. Process Sibling Moves (Trace Only):
  - board_for_siblings = self.current_board.copy() (board state before move_uci)
  - For sibling_move_obj in board_for_siblings.legal_moves:
  - If sibling_move_obj.uci() == move_uci, continue (already processed).
  - board_after_sibling = board_for_siblings.copy()
  - board_after_sibling.push(sibling_move_obj)
  - trace_sibling = self.\_get_trace_for_board(board_after_sibling)
  - Create sibling_node = self.\_create_node(...) for sibling_move_obj.uci(), parent=parent_node_id, context="alternative", trace=trace_sibling, score=0.
  - self.move_tree[sibling_node.id] = sibling_node
  - updated_nodes_batch[sibling_node.id] = sibling_node
  - Return updated_nodes_batch.

### Helper \_get_trace_for_board(self, board: chess.Board) -> Dict:

- This method encapsulates getting the trace.
- If engine_connector.trace() works after just setting position:
- self.engine_connector.engine.position(board) (if direct access to engine's position setting is needed)
- raw_trace = self.engine_connector.trace()
- Else, if a shallow analysis is needed:
- analysis_shallow = self.engine_connector.analyse(board, depth=1, multiPv=1)
- raw_trace = self.engine_connector.trace() (assuming trace is available after any analysis)
- (Optionally, could extract shallow_score here too for PV/alternative nodes)
- Return self.\_process_trace(raw_trace).
- Your existing \_create_node, \_extract_score, \_extract_pv, \_process_trace methods can be reused within the AnalysisSession class, possibly with minor adaptations.

Next Steps (Sequential Implementation Plan):

- Define WebSocket Message Models: Create the Pydantic models in app/models_ws/ for client and server messages, including the MessageType enums.
- Implement Basic API Endpoint (/api/v1/analysis/submit_pgn):
- Create AnalysisSession class (initially simple: session_id, engine_connector, move_tree={}, node_counter=0).
- Implement the PGN submission, session creation, and storage in active_sessions.
- Implement Basic WebSocket Endpoint (/ws/analysis/{session_id}):
- Setup FastAPI WebSocket route.
- Handle connection, retrieve session, handle disconnection.
- Implement receiving GET_INITIAL_STATE and sending a placeholder INITIAL_STATE response.
- Implement Initial State Analysis:
- Flesh out AnalysisSession.handle_get_initial_state() to perform deep analysis on the root position and create the root node.
- Update WebSocket endpoint to call this and send the actual InitialStatePayload.
- Implement Core REQUEST_ANALYSIS Logic (Part 1: Deep analysis of requested move):
- Flesh out AnalysisSession.handle_request_analysis() up to step (b) - analyzing the main requested move.
- Update WebSocket endpoint to call this and send AnalysisUpdatePayload with just the main new node.
- Implement REQUEST_ANALYSIS Logic (Part 2: PVs - Trace Only):
- Implement step (c) in handle_request_analysis() and the \_get_trace_for_board() helper.
- Update AnalysisUpdatePayload to include PV nodes.
- Implement REQUEST_ANALYSIS Logic (Part 3: Siblings - Trace Only):
- Implement step (d) in handle_request_analysis().
- Update AnalysisUpdatePayload to include sibling nodes.
- Refinement and Error Handling: Add robust error handling, logging, and consider edge cases.
- This structured approach should allow for incremental development and testing. The most crucial part to verify early on will be the behavior of engine_connector.trace() and how to reliably get static evaluation terms for the "trace-only" analyses.
