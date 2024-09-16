from src.utils.simulator import Simulator


class ChessGame:
    """Class to represent a chess game and interact with it programmatically."""

    def __init__(
        self,
        pgn_path,
        game_number=1,
        engine_path=r"stockfish.exe",
    ):
        self.simulator = Simulator(pgn_path, game_number, engine_path)
        self.game = self.simulator.game
        self.engine = self.simulator.engine

    def simulate_game(self, display_board=False, print_evaluation=False, n_moves=None):
        """Simulate the entire game and optionally display the board after each move."""

        self.simulator.simulate_game(display_board, print_evaluation, n_moves)

    def get_nth_move(
        self, n, display_board=False, print_evaluation=False, get_chatgpt_comment=False
    ):
        """Retrieve the n-th move and optionally display the board."""

        return self.simulator.get_nth_move(
            n, display_board, print_evaluation, get_chatgpt_comment
        )

    def __del__(self):
        """Close the engine properly on deletion of the object."""

        self.engine.quit()
