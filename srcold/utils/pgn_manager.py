import chess.pgn
from chess.engine import SimpleEngine
from tabulate import tabulate


class PgnManager:
    """Class to manage PGN files."""

    def __init__(self, engine: SimpleEngine):
        self.engine = engine

    def load_nth_pgn(self, pgn_path, n=1):
        """Load the n-th game from a PGN file."""
        game = None

        with open(pgn_path) as pgn_file:
            current_game = None
            for _ in range(n):
                current_game = chess.pgn.read_game(pgn_file)
                if current_game is None:
                    raise ValueError("No such game number exists in the PGN file.")
            game = current_game

            table = [
                ["Game number", n],
                ["Round", current_game.headers["Round"]],
                [
                    "White",
                    f"{current_game.headers['White']} {current_game.headers['WhiteElo']}ELO",
                ],
                [
                    "Black",
                    f"{current_game.headers['Black']} {current_game.headers['BlackElo']}ELO",
                ],
                ["Result", current_game.headers["Result"]],
            ]
            print(tabulate(table, tablefmt="outline"))

        return game
