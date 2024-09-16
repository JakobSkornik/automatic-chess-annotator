import chess.pgn
import os


class PGNReader:
    def __init__(self, pgn_file_path):
        """
        Initializes the PGNReader with the path to the PGN file.

        :param pgn_file_path: Path to the PGN file.
        """
        if not os.path.isfile(pgn_file_path):
            raise FileNotFoundError(f"File not found: {pgn_file_path}")

        self.pgn_file_path = pgn_file_path

    def read_game(self, n=1):
        """
        Reads the PGN file and returns the first chess.pgn.Game object.

        :return: A chess.pgn.Game object.
        """

        with open(self.pgn_file_path, "r") as pgn_file:
            current_game = None
            for _ in range(n):
                current_game = chess.pgn.read_game(pgn_file)
                if current_game is None:
                    raise ValueError("No such game number exists in the PGN file.")

            self.print_game_logs(current_game)
            return current_game

    def read_all_games(self):
        """
        Reads all games from the PGN file and returns them as a list of chess.pgn.Game objects.

        :return: A list of chess.pgn.Game objects.
        """

        games = []
        with open(self.pgn_file_path, "r") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games

    def print_game_logs(self, game):
        """
        Prints base info of the game.
        """

        print(f"Event: {game.headers['Event']}")
        print(f"Site: {game.headers['Site']}")
        print(f"Date: {game.headers['Date']}")
        print(f"White: {game.headers['White']}")
        print(f"WhiteElo: {game.headers['WhiteElo']}")
        print(f"Black: {game.headers['Black']}")
        print(f"BlackElo: {game.headers['BlackElo']}")
        print(f"Result: {game.headers['Result']}")
        print(f"Opening: {game.headers['Opening']}")
