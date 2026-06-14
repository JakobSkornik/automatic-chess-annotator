import io

import chess.pgn


class PGNReader:
    @staticmethod
    def count_games(pgn_string: str) -> int:
        """Count the number of games in a PGN string."""
        pgn_file = io.StringIO(pgn_string)
        count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            count += 1
        return count

    @staticmethod
    def validate_single_game(pgn_string: str) -> None:
        """Raise ValueError if the PGN contains zero or more than one game."""
        count = PGNReader.count_games(pgn_string)
        if count == 0:
            raise ValueError("Could not parse PGN data.")
        if count > 1:
            raise ValueError(
                "PGN contains multiple games. Only a single game is allowed."
            )

    @staticmethod
    def read_game_from_string(pgn_string: str) -> chess.pgn.Game | None:
        """
        Reads a game from a PGN string.

        :param pgn_string: PGN data as a string.
        :return: A chess.pgn.Game object or None.
        """
        pgn_file = io.StringIO(pgn_string)
        current_game = chess.pgn.read_game(pgn_file)
        if current_game is None:
            raise ValueError("Could not parse PGN data.")
        return current_game

    @staticmethod
    def get_game_logs(game: chess.pgn.Game) -> dict:
        """
        Extracts basic information from a chess.pgn.Game object.

        :param game: A chess.pgn.Game object.
        :return: Dictionary containing game metadata.
        """
        return {
            "Event": game.headers.get("Event", "Unknown"),
            "Site": game.headers.get("Site", "Unknown"),
            "Date": game.headers.get("Date", "Unknown"),
            "White": game.headers.get("White", "Unknown"),
            "WhiteElo": game.headers.get("WhiteElo", "Unknown"),
            "Black": game.headers.get("Black", "Unknown"),
            "BlackElo": game.headers.get("BlackElo", "Unknown"),
            "Result": game.headers.get("Result", "Unknown"),
            "Opening": game.headers.get("Opening", "Unknown"),
        }
