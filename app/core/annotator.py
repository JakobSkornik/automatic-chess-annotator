import chess


class Annotator:
    def __init__(self, move_evaluations):
        """
        Initializes the Annotator with a list of move evaluations.

        :param move_evaluations: A list of dictionaries, each containing move information, evaluation, and FEN string.
        """
        self.move_evaluations = move_evaluations

    def annotate_game(self, game, debug=False):
        """
        Annotates a chess.pgn.Game object with comments based on the move evaluations.

        :param game: A chess.pgn.Game object representing the game to be annotated.
        :param debug: A boolean indicating whether to display debug information.
        :return: A new chess.pgn.Game object with annotations.
        """
        node = game
        node = node.variation(0)
        eval_index = 0

        # Loop through the moves in the game
        for move in game.mainline_moves():
            if eval_index >= len(self.move_evaluations) - 1:
                break

            move_eval = self.move_evaluations[eval_index]
            eval_move = move_eval["move"]

            # Check if the current move in the game matches the move evaluation
            if move == eval_move:
                # Add a comment to the current node
                node.comment = self._create_comment(move_eval, debug)
                eval_index += 1  # Move to the next evaluation
            else:
                print(f"Move mismatch: {move} != {eval_move}")

            # Move to the next node in the game tree
            node = node.variation(0)

        return game

    def _create_comment(self, move_eval, debug=False):
        """
        Creates a comment string based on the move evaluation.

        :param move_eval: A dictionary containing the move, its evaluation, and the engine score.
        :param debug: A boolean indicating whether to display debug information.
        :return: A string comment to be added to the PGN.
        """
        move = move_eval["move"]

        tag = move_eval["evaluation"].get("tag", "")
        score = move_eval["evaluation"].get("score", None)
        explanation = move_eval["evaluation"].get("annotation", "")
        best_continuations = move_eval["evaluation"].get("best_continuations", [])

        # Best continuations (if available, handle tuples)
        if best_continuations:
            best_continuations_str = ", ".join(
                [f"{move} ({score:0.2f})" for move, score in best_continuations]
            )
        else:
            best_continuations_str = "N/A"

        # Build the comment
        comment_parts = []

        comment_parts.append(
            f"Comment: {tag if tag else ''} {explanation if explanation else ''}"
        )
        comment_parts.append(f"Score: {score:0.2f}")
        comment_parts.append(f"Best Continuations: [{best_continuations_str}]")

        comment = " | ".join(comment_parts)

        return comment

