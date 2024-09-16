import chess.engine


class MoveAnalyser:
    """Class that analyses individual moves."""

    def analyse_move(self, chess_game, move):
        """Analyse a move."""

        moves = self.get_legal_moves_and_scores(chess_game)

        if self.check_brilliant_move(chess_game, move, moves):
            return "!!"

        if self.check_gross_mistake(chess_game, move, moves):
            return "??"

        if self.check_mistake(chess_game, move, moves):
            return "?"

    def get_legal_moves_and_scores(self, chess_game):
        """Get all legal moves and their scores."""

        game = chess_game.game
        board = game.board()

        moves = []
        for legal_move in board.legal_moves:
            board.push(legal_move)
            legal_move_info = chess_game.engine.analyse(
                board, chess.engine.Limit(time=0.2)
            )
            board.pop()

            moves.append((legal_move, legal_move_info["score"].relative))

        moves = sorted(moves, key=lambda x: x[1], reverse=True)
        for move in moves:
            print(f"Legal move: {move[0]}, score: {move[1]}")

        return moves

    def check_brilliant_move(self, chess_game, move, moves):
        """A move is brilliant if it is graded much higher than other possible moves."""

        BRILLIANCE_THRESHOLD = 500

        move_score = None
        best_move_score = moves[0][1]

        for m in moves:
            if m[0] == move:
                move_score = m[1]
                break

        if not move_score:
            return False

        if move_score.score() - best_move_score > BRILLIANCE_THRESHOLD:
            return True

        return False

    def check_gross_mistake(self, chess_game, move, moves):
        """A move is a gross mistake if it is graded much lower than other possible moves."""

        GROSS_MISTAKE_THRESHOLD = 500

        move_score = None
        best_move_score = moves[0][1]

        for m in moves:
            if m[0] == move:
                move_score = m[1]
                break

        if not move_score:
            return False

        if best_move_score - move_score.score() > GROSS_MISTAKE_THRESHOLD:
            return True

        return False

    def check_mistake(self, chess_game, move, moves):
        """A move is a mistake if it is graded lower than other possible moves."""

        MISTAKE_THRESHOLD = 200

        move_score = None
        best_move_score = moves[0][1]

        for m in moves:
            if m[0] == move:
                move_score = m[1]
                break

        if not move_score:
            return False

        if best_move_score - move_score.score() > MISTAKE_THRESHOLD:
            return True

        return False
