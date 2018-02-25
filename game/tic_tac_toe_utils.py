from collections import deque

from game.game_utils import GameUtils
from game.tic_tac_toe_board import TicTacToeBoard

class TicTacToeUtils(GameUtils):
    """The utility class for tic tac toe games.
    The current board contains the current player and other board infomration defined in TicTacToeBoard.
    The utilities are static functions called by TicTacToeUtils.function_name()
    """
    def __init__(self):
        pass
        
    def is_valid_move(self, board, move):
        """Check if a potential move for the game is valid.
        Args:
            board: current board as a TicTacToeBoard object
            move: (r, c) tuple indicating the position of the considered move
        Returns:
            boolean variable indicating if the move is valid.
        """
        if move == (-1, -1):
            return True

        board_dimension = board.board_dimension
        (r,c) = move
        if r < 0 or c < 0 or r >= board_dimension or c >= board_dimension:
            return False

        if board.board_grid[r][c] != 0:
            return False
        return True

    def make_move(self, board, move):
        """Make a move (row,col) on a Tic Tac Toe Board
        Args:
            board: current board as a TicTacToeBoard object
            move: (r, c) tuple indicating the position of the considered move
        Returns:
            A tuple indicating if the move was valid (boolean value) and board config
            new board config if the move was successfully placed
            old config if board is not updated
        """
        if self.is_valid_move(board, move):
            board_copy = board.copy()
            (r, c) = move
            board_copy.board_grid[r][c] = board.player
            board_copy.add_move_to_history(r, c)
            board_copy.flip_player()
            return True, board_copy
        else:
            return False, board_copy

    def evaluate_winner(self, board_grid):
        """Evaluate who is the winner of the board configuration
        Args:
            board_grid: 2d array representation of the board
        Returns:
            player who won the game. 1: black or -1: white
        """
        # Evaluate score for each of the 8 lines (3 rows, 3 columns, 2 diagonals)
        black_wins = False
        white_wins = False
        score1 = self._evaluateLine(board_grid, 0, 0, 0, 1, 0, 2)  # row 0
        score2 = self._evaluateLine(board_grid, 1, 0, 1, 1, 1, 2)  # row 1
        score3 = self._evaluateLine(board_grid, 2, 0, 2, 1, 2, 2)  # row 2
        score4 = self._evaluateLine(board_grid, 0, 0, 1, 0, 2, 0)  # col 0
        score5 = self._evaluateLine(board_grid, 0, 1, 1, 1, 2, 1)  # col 1
        score6 = self._evaluateLine(board_grid, 0, 2, 1, 2, 2, 2)  # col 2
        score7 = self._evaluateLine(board_grid, 0, 0, 1, 1, 2, 2)  # diagonal
        score8 = self._evaluateLine(board_grid, 0, 2, 1, 1, 2, 0)  # alternate diagonal

        if self._almost_equal(score1, 1) or self._almost_equal(score2, 1) or self._almost_equal(score3, 1) \
            or self._almost_equal(score4, 1) or self._almost_equal(score5, 1) or self._almost_equal(score6, 1) \
            or self._almost_equal(score7, 1) or self._almost_equal(score8, 1):
                black_wins = True

        if self._almost_equal(score1, -1) or self._almost_equal(score2, -1) or self._almost_equal(score3, -1) \
            or self._almost_equal(score4, -1) or self._almost_equal(score5, -1) or self._almost_equal(score6, -1) \
            or self._almost_equal(score7, -1) or self._almost_equal(score8, -1):
                white_wins = True
                
        assert (black_wins and white_wins) == False
        if black_wins:
            return 1
        elif white_wins:
            return -1
        else:
            return 0

    def is_game_finished(self, board):
        """Check if the tic tac toe game is finished by looking at its game history
        The game is finished if the last two actions were both pass
        Args:
            board: current board as a go_board object
        Returns:
            Boolean variable indicating if the game is finished
        """
        if len(board.game_history) < 2:
            return False

        (_, r1, c1) = board.game_history[-1]
        (_, r2, c2) = board.game_history[-2]
        last_move = (r1, c1)
        second_to_last_move = (r2, c2)

        double_passed = (last_move == (-1, -1) and second_to_last_move == (-1, -1))
        someone_won = not self._almost_equal(self.evaluate_winner(board.board_grid), 0)
        return double_passed or someone_won

    def _evaluateLine(self, board_grid, r1, c1,r2, c2, r3, c3):
        """Evaluate if a line forms a black win, white win, or no win
        Args:
            board_grid: 2d array representation of the board
            row and col indices for the three cells we refer to
        Returns:
            1 indicating the line causes black to win, -1 if it causes white to win, 0 otherwise
        """
        if abs(board_grid[r1][c1] + board_grid[r2][c2] + board_grid[r3][c3] - 3) < 1e-3:
            return 1
        elif abs(board_grid[r1][c1] + board_grid[r2][c2] + board_grid[r3][c3] + 3) < 1e-3:
            return -1
        else:
            return 0

    def _almost_equal(self, a, b):
        return abs(a - b) < 1e-3





