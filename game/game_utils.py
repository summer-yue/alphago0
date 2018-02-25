from collections import deque
from abc import ABC, abstractmethod

class GameUtils(ABC):
    """The general class for a grid based two player game utilities.
    The games in this category makes moves by specifying a row and column number.
    The current board contains the current player and other board infomration defined in game_board.
    The utilities are static functions called by GameUtils.function_name()
    """
    def __init__(self):
        pass
        
    @abstractmethod
    def make_move(self, board, move):
        """Make a move (row,col) on a game board
        Args:
            board: current board as a game_board object
            move: (r, c) tuple indicating the position of the considered move
        Returns:
            A tuple indicating board config and if the move was valid (boolean value)
            new board config if the move was successfully placed
            old config if board is not updated
        """
        pass

    @abstractmethod
    def is_valid_move(self, board, move):
        """Check if a potential move for the game is valid.
        Args:
            board: current board as a game_board object
            move: (r, c) tuple indicating the position of the considered move
        Returns:
            boolean variable indicating if the move is valid.
        """
        pass

    @abstractmethod
    def evaluate_winner(self, board_grid):
        """Evaluate who is the winner of the board configuration
        Args:
            board_grid: 2d array representation of the board
        Returns:
            player who won the game. 1: black or -1: white
            Absolute difference in scores
        """
        pass

    @abstractmethod
    def is_game_finished(self, board):
        """Check if the game is finished by looking at its game history
        Args:
            board: current board as a game_board object
        Returns:
            Boolean variable indicating if the game is finished
        """
        pass


