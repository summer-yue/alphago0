import copy 

from game.game_board import GameBoard

class TicTacToeBoard(GameBoard):
    def __init__(self, player=1, board_grid = [], game_history = None):
        """Initialize a tic tac toe board
        Args:
            player: 1 (player who plays first) or -1 indicating the two players
            board_grid: the original grid of the board, 2d array of 1 - black,
                -1: white and 0: not occupied
            game_history: the original order in which player played the game, a list of move tuples
                (player, r, c) such as (-1, 4, 6), or (1, -1, -1) means black passes a move
        """
        super(TicTacToeBoard, self).__init__(board_dimension=3, player=player, board_grid = board_grid, game_history = game_history)

    def __str__(self):
        """Define a more human friendly print for tic tac toe boards"""
        return str(self.board_dimension) + "x" + str(self.board_dimension) + " tic tac toe board\n" \
            + "with current player " + str(self.player) + "\n with current grid" \
            + str(self.board_grid) + "\n with game history" + str(self.game_history)