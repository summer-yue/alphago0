import copy 
import numpy as np

from game.game_board import GameBoard

class GoBoard(GameBoard):
    def __str__(self):
        """Define a more human friendly print for go boards"""
        return str(self.board_dimension) + "x" + str(self.board_dimension) + " go board\n" \
            + "with current player " + str(self.player) + "\n with current grid" \
            + str(self.board_grid) + "\n with game history " + str(self.game_history)
