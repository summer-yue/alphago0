import unittest
import numpy.testing as npt
import math

from game.tic_tac_toe_utils import TicTacToeUtils
from game.tic_tac_toe_board import TicTacToeBoard

class TicTacToeUtilsTest(unittest.TestCase):
    def test_is_valid_move(self):
        board = TicTacToeBoard() #A new tic tac toe board
        utils = TicTacToeUtils()
        self.assertTrue(utils.is_valid_move(board, (1, 1)))
        self.assertTrue(utils.is_valid_move(board, (0, 0)))
        self.assertTrue(utils.is_valid_move(board, (2, 2)))
        self.assertFalse(utils.is_valid_move(board, (0, 3)))
        self.assertFalse(utils.is_valid_move(board, (3, 2)))

    def test_move(self):
        board = TicTacToeBoard() #A new tic tac toe board
        utils = TicTacToeUtils()
        is_valid, new_board = utils.make_move(board, (1, 1))
        self.assertTrue(is_valid)
        self.assertFalse(utils.is_valid_move(new_board, (1, 1)))
        is_valid, _ = utils.make_move(board, (1, 2))
        self.assertTrue(is_valid)
        print(board)

    def test_evaluate_winner(self):
        utils = TicTacToeUtils()
        grid = [[-1, 0, 0], [1, -1, 0], [1, 0, 1]]
        history = [(1, 2, 0), (-1, 1, 1), (1, 1, 0), (-1, 0, 0), (1, 2, 2)]
        board = TicTacToeBoard(-1, grid, history)
        winner, _ = utils.evaluate_winner(grid)
        print("winner is", winner)
        self.assertTrue(abs(winner) < 1e-3)

if __name__ == '__main__':
    unittest.main()