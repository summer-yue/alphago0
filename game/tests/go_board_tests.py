import unittest
import numpy.testing as npt
import math
import numpy as np

from game.go_utils import GoUtils
from game.go_board import GoBoard

class GoBoardTest(unittest.TestCase):
    board_grid = [[ 0, 1, 0, 0],
                  [ 0, 0, 0, 0],
                  [ 0, 0, 0, 0],
                  [ 0, 0, 0, 0]]
    game_history = [( 1, 0, 1)]
    board1 = GoBoard(board_dimension=4, player=-1, board_grid = np.array(board_grid), game_history = game_history)
    # for augmented_board in board.generate_augmented_boards():
    #     print(augmented_board)
    #     print()

    history_boards = [board1, board1]
    print(np.array([augment_board for history_board in history_boards for augment_board in history_board.generate_augmented_boards()]))

if __name__ == '__main__':
    unittest.main()