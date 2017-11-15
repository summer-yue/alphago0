import unittest
import numpy.testing as npt
import math

import go_board as gb
from go_utils_terminal import *

class GoUtilsTerminalTest(unittest.TestCase):
    def test_find_connected_empty_pieces_1(self):
        board_grid = [ [0, 1,-1, 0, 0],
                       [1, 0, 1,-1, 0],
                       [0, 0, 1,-1,-1],
                       [0, 1,-1, 0, 0],
                       [1,-1, 0, 0, 0]]

        piece1 = (1, [(0, 0)])
        piece2 = (1, [(1, 1), (2, 1), (2, 0), (3, 0)])
        piece3 = (-1, [(0, 3), (0, 4), (1, 4)])
        piece4 = (-1, [(3, 3), (3, 4), (4, 3), (4, 4), (4, 2)])
        
        self.assertTrue(piece1 in find_connected_empty_pieces(board_grid))
        self.assertTrue(piece2 in find_connected_empty_pieces(board_grid))
        self.assertTrue(piece3 in find_connected_empty_pieces(board_grid))
        self.assertTrue(piece4 in find_connected_empty_pieces(board_grid))
        self.assertEqual(len(find_connected_empty_pieces(board_grid)), 4)

if __name__ == '__main__':
    unittest.main()