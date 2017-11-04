import unittest
import numpy.testing as npt
import math

import go_board as gb
from go_utils import *

class GoUtilsTest(unittest.TestCase):
    def test_is_move_in_board_valid_move(self):
        self.assertTrue(is_move_in_board((1, 4), 9))
        self.assertTrue(is_move_in_board((0, 8), 9))

    def test_is_move_in_board_invalid_moves(self):
        self.assertFalse(is_move_in_board((-1,1), 3))
        self.assertFalse(is_move_in_board((4,9), 9))

    def test_is_valid_move_invalid_not_in_board(self):
        move = (-1,1)
        board = gb.go_board(board_dimension=9, player='b', board_grid = None, game_history = None)
        self.assertFalse(is_valid_move(board, move))

    def test_is_valid_move_invalid_on_another_stone_no_capture(self):
        move = (0, 1)
        board_grid = [['0','b','0','0'],
                      ['0','0','0','0'],
                      ['0','0','0','0'],
                      ['0','0','0','0']]
        game_history = [('b', 0, 1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_valid_move(board, move))

    def test_is_valid_move_invalid_move_into_an_eye(self):
        move = (3, 0)
        board_grid = [['0','0','0','0'],
                      ['w','0','0','0'],
                      ['b','0','0','0'],
                      ['0','b','0','0']]
        game_history = [('b', 2, 0), ('w', 1, 0), ('b', 3, 1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_valid_move(board, move))

    def test_is_valid_move_invalid_move_into_an_eye_2(self):
        move = (1, 0)
        board_grid = [['b','0','0','0'],
                      ['0','b','0','0'],
                      ['b','0','0','0'],
                      ['0','0','0','0']]
        game_history = [('b', 0, 0), ('w', -1, -1), ('b', 1, 1), ('w', -1, -1), ('b', 2, 0)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_valid_move(board, move))

    def test_is_valid_move_invalid_move_into_an_eye_3(self):
        move = (1, 0)
        board_grid = [['0','b','0','0'],
                      ['b','0','b','0'],
                      ['0','b','0','0'],
                      ['0','0','0','0']]
        game_history = [('b', 1, 0), ('w', -1, -1), ('b', 0, 1), ('w', -1, -1), ('b', 1, 2),
            ('w', -1, -1), ('b', 2, 1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_valid_move(board, move))

    def test_is_valid_move_valid_move_capture_stone_1(self):
        move = (3, 0)
        board_grid = [['0','0','0','0'],
                      ['w','0','0','0'],
                      ['b','w','b','0'],
                      ['0','b','0','0']]
        game_history = [('b', 2, 0), ('w', 1, 0), ('b', 3, 1), ('w', 2, 1), ('b', 2, 2)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_valid_move(board, move))

    def test_is_valid_move_valid_move_capture_stone_2(self):
        move = (2, 0)
        board_grid = [['0','0','0','0'],
                      ['0','0','0','0'],
                      ['0','b','0','0'],
                      ['b','w','0','0']]
        game_history = [('b', 3, 0), ('w', 3, 1), ('b', 2, 1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_valid_move(board, move))

    def test_is_valid_move_valid_move_capture_stone_3(self):
        move = (2, 0)
        board_grid = [['0','0','0','0'],
                      ['b','0','0','0'],
                      ['0','b','w','0'],
                      ['b','w','0','0']]
        game_history = [('b', 3, 0), ('w', 3, 1), ('b', 2, 1), ('w', 2, 2), ('b', 1, 0)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_valid_move(board, move))

    def test_is_valid_move_valid_move_capture_stone_4(self):
        move = (3, 2)
        board_grid = [['0','0','0','0'],
                      ['0','b','w','0'],
                      ['b','w','b','w'],
                      ['0','0','0','0']]
        game_history = [('b', 2, 0), ('w', 2, 1), ('b', 1, 1), ('w', 1, 2), ('b', 2, 2),
            ('w', 2, 3), ('b', -1, -1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_valid_move(board, move))

    def test_is_valid_move_valid_move_capture_stone_5(self):
        move = (0, 0)
        board_grid = [['0','b','w','0'],
                      ['b','w','0','0'],
                      ['b','b','w','0'],
                      ['b','w','0','0']]
        game_history = [('b', 2, 0), ('w', 2, 2), ('b', 2, 1), ('w', 1, 1), ('b', 1, 0),
            ('w', 3, 1), ('b', 3, 0), ('w', 0, 2), ('b', 0, 1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_valid_move(board, move))

    def test_is_valid_move_valid_move_pass(self):
        move = (-1, -1)
        board_grid = [['0','0','0','0'],
                      ['w','0','0','0'],
                      ['b','0','0','0'],
                      ['0','b','0','0']]
        game_history = [('b', 2, 0), ('w', 1, 0), ('b', 3, 1)]
        board = gb.go_board(board_dimension=4, player='w', board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_valid_move(board, move))

if __name__ == '__main__':
    unittest.main()
