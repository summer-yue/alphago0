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

    def test_make_move_invalid_not_in_board(self):
        move = (-1,1)
        board = gb.go_board(board_dimension=9, player= 1, board_grid = None, game_history = None)
        self.assertTrue(make_move(board, move) is None)

    def test_make_move_invalid_on_another_stone_no_capture(self):
        move = (0, 1)
        board_grid = [[ 0, 1, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 0, 1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(make_move(board, move) is None)

    def test_make_move_invalid_move_into_an_eye(self):
        move = (3, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [-1, 0, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(make_move(board, move) is None)

    def test_make_move_invalid_move_into_an_eye_2(self):
        move = (1, 0)
        board_grid = [[ 1, 0, 0, 0],
                      [ 0, 1, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 0, 0), (-1, -1, -1), ( 1, 1, 1), (-1, -1, -1), ( 1, 2, 0)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(make_move(board, move) is None)

    def test_make_move_invalid_move_into_an_eye_3(self):
        move = (1, 0)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 0, 1, 0],
                      [ 0, 1, 0, 0],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 1, 0), (-1, -1, -1), ( 1, 0, 1), (-1, -1, -1), ( 1, 1, 2),
            (-1, -1, -1), ( 1, 2, 1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(make_move(board, move) is None)

    def test_make_move_valid_move_capture_stone_1(self):
        move = (3, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [-1, 0, 0, 0],
                      [ 1,-1, 1, 0],
                      [ 0, 1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1), (-1, 2, 1), ( 1, 2, 2)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [-1, 0, 0, 0],
                          [ 0,-1, 1, 0],
                          [-1, 1, 0, 0]]
        new_game_history = game_history + [(-1, 3, 0)]
        new_board = gb.go_board(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(make_move(board, move), new_board)

    def test_make_move_valid_move_capture_stone_2(self):
        move = (2, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 1, 0, 0],
                      [ 1,-1, 0, 0]]
        game_history = [( 1, 3, 0), (-1, 3, 1), ( 1, 2, 1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [ 0, 0, 0, 0],
                          [-1, 1, 0, 0],
                          [ 0,-1, 0, 0]]
        new_game_history = game_history + [(-1, 2, 0)]
        new_board = gb.go_board(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(make_move(board, move), new_board)

    def test_make_move_valid_move_capture_stone_3(self):
        move = (2, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0, 0]]
        game_history = [( 1, 3, 0), (-1, 3, 1), ( 1, 2, 1), (-1, 2, 2), ( 1, 1, 0)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [ 1, 0, 0, 0],
                          [-1, 1,-1, 0],
                          [ 0,-1, 0, 0]]
        new_game_history = game_history + [(-1, 2, 0)]
        new_board = gb.go_board(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(make_move(board, move), new_board)

    def test_make_move_valid_move_capture_stone_4(self):
        move = (3, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 1,-1],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 2, 1), ( 1, 1, 1), (-1, 1, 2), ( 1, 2, 2),
            (-1, 2, 3), ( 1, -1, -1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [ 0, 1,-1, 0],
                          [ 1,-1, 0,-1],
                          [ 0, 0,-1, 0]]
        new_game_history = game_history + [(-1, 3, 2)]
        new_board = gb.go_board(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(make_move(board, move), new_board)

    def test_make_move_valid_move_capture_stone_5(self):
        move = (0, 0)
        board_grid = [[ 0, 1,-1, 0],
                      [ 1,-1, 0, 0],
                      [ 1, 1,-1, 0],
                      [ 1,-1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 2, 2), ( 1, 2, 1), (-1, 1, 1), ( 1, 1, 0),
            (-1, 3, 1), ( 1, 3, 0), (-1, 0, 2), ( 1, 0, 1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[-1, 0,-1, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0,-1, 0, 0]]
        new_game_history = game_history + [(-1, 0, 0)]
        new_board = gb.go_board(board_dimension=4, player=1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(make_move(board, move), new_board)

    def test_make_move_valid_move_pass(self):
        move = (-1, -1)
        board_grid = [[ 0, 0, 0, 0],
                      [-1, 0, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1)]
        board = gb.go_board(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = board_grid
        new_game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1), (-1, -1, -1)]
        new_board = gb.go_board(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(make_move(board, move), new_board)

    def test_find_adjacent_positions_with_same_color_1(self):
        position = (1, 1)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 1, 1, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(0, 1), (1, 0), (1, 2)}
        self.assertEqual(find_adjacent_positions_with_same_color(position=position, board_grid=board_grid),
            expected_solution)

    def test_find_adjacent_positions_with_same_color_2(self):
        position = (0, 0)
        board_grid = [[ 1, 1, 0, 0],
                      [ 1, 1, 1, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(1, 0), (0, 1)}
        self.assertEqual(find_adjacent_positions_with_same_color(position=position, board_grid=board_grid),
            expected_solution)

    def test_find_adjacent_positions_with_same_color_3(self):
        position = (1, 1)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 1,-1, 0],
                      [ 0,-1, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(0, 1), (1, 0)}
        self.assertEqual(find_adjacent_positions_with_same_color(position=position, board_grid=board_grid),
            expected_solution)

    def test_find_pieces_in_group_1(self):
        position = (1, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 1,-1, 0, 0],
                      [-1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(1, 0)}
        self.assertEqual(find_pieces_in_group(position, board_grid), expected_solution)

    def test_find_pieces_in_group_2(self):
        position = (1, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 1, 1, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 0,-1, 1, 0]]
        expected_solution = {(1, 0), (1, 1), (2, 1)}
        self.assertEqual(find_pieces_in_group(position, board_grid), expected_solution)

    def test_find_pieces_in_group_3(self):
        position = (1, 0)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 1, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0, 0]]
        expected_solution = {(1, 0), (0, 1), (1, 1), (2, 1)}
        self.assertEqual(find_pieces_in_group(position, board_grid), expected_solution)

    def test_count_liberty_1(self):
        position1 = (0, 0)
        position2 = (1, 1)
        board_grid = [[ 1, 0, 0, 0],
                      [-1, 1, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 0, 0, 0]]
        self.assertEqual(count_liberty(board_grid, position1), 1)
        self.assertEqual(count_liberty(board_grid, position2), 3)

    def test_count_liberty_2(self):
        position1 = (2, 2)
        position2 = (2, 1)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0,-1, 0, 0],
                      [-1, 1, 1, 0],
                      [ 0,-1, 1, 0]]
        self.assertEqual(count_liberty(board_grid, position1), 3)
        self.assertEqual(count_liberty(board_grid, position2), 3)

    def test_remove_pieces_if_no_liberty_1(self):
        position = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0,-1,-1, 0],
                      [-1, 1, 1,-1],
                      [ 0,-1, 1,-1]]
        expected_solution =  [[ 0, 0, 0, 0],
                              [ 0,-1,-1, 0],
                              [-1, 0, 0,-1],
                              [ 0,-1, 0,-1]]
        self.assertEqual(remove_pieces_if_no_liberty(position, board_grid), expected_solution)

    def test_remove_pieces_if_no_liberty_2(self):
        position = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0,-1,-1, 0],
                      [-1, 1, 1, 0],
                      [ 0,-1, 1,-1]]
        expected_solution = board_grid
        self.assertEqual(remove_pieces_if_no_liberty(position, board_grid), expected_solution)

    def test_remove_pieces_if_no_liberty_3(self):
        position = (3, 1)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 0, 0, 0],
                      [-1, 1, 1, 0],
                      [ 1,-1, 1, 0]]
        expected_solution =  [[ 0, 0, 0, 0],
                              [ 0, 0, 0, 0],
                              [-1, 1, 1, 0],
                              [ 1, 0, 1, 0]]
        self.assertEqual(remove_pieces_if_no_liberty(position, board_grid), expected_solution)

    def test_is_invalid_move_because_of_ko1_ko_corner(self):
        move = (1, 0)
        board_grid = [[-1, 1, 0, 0],
                      [ 0,-1, 0, 0],
                      [-1, 0, 0, 0],
                      [ 0, 0, 0, 1]]
        game_history = [(1, 0, 1), (-1, 1, 1), (1, 1, 0), (-1, 2, 0), (1, 3, 3), (-1, 0, 0)]
        board = gb.go_board(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)

        self.assertTrue(is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko2_ko_side(self):
        move = (1, 0)
        board_grid = [[-1, 0, 0, 0],
                      [ 0,-1, 0, 0],
                      [-1, 1, 0, 0],
                      [ 1, 0, 0, 0]]
        game_history = [(1, 1, 0), (-1, 0, 0), (1, 2, 1), (-1, 1, 1), (1, 3, 0), (-1, 2, 0)]
        board = gb.go_board(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko3_ko_center(self):
        move = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0]]
        game_history = [(1, 2, 0), (-1, 2, 3), (1, 1, 1), (-1, 1, 2), (1, 2, 2), (-1, 3, 2), (1, 3, 1), (-1, 2, 1)]
        board = gb.go_board(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko4_not_ko_corner(self):
        #Current move is not surrounded by opponents' stones
        move = (0, 1)
        board_grid = [[-1, 0, 0, 0],
                      [ 1,-1, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 1, 0, 0, 0]]
        game_history = [(1, 3, 0), (-1, 0, 0), (1, 1, 0), (-1, 1, 1)]
        board = gb.go_board(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko5_not_ko_center(self):
        #Current move captures two adjacent groups 
        move = (1, 2)
        board_grid = [[ 0, 1,-1, 1],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0],
                      [ 0, 0, 0, 0]]
        game_history = [(1, 1, 0), (-1, 0, 2), (1, 0, 1), (-1, 2, 2), (1, 2, 1), (-1, 1, 3), (1, 0, 3), (-1, 1, 1)]
        board = gb.go_board(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko6_not_ko_center(self):
        #Capture Two stones that are connected from the move
        move = (2, 1)
        board_grid = [[ 0, 0, 0, 0, 0],
                      [ 0, 1,-1,-1, 0],
                      [ 1, 0, 1, 1,-1],
                      [ 0, 1,-1,-1, 0],
                      [ 0, 0, 0, 0, 0]]
        game_history = [(1, 1, 1), (-1, 1, 2), (1, 2, 2), (-1, 1, 3), (1, 2, 3),
                        (-1, 2, 4), (1, -1, -1), (-1, 3, 3), (1, 3, 1), (-1, 3, 2),
                        (1, 2, 0)]
        board = gb.go_board(board_dimension=5, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko7_not_ko_center(self):
        #stone with no liberty from 2's position was not played in the last move
        move = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0]]
        game_history = [(1, 1, 1), (-1, 1, 2), (1, 2, 2), (-1, 2, 3), (1, 3, 1), (-1, 3, 2), (1, 2, 0), (-1, 2, 1),
                        (1, -1, -1), (-1, -1, -1)]
        board = gb.go_board(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(is_invalid_move_because_of_ko(board, move))

if __name__ == '__main__':
    unittest.main()
