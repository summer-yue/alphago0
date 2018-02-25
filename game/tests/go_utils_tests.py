import unittest
import numpy.testing as npt
import math

from game.go_utils import GoUtils
from game.go_board import GoBoard

class GoUtilsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GoUtilsTest, self).__init__(*args, **kwargs)
        self.utils = GoUtils()

    def test_is_move_in_board_valid_move(self):
        self.assertTrue(GoUtils._is_move_in_board((1, 4), 9))
        self.assertTrue(GoUtils._is_move_in_board((0, 8), 9))

    def test_is_move_in_board_invalid_moves(self):
        self.assertFalse(GoUtils._is_move_in_board((-1, 1), 3))
        self.assertFalse(GoUtils._is_move_in_board((4, 9), 9))

    def test_make_move_invalid_not_in_board(self):
        move = (-1,1)
        board = GoBoard(board_dimension=9, player= 1, board_grid = None, game_history = None)
        self.assertEqual(self.utils.make_move(board, move), (False, board))

    def test_make_move_invalid_on_another_stone_no_capture(self):
        move = (0, 1)
        board_grid = [[ 0, 1, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 0, 1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertEqual(self.utils.make_move(board, move), (False, board))

    def test_make_move_invalid_move_into_an_eye(self):
        move = (3, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [-1, 0, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertEqual(self.utils.make_move(board, move), (False, board))

    def test_make_move_invalid_move_into_an_eye_2(self):
        move = (1, 0)
        board_grid = [[ 1, 0, 0, 0],
                      [ 0, 1, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 0, 0), (-1, -1, -1), ( 1, 1, 1), (-1, -1, -1), ( 1, 2, 0)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertEqual(self.utils.make_move(board, move), (False, board))

    def test_make_move_invalid_move_into_an_eye_3(self):
        move = (1, 0)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 0, 1, 0],
                      [ 0, 1, 0, 0],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 1, 0), (-1, -1, -1), ( 1, 0, 1), (-1, -1, -1), ( 1, 1, 2),
            (-1, -1, -1), ( 1, 2, 1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertEqual(self.utils.make_move(board, move), (False, board))

    def test_make_move_valid_move_capture_stone_1(self):
        move = (3, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [-1, 0, 0, 0],
                      [ 1,-1, 1, 0],
                      [ 0, 1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1), (-1, 2, 1), ( 1, 2, 2)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [-1, 0, 0, 0],
                          [ 0,-1, 1, 0],
                          [-1, 1, 0, 0]]
        new_game_history = game_history + [(-1, 3, 0)]
        new_board = GoBoard(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))

    def test_make_move_valid_move_capture_stone_2(self):
        move = (2, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 1, 0, 0],
                      [ 1,-1, 0, 0]]
        game_history = [( 1, 3, 0), (-1, 3, 1), ( 1, 2, 1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [ 0, 0, 0, 0],
                          [-1, 1, 0, 0],
                          [ 0,-1, 0, 0]]
        new_game_history = game_history + [(-1, 2, 0)]
        new_board = GoBoard(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))

    def test_make_move_valid_move_capture_stone_3(self):
        move = (2, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0, 0]]
        game_history = [( 1, 3, 0), (-1, 3, 1), ( 1, 2, 1), (-1, 2, 2), ( 1, 1, 0)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [ 1, 0, 0, 0],
                          [-1, 1,-1, 0],
                          [ 0,-1, 0, 0]]
        new_game_history = game_history + [(-1, 2, 0)]
        new_board = GoBoard(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))

    def test_make_move_valid_move_capture_stone_4(self):
        move = (3, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 1,-1],
                      [ 0, 0, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 2, 1), ( 1, 1, 1), (-1, 1, 2), ( 1, 2, 2),
            (-1, 2, 3), ( 1, -1, -1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 0, 0, 0],
                          [ 0, 1,-1, 0],
                          [ 1,-1, 0,-1],
                          [ 0, 0,-1, 0]]
        new_game_history = game_history + [(-1, 3, 2)]
        new_board = GoBoard(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))

    def test_make_move_valid_move_capture_stone_5(self):
        move = (0, 0)
        board_grid = [[ 0, 1,-1, 0],
                      [ 1,-1, 0, 0],
                      [ 1, 1,-1, 0],
                      [ 1,-1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 2, 2), ( 1, 2, 1), (-1, 1, 1), ( 1, 1, 0),
            (-1, 3, 1), ( 1, 3, 0), (-1, 0, 2), ( 1, 0, 1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[-1, 0,-1, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0,-1, 0, 0]]
        new_game_history = game_history + [(-1, 0, 0)]
        new_board = GoBoard(board_dimension=4, player=1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))

    def test_make_move_valid_move_pass(self):
        move = (-1, -1)
        board_grid = [[ 0, 0, 0, 0],
                      [-1, 0, 0, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1)]
        board = GoBoard(board_dimension=4, player=-1, board_grid = board_grid, game_history = game_history)

        new_board_grid = board_grid
        new_game_history = [( 1, 2, 0), (-1, 1, 0), ( 1, 3, 1), (-1, -1, -1)]
        new_board = GoBoard(board_dimension=4, player= 1, board_grid = new_board_grid, game_history = new_game_history)
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))

    def test_make_move_valid_move_no_ko(self):
        move = (1, 2)
        board_grid = [[ 0, 1,-1, 1],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0],
                      [ 0, 0, 0, 0]]
        game_history = [(1, 1, 0), (-1, 0, 2), (1, 0, 1), (-1, 2, 2), (1, 2, 1), (-1, 1, 3), (1, 0, 3), (-1, 1, 1)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)

        new_board_grid = [[ 0, 1, 0, 1],
                          [ 1, 0, 1,-1],
                          [ 0, 1,-1, 0],
                          [ 0, 0, 0, 0]]
        new_game_history = [(1, 1, 0), (-1, 0, 2), (1, 0, 1), (-1, 2, 2), (1, 2, 1), (-1, 1, 3), (1, 0, 3), (-1, 1, 1), (1, 1, 2)]
        new_board = GoBoard(board_dimension=4, player=-1, board_grid = new_board_grid, game_history = new_game_history)
 
        self.assertEqual(self.utils.make_move(board, move), (True, new_board))
        self.assertNotEqual(new_board, board)

    def test_make_move_invalid_move_ko(self):
        move = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0]]
        game_history = [(1, 2, 0), (-1, 2, 3), (1, 1, 1), (-1, 1, 2), (1, 2, 2), (-1, 3, 2), (1, 3, 1), (-1, 2, 1)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)

        self.assertEqual(self.utils.make_move(board, move), (False, board))

    def test_find_adjacent_positions_with_same_color_1(self):
        position = (1, 1)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 1, 1, 0],
                      [ 1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(0, 1), (1, 0), (1, 2)}
        self.assertEqual(GoUtils._find_adjacent_positions_with_same_color(position=position, board_grid=board_grid),
            expected_solution)

    def test_find_adjacent_positions_with_same_color_2(self):
        position = (0, 0)
        board_grid = [[ 1, 1, 0, 0],
                      [ 1, 1, 1, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(1, 0), (0, 1)}
        self.assertEqual(GoUtils._find_adjacent_positions_with_same_color(position=position, board_grid=board_grid),
            expected_solution)

    def test_find_adjacent_positions_with_same_color_3(self):
        position = (1, 1)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 1,-1, 0],
                      [ 0,-1, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(0, 1), (1, 0)}
        self.assertEqual(GoUtils._find_adjacent_positions_with_same_color(position=position, board_grid=board_grid),
            expected_solution)

    def test_find_pieces_in_group_1(self):
        position = (1, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 1,-1, 0, 0],
                      [-1, 0, 0, 0],
                      [ 0, 1, 0, 0]]
        expected_solution = {(1, 0)}
        self.assertEqual(GoUtils._find_pieces_in_group(position, board_grid), expected_solution)

    def test_find_pieces_in_group_2(self):
        position = (1, 0)
        board_grid = [[ 0, 0, 0, 0],
                      [ 1, 1, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 0,-1, 1, 0]]
        expected_solution = {(1, 0), (1, 1), (2, 1)}
        self.assertEqual(GoUtils._find_pieces_in_group(position, board_grid), expected_solution)

    def test_find_pieces_in_group_3(self):
        position = (1, 0)
        board_grid = [[ 0, 1, 0, 0],
                      [ 1, 1, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0, 0]]
        expected_solution = {(1, 0), (0, 1), (1, 1), (2, 1)}
        self.assertEqual(GoUtils._find_pieces_in_group(position, board_grid), expected_solution)

    def test_count_liberty_1(self):
        position1 = (0, 0)
        position2 = (1, 1)
        board_grid = [[ 1, 0, 0, 0],
                      [-1, 1, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 0, 0, 0, 0]]
        self.assertEqual(GoUtils._count_liberty(board_grid, position1), 1)
        self.assertEqual(GoUtils._count_liberty(board_grid, position2), 3)

    def test_count_liberty_2(self):
        position1 = (2, 2)
        position2 = (2, 1)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0,-1, 0, 0],
                      [-1, 1, 1, 0],
                      [ 0,-1, 1, 0]]
        self.assertEqual(GoUtils._count_liberty(board_grid, position1), 3)
        self.assertEqual(GoUtils._count_liberty(board_grid, position2), 3)

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
        self.assertEqual(GoUtils._remove_pieces_if_no_liberty(position, board_grid), expected_solution)

    def test_remove_pieces_if_no_liberty_2(self):
        position = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0,-1,-1, 0],
                      [-1, 1, 1, 0],
                      [ 0,-1, 1,-1]]
        expected_solution = board_grid
        self.assertEqual(GoUtils._remove_pieces_if_no_liberty(position, board_grid), expected_solution)

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
        self.assertEqual(GoUtils._remove_pieces_if_no_liberty(position, board_grid), expected_solution)

    def test_is_invalid_move_because_of_ko1_ko_corner(self):
        move = (1, 0)
        board_grid = [[-1, 1, 0, 0],
                      [ 0,-1, 0, 0],
                      [-1, 0, 0, 0],
                      [ 0, 0, 0, 1]]
        game_history = [(1, 0, 1), (-1, 1, 1), (1, 1, 0), (-1, 2, 0), (1, 3, 3), (-1, 0, 0)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)

        self.assertTrue(GoUtils._is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko2_ko_side(self):
        move = (1, 0)
        board_grid = [[-1, 0, 0, 0],
                      [ 0,-1, 0, 0],
                      [-1, 1, 0, 0],
                      [ 1, 0, 0, 0]]
        game_history = [(1, 1, 0), (-1, 0, 0), (1, 2, 1), (-1, 1, 1), (1, 3, 0), (-1, 2, 0)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(GoUtils._is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko3_ko_center(self):
        move = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0]]
        game_history = [(1, 2, 0), (-1, 2, 3), (1, 1, 1), (-1, 1, 2), (1, 2, 2), (-1, 3, 2), (1, 3, 1), (-1, 2, 1)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertTrue(GoUtils._is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko4_not_ko_corner(self):
        #Current move is not surrounded by opponents' stones
        move = (0, 1)
        board_grid = [[-1, 0, 0, 0],
                      [ 1,-1, 0, 0],
                      [ 0, 0, 0, 0],
                      [ 1, 0, 0, 0]]
        game_history = [(1, 3, 0), (-1, 0, 0), (1, 1, 0), (-1, 1, 1)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(GoUtils._is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko5_not_ko_center(self):
        #Current move captures two adjacent groups 
        move = (1, 2)
        board_grid = [[ 0, 1,-1, 1],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0],
                      [ 0, 0, 0, 0]]
        game_history = [(1, 1, 0), (-1, 0, 2), (1, 0, 1), (-1, 2, 2), (1, 2, 1), (-1, 1, 3), (1, 0, 3), (-1, 1, 1)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(GoUtils._is_invalid_move_because_of_ko(board, move))

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
        board = GoBoard(board_dimension=5, player=-1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(GoUtils._is_invalid_move_because_of_ko(board, move))

    def test_is_invalid_move_because_of_ko7_not_ko_center(self):
        #stone with no liberty from 2's position was not played in the last move
        move = (2, 2)
        board_grid = [[ 0, 0, 0, 0],
                      [ 0, 1,-1, 0],
                      [ 1,-1, 0,-1],
                      [ 0, 1,-1, 0]]
        game_history = [(1, 1, 1), (-1, 1, 2), (1, 2, 2), (-1, 2, 3), (1, 3, 1), (-1, 3, 2), (1, 2, 0), (-1, 2, 1),
                        (1, -1, -1), (-1, -1, -1)]
        board = GoBoard(board_dimension=4, player=1, board_grid = board_grid, game_history = game_history)
        self.assertFalse(GoUtils._is_invalid_move_because_of_ko(board, move))

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
        
        self.assertTrue(piece1 in GoUtils._find_connected_empty_pieces(board_grid))
        self.assertTrue(piece2 in GoUtils._find_connected_empty_pieces(board_grid))
        self.assertTrue(piece3 in GoUtils._find_connected_empty_pieces(board_grid))
        self.assertTrue(piece4 in GoUtils._find_connected_empty_pieces(board_grid))
        self.assertEqual(len(GoUtils._find_connected_empty_pieces(board_grid)), 4)

    def test_find_connected_empty_pieces_2(self):
        board_grid = [ [0, 1,-1, 0, 0],
                       [1, 1, 0,-1, 0],
                       [0, 1, 0,-1,-1],
                       [0, 1,-1, 0, 0],
                       [1,-1, 0, 0, 0]]

        piece1 = (1, [(0, 0)])
        piece2 = (1, [(2, 0), (3, 0)])
        piece3 = (-1, [(0, 3), (0, 4), (1, 4)])
        piece4 = (-1, [(3, 3), (3, 4), (4, 3), (4, 4), (4, 2)])
        piece5 = (0, [(1, 2), (2, 2)])
        
        connected_pieces = GoUtils._find_connected_empty_pieces(board_grid)
        self.assertTrue(piece1 in connected_pieces)
        self.assertTrue(piece2 in connected_pieces)
        self.assertTrue(piece3 in connected_pieces)
        self.assertTrue(piece4 in connected_pieces)
        self.assertTrue(piece5 in connected_pieces)
        self.assertEqual(len(GoUtils._find_connected_empty_pieces(board_grid)), 5)

    def test_evaluate_winner_1(self):
        board_grid = [ [0, 1,-1, 0, 0],
                       [1, 0, 1,-1, 0],
                       [0, 0, 1,-1,-1],
                       [0, 1,-1, 0, 0],
                       [1,-1, 0, 0, 0]]
        self.assertEqual(self.utils.evaluate_winner(board_grid), (-1, 3)) #white 14 black 11

    def test_evaluate_winner_2(self):
        board_grid = [ [0, 1,-1, 0, 0],
                       [1, 1, 0,-1, 0],
                       [0, 1, 0,-1,-1],
                       [0, 1,-1, 0, 0],
                       [1,-1, 0, 0, 0]]
        self.assertEqual(self.utils.evaluate_winner(board_grid), (-1, 5)) #white 14 black 9

    def test_evaluate_winner_3(self):
        board_grid = [ [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
        self.assertEqual(self.utils.evaluate_winner(board_grid), (-1, 0)) #white 14 black 11

if __name__ == '__main__':
    unittest.main()
