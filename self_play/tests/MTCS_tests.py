import unittest
import tensorflow as tf

from game.go_board import GoBoard
from game.go_utils import GoUtils
from game.tic_tac_toe_board import TicTacToeBoard
from game.tic_tac_toe_utils import TicTacToeUtils
from self_play.mcts import MCTS
from value_policy_net.tests.uniform_prediction_net import UniformPredictionNet
from value_policy_net.resnet import ResNet
from value_policy_net.tests.go_board_2x2_heuristics import GoBoard2Heuristics

BLACK = 1
WHITE = -1

class MCTSTest(unittest.TestCase):
    def test_solve_tic_tac_toe(self):
        """ Run one simulation on graph with just one root node
        """
        board = TicTacToeBoard()
        utils = TicTacToeUtils() 
        nn = UniformPredictionNet(board_dimension = 3)

        for i in range(10):
            mcts_instance = MCTS(board, nn, utils, simluation_number = 10000, random_seed=2)
            board, move, policy = mcts_instance.run_all_simulations(temp1 = 0.2, temp2 = 0.1, step_boundary=2)
            if utils.is_game_finished(board):
                print("winner is : {}".format(utils.evaluate_winner(board.board_grid)))
            print("move {} is {}".format(i, move))

    def test_solve_tic_tac_toe_2(self):
        """ Run one simulation on graph with just one root node
        """
        grid = [[1, 0, -1], [0, 0, 0], [1, 0, 0]]
        history = [(1, 0, 0), (-1, 0, 2), (1, 2, 0)]

        # X1O
        # 111
        # X11

        board = TicTacToeBoard(player=-1, board_grid = grid, game_history = history)
        utils = TicTacToeUtils() 
        nn = UniformPredictionNet(board_dimension = 3)

        mcts_instance = MCTS(board, nn, utils, simluation_number = 1000, random_seed=2)
        board, move, policy = mcts_instance.run_all_simulations(temp1 = 0.2, temp2 = 0.1, step_boundary=2)
        self.assertEqual(move, (1, 0))
        
        print("board afer move is {} is with policy {}".format(board, policy))

if __name__ == '__main__':
    unittest.main()