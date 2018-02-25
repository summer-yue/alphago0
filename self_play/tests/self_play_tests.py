import unittest

from game.go_board import GoBoard
from game.go_utils import GoUtils
from game.tic_tac_toe_board import TicTacToeBoard
from game.tic_tac_toe_utils import TicTacToeUtils
from self_play.self_play import SelfPlay
from value_policy_net.tests.uniform_prediction_net import UniformPredictionNet
from value_policy_net.tests.go_board_2x2_heuristics import GoBoard2Heuristics

BLACK = 1
WHITE = -1

class SelfPlayTest(unittest.TestCase):
    def test_play_one_move(self):
        board = TicTacToeBoard()
        utils = TicTacToeUtils() 
        nn = UniformPredictionNet(board_dimension = 3)

        self_play_instance = SelfPlay(board, nn, utils, simluation_number=500)
        self_play_instance.play_till_finish()
        print(self_play_instance.history_boards)

if __name__ == '__main__':
    unittest.main()