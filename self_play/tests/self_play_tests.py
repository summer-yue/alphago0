import unittest

from game.go_board import GoBoard
from game.go_utils import GoUtils
from self_play.self_play import SelfPlay
from value_policy_net.tests.uniform_prediction_net import UniformPredictionNet
from value_policy_net.tests.go_board_2x2_heuristics import GoBoard2Heuristics

BLACK = 1
WHITE = -1

class SelfPlayTest(unittest.TestCase):
    def test_play_one_move(self):
        board = GoBoard(board_dimension=2, player=BLACK)
        nn = GoBoard2Heuristics()

        utils = GoUtils()
        self_play_instance = SelfPlay(board, nn, utils)
        self_play_instance.play_one_move()
        self_play_instance.play_till_finish()
        print(self_play_instance.history_boards)

if __name__ == '__main__':
    unittest.main()