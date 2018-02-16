from self_play import self_play
from go import go_board
from value_policy_net.tests import uniform_prediction_net
import unittest

BLACK = 1
WHITE = -1

class SelfPlayTest(unittest.TestCase):
    def test_init_1(self):
        board = go_board.go_board(board_dimension=2, player=BLACK)
        nn = uniform_prediction_net.UniformPredictionNet()
        self_play_instance = self_play.self_play(board, nn)

    def test_play_one_move(self):
        board = go_board.go_board(board_dimension=2, player=BLACK)
        nn = uniform_prediction_net.UniformPredictionNet()
        self_play_instance = self_play.self_play(board, nn)
        self_play_instance.play_one_move()
        self_play_instance.play_till_finish()
        print(self_play_instance.history_boards)

if __name__ == '__main__':
    unittest.main()