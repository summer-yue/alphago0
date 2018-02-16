import unittest
import numpy.testing as npt
import numpy as np
import random

from go import go_board as gb
from go import go_utils
from value_policy_net import resnet

def generate_fake_data(training_data_num, go_board_dimension):
    """Generate fake boards and counts the number of black and white stones as labels.
    Args:
        training_data_num: the number of fake training data we want to generate
    Returns:
        Xs: a list of training boards
        Ys: a list of training labels, each label is a size 2 array indicating the count for black and white stones
    """
    Xs = []
    Ys = []

    options = [-1, 0, 1] #white empty black
    for i in range(training_data_num):
        black_stone_count = 0
        white_stone_count = 0

        board = [[random.choice(options) for c in range(go_board_dimension)] for r in range(go_board_dimension)]
        for r in range(go_board_dimension):
            for c in range(go_board_dimension):
                if board[r][c] == -1:
                    white_stone_count += 1
                elif board[r][c] == 1:
                    black_stone_count += 1
        Xs.append(board)
        Ys.append([black_stone_count, white_stone_count])
    return Xs, Ys

class ResNetTest(unittest.TestCase):
    def test_gen_fake_data(self):
        Xs, Ys = generate_fake_data(training_data_num=100, go_board_dimension=5)
        self.assertEqual(len(Xs), len(Ys))
        # print(np.array(Xs).shape)
        # print(np.array(Ys).shape)

        # print(Xs[0])
        # print(Ys[0])

    def test_train_resnet(self):
        Xs, Ys = generate_fake_data(training_data_num=100, go_board_dimension=5)
        res = resnet.ResNet(go_board_dimension=5)

if __name__ == '__main__':
    unittest.main()
