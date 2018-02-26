import unittest
import numpy.testing as npt
import numpy as np
import random
import tensorflow as tf
from game.go_board import GoBoard

from game.go_utils import GoUtils
from value_policy_net.resnet import ResNet

class ResNetTest(unittest.TestCase):
    def test_train_resnet(self):
        with tf.Session().as_default():
            res = ResNet(board_dimension=5, l2_beta=0.01)
            res.fake_train("../models_fake")

    # def test_convert_to_onehot(self):
    #     with tf.Session().as_default():
    #         res = ResNet(board_dimension=5)
    #         board = GoBoard(board_dimension=5, player=1)
    #         converted = res.convert_to_one_hot_boards(board.board_grid)
    #         print(np.array(converted).shape)
    #         ones = np.ones((5, 5, 1))
    #         append_result = np.append(np.array(converted), ones, axis = 2)
    #         print(append_result.shape)

if __name__ == '__main__':
    unittest.main()
