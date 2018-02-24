import unittest
import numpy.testing as npt
import numpy as np
import random
import tensorflow as tf
from go import go_board

from go import go_board as gb
from go import go_utils
from value_policy_net import resnet

class ResNetTest(unittest.TestCase):
    def test_train_resnet(self):
        with tf.Session().as_default():
            res = resnet.ResNet(go_board_dimension=5)
            res.fake_train("../models_fake")

    # def test_convert_to_onehot(self):
    #     with tf.Session().as_default():
    #         res = resnet.ResNet(go_board_dimension=5)
    #         board = go_board.go_board(board_dimension=5, player=1)
    #         converted = res.convert_to_one_hot_go_boards(board.board_grid)
    #         print(np.array(converted).shape)
    #         ones = np.ones((5, 5, 1))
    #         append_result = np.append(np.array(converted), ones, axis = 2)
    #         print(append_result.shape)

if __name__ == '__main__':
    unittest.main()
