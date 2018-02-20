import unittest
import numpy.testing as npt
import numpy as np
import random

from go import go_board as gb
from go import go_utils
from value_policy_net import resnet

class ResNetTest(unittest.TestCase):
    def test_train_resnet(self):
        res = resnet.ResNet(go_board_dimension=5)
        res.fake_train("../models")

if __name__ == '__main__':
    unittest.main()
