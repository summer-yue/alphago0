import self_play
from go import go_board
from value_policy_net import resnet
import unittest

BLACK = 1
WHITE = -1

class ResNetTest(unittest.TestCase):
    def test_init_1(self):
        ag0 = resnet.ResNet()
        
if __name__ == '__main__':
    unittest.main()