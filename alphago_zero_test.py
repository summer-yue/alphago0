import self_play
import go_board
import alphago_zero
import unittest

BLACK = 1
WHITE = -1

class AlphaGo_Zero_Test(unittest.TestCase):
    def test_init_1(self):
        ag0 = alphago_zero.AlphaGo_Zero()
        
if __name__ == '__main__':
    unittest.main()