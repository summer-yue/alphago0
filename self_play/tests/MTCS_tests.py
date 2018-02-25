import unittest
import tensorflow as tf

from game.go_board import GoBoard
from game.go_utils import GoUtils
from self_play.mcts import MCTS
from value_policy_net.tests.uniform_prediction_net import UniformPredictionNet
from value_policy_net.resnet import ResNet
from value_policy_net.tests.go_board_2x2_heuristics import GoBoard2Heuristics

BLACK = 1
WHITE = -1

class MCTSTest(unittest.TestCase):
    def test_init_1(self):
        board = GoBoard(board_dimension=3, player=BLACK)
        nn = UniformPredictionNet()
        utils = GoUtils()
        mcts_instance = MCTS(board, nn, utils, simluation_number = 1000)

    def test_select_edge_1(self):
        """ For graph with just a root node, select edge returns None
        """
        board = GoBoard(board_dimension=2, player=BLACK)
        nn = UniformPredictionNet()
        utils = GoUtils()
        mcts_instance = MCTS(board, nn, utils, simluation_number = 1000, random_seed=2)
        self.assertEqual(mcts_instance.select_edge(mcts_instance.root_node, "max"), None)

    def test_run_one_simluation_1(self):
        """ Run one simulation on graph with just one root node
        """
        board = GoBoard(board_dimension=2, player=BLACK)
        utils = GoUtils()
        self.sess = tf.Session()
        
        nn = GoBoard2Heuristics(board_dimension = 2)
        #with self.sess.as_default():
            #nn = ResNet(board_dimension = 2, model_path = "../models", restored=False)

        mcts_instance = MCTS(board, nn, utils, simluation_number = 1000, random_seed=3)
        new_board, move, policy = mcts_instance.run_all_simulations()
        print(move)

        mcts_instance2 = MCTS(new_board, nn, utils, simluation_number = 1000, random_seed=1)
        new_board2, move2, policy2 = mcts_instance2.run_all_simulations()
        print(move2)

        mcts_instance3 = MCTS(new_board2, nn, utils, simluation_number = 1000, random_seed=2)
        new_board3, move3, policy3 = mcts_instance3.run_all_simulations()
        print(move3)

        mcts_instance4 = MCTS(new_board3, nn, utils, simluation_number = 1000, random_seed=2)
        new_board4, move4, policy4 = mcts_instance4.run_all_simulations()
        print(move4)

        # Print all leaf nodes in order
        # stack = [mcts_instance.root_node]
        # while stack != []:
        #     current_node = stack.pop()
        #     if current_node.is_leaf():
        #         print(current_node.board)
        #         print(current_node.action_value)
        #         print("")
        #     else:
        #         for edge in current_node.edges:
        #             stack.append(edge.to_node)

        # # print all edges connected to leaf noodes
        # stack = [mcts_instance.root_node]
        # while stack != []:
        #     current_node = stack.pop()
        #     if current_node.is_leaf():
        #         print(current_node.parent_edge)
        #         print("")
        #     else:
        #         for edge in current_node.edges:
        #             stack.append(edge.to_node)

        # print(mcts_instance.root_node.action_value)

if __name__ == '__main__':
    unittest.main()