import mcts
from value_policy_net.tests import uniform_prediction_net
from go import go_board
from self_play import mcts
import unittest

BLACK = 1
WHITE = -1

class MCTSTest(unittest.TestCase):
    def test_init_1(self):
        board = go_board.go_board(board_dimension=3, player=BLACK)
        nn = uniform_prediction_net.UniformPredictionNet()
        mcts_instance = mcts.MCTS(board, nn, simluation_number = 1000)
        #print(str(mcts_instance.root_node))

    def test_select_edge_1(self):
        """ For graph with just a root node, select edge returns None
        """
        board = go_board.go_board(board_dimension=2, player=BLACK)
        nn = uniform_prediction_net.UniformPredictionNet()
        mcts_instance = mcts.MCTS(board, nn, simluation_number = 1000)
        self.assertEqual(mcts_instance.select_edge(mcts_instance.root_node), None)

    def test_run_one_simluation_1(self):
        """ Run one simulation on graph with just one root node
        """
        board = go_board.go_board(board_dimension=2, player=BLACK)
        nn = uniform_prediction_net.UniformPredictionNet()
        mcts_instance = mcts.MCTS(board, nn, simluation_number = 1000)

        # mcts_instance.run_one_simluation()
        # mcts_instance.run_one_simluation()
        # mcts_instance.run_one_simluation()
        # mcts_instance.run_one_simluation()
        # mcts_instance.run_one_simluation()
        print(mcts_instance.run_all_simulations())
        print("FINISHED RUNNING 1000 SIMULATIONS")
        #print(str(mcts_instance.root_node))
        #print("\n")
        
        # Print all leaf nodes in order
        stack = [mcts_instance.root_node]
        while stack != []:
            current_node = stack.pop()
            if current_node.is_leaf():
                print(current_node.go_board)
                print(current_node.action_value)
                print("")
            else:
                for edge in current_node.edges:
                    stack.append(edge.to_node)

        # print all edges connected to leaf noodes
        stack = [mcts_instance.root_node]
        while stack != []:
            current_node = stack.pop()
            if current_node.is_leaf():
                print(current_node.parent_edge)
                print("")
            else:
                for edge in current_node.edges:
                    stack.append(edge.to_node)

        print(mcts_instance.root_node.action_value)

if __name__ == '__main__':
    unittest.main()