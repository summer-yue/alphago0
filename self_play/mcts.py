from self_play import edge
from self_play import node
import random
import time
import numpy as np

from pyprind import prog_bar
from go import go_utils
from math import sqrt

from value_policy_net.tests import go_board_2x2_heuristics as heuristics

class MCTS():
    """Perform MCTS with a large number of simluations to determine the next move policy
    for a given board
    """
    def __init__(self, board, nn, simluation_number = 100, random_seed = 2):
        """Initialize the MCTS instance
        Args:
            simluation_number: number of simluations in MCTS before calculating a pi (next move policy)
        Fields:
            self.simluation_number_remaining: how many simluations are yet to be done
            self.root_node: the root node for MCTS simluations
            self.nn: instance of AlphaGo0 model used for this iteration of self play
        """
        self.simluation_number_remaining = simluation_number
        self.nn = nn
       
        #self.nn = heuristics.GoBoard2Heuristics()
        move_p_dist_root, _ = self.nn.predict(board)
        self.root_node = node.node(board, parent_edge = None, edges=[], action_value=0, move_p_dist=move_p_dist_root)
        self.random_seed = random_seed

    def calculate_U_for_edge(self, parent_node, edge):
        """ Calculate U (related to prior probability and explore factor) for an edge using
        a variant of the PUCT algorithm
        U = c_puct * P(edge) * sqrt(sum of N of parent_node's all edges) / (1 + N(edge))
        """
        c_puct = 0.5 # Exploration constant
        sum_N_for_all_edges = sum([other_edge.N for other_edge in parent_node.edges])
        U = c_puct * edge.P * sqrt(sum_N_for_all_edges) / (1 + edge.N)
        return U

    def select_edge(self, current_node, type):
        """Select the edge attached to current_node that has the largest U+Q
        U is related to prior probability and explore factor and Q is action value
        Args:
            current_node: the node from which the edges will be selected
            type: "max" or "min" indicating how the edge is selected
        Returns:
            edge: edge class instance with the largest U+Q, None if no edge exists
        """
        all_edges = current_node.edges
        selected_edge = None
        largest_qu_val = 0

        edge_to_qu_val = {edge: edge.Q + self.calculate_U_for_edge(current_node, edge) for edge in all_edges}
        if edge_to_qu_val != {}:
            selected_edge = max(edge_to_qu_val, key=edge_to_qu_val.get)
        return selected_edge

    def run_one_simluation(self):
        """Run one simluation within MCTS including select, expand leaf node and backup
        """
        current_node = self.root_node

        #traverse the tree till leaf node
        edge_type_max = True
        selected_edge = True #Initial value != None, will change in loop
        while selected_edge != None:
            if edge_type_max:
                selected_edge = self.select_edge(current_node, "max")
            else:
                selected_edge = self.select_edge(current_node, "min")
            edge_type_max = not edge_type_max
            if selected_edge != None:
                current_node = selected_edge.to_node
        #Now current_node is a leaf node with no outgoing edges

        #expand and evaluate
        current_board = current_node.go_board
        (move_p_dist, v) = current_node.move_p_dist, current_node.action_value

        for next_move in move_p_dist:
            p = move_p_dist[next_move]
            is_move_valid, new_board = go_utils.make_move(current_board, next_move)
         
            if is_move_valid: #expand the edge

                new_edge = edge.edge(from_node=current_node, to_node=None, W=0, Q=0, N=0, P=p, move=next_move)
                current_node.edges.append(new_edge)

                move_p_dist_next_node, action_value_next_node = self.nn.predict(new_board)
     
                next_node = node.node(new_board, new_edge, edges=[], action_value=action_value_next_node, move_p_dist=move_p_dist_next_node)
                new_edge.to_node = next_node

                #backup from leaf node next_node to root
                while next_node.parent_edge != None: #Continue when it is not root node
                    next_node.parent_edge.N = next_node.parent_edge.N + 1
                    next_node.parent_edge.W = next_node.parent_edge.W + next_node.action_value
                    next_node.parent_edge.Q = next_node.parent_edge.W * 1.0 / next_node.parent_edge.N
                    next_node = next_node.parent_edge.from_node

                    child_node_counter = 0
                    next_node.action_value = 0
                    for next_node_edge in next_node.edges:
                        next_node.action_value += next_node_edge.to_node.action_value
                        child_node_counter += 1
                   
                    next_node.action_value /= child_node_counter

    def run_all_simulations(self):
        """Run the specified number of simluations according to simluation_number
        when initializing the object. Returns a policy pi for board's next move
        according to these simluations
        Returns: 
            (new_board, move)
            move: the best move generated according to the MCTS simulations
            new_board: board and its configurations after the best move is placed
            policy: a size dimension x dimension + 1 array indicating the possibility of each move
        """
        for i in range(self.simluation_number_remaining):
            self.run_one_simluation()

        np.random.seed(seed=self.random_seed)
        #Pick the most explored move for root node with randomization
        root_edges = self.root_node.edges
    
        policy = np.zeros(self.nn.go_board_dimension*self.nn.go_board_dimension+1)
        sum_N = sum([edge.N for edge in root_edges])
        for edge in root_edges:
            (r, c) = edge.move
            if (r == -1) and (c == -1): #Pass
                policy[self.nn.go_board_dimension*self.nn.go_board_dimension] = (edge.N * 1.0 / sum_N)**2
            else:
                policy[r*self.nn.go_board_dimension+c] = (edge.N * 1.0 / sum_N)**2 #Temparature = 0.5 intermediate amount of exploration

        #Additional exploration is achieved by adding Dirichlet noise to the prior probabilities 
        policy_with_noise = 0.75 * policy
        sum_prob = sum(policy)
        policy = [p / sum_prob for p in policy]
        noise = 0.25 * np.random.dirichlet(0.3 * np.ones(len(policy)))
        #Not adding noise to moves where p = 0
        policy_with_noise = [ p + noise[i] if abs(p) > 1e-3 else p for (i, p) in enumerate(policy_with_noise)]
        #Make probbilities add up to zero
        sum_prob = sum(policy_with_noise)
        policy_with_noise = [p / sum_prob for p in policy_with_noise]
        move_indices = [i for i in range(self.nn.go_board_dimension**2+1)]
  
        move_index = np.random.choice(move_indices, 1, p = policy_with_noise)[0]

        if move_index == self.nn.go_board_dimension**2:
            move = (-1, -1)
        else:
            r = int(move_index / self.nn.go_board_dimension)
            c = move_index % self.nn.go_board_dimension
            move = (r, c)
        valid_move, new_board = go_utils.make_move(self.root_node.go_board, move)

        assert valid_move == True

        return new_board, move, policy
