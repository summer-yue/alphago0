import edge
import node
import go_utils

from math import sqrt

class MCTS():
    """Perform MCTS with a large number of simluations to determine the next move policy
    for a given board
    """
    def __init__(self, board, nn, simluation_number = 1000):
        """Initialize the MCTS instance
        Args:
            simluation_number: number of simluations in MCTS before calculating a pi (next move policy)
        Fields:
            self.simluation_number_remaining: how many simluations are yet to be done
            self.root_node: the root node for MCTS simluations
            self.nn: instance of AlphaGo0 model used for this iteration of self play
        """
        self.simluation_number_remaining = simluation_number
        self.root_node = node.node(board, parent_edge = None, edges=[], action_value=0)
        
        self.nn = nn

    def calculate_U_for_edge(self, parent_node, edge):
        """ Calculate U (related to prior probability and explore factor) for an edge using
        a variant of the PUCT algorithm
        U = c_puct * P(edge) * sqrt(sum of N of parent_node's all edges) / (1 + N(edge))
        """
        c_puct = 0.5 # Exploration constant
        sum_N_for_all_edges = 0
        for other_edge in parent_node.edges:
            sum_N_for_all_edges = sum_N_for_all_edges + other_edge.N
        U = c_puct * edge.P * sqrt(sum_N_for_all_edges) / (1 + edge.N)
        return U

    def select_edge(self, current_node):
        """Select the edge attached to current_node that has the largest U+Q
        U is related to prior probability and explore factor and Q is action value
        Returns:
            edge: edge class instance with the largest U+Q, None if no edge exists
        """
        all_edges = current_node.edges
        selected_edge = None
        largest_qu_val = 0

        for edge in all_edges:
            #print(edge)
            edge_u = self.calculate_U_for_edge(current_node, edge)
            #print("edge.u:" + str(edge_u))
            #print("edge.Q:" + str(edge.Q))
            qu_val = edge.Q + edge_u
            if qu_val > largest_qu_val:
                largest_qu_val = qu_val
                selected_edge = edge
        return selected_edge

    def run_one_simluation(self):
        """Run one simluation within MCTS including select, expand leaf node and backup
        """
        current_node = self.root_node

        #traverse the tree till leaf node
        selected_edge = 0
        while selected_edge != None:
            selected_edge = self.select_edge(current_node)
            if selected_edge != None:
                current_node = selected_edge.to_node
        #Now current_node is a leaf node with no outgoing edges

        #expand and evaluate
        potential_next_moves = []
        current_board = current_node.go_board

        (move_p_dist, v) = self.nn.predict(current_board)
        for (next_move, p) in move_p_dist:

            is_move_valid, new_board = go_utils.make_move(current_board, next_move)

            if is_move_valid: #expand the edge
                potential_next_moves.append(next_move)

                new_edge = edge.edge(from_node=current_node, to_node=None, W=0, Q=0, N=0, P=p, move=next_move)
                current_node.edges.append(new_edge)

                _, av_next_node = self.nn.predict(new_board)
                next_node = node.node(new_board, new_edge, edges=[], action_value=av_next_node)
                new_edge.to_node = next_node

                #TODO @Ben: batch the evaluation step here with resNet to improve efficiency
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
                    #print("child_node_counter:", child_node_counter)
                    #print("next_node.action_value", next_node.action_value)
                    next_node.action_value /= child_node_counter

    def run_all_simulations(self):
        """Run the specified number of simluations according to simluation_number
        when initializing the object. Returns a policy pi for board's next move
        according to these simluations
        Returns: 
            (new_board, move)
            move: the best move generated according to the MCTS simulations
            new_board: board and its configurations after the best move is placed
        """
        for i in range(self.simluation_number_remaining):
            self.run_one_simluation()

        #Pick the mostly explored edge for root node
        root_edges = self.root_node.edges
        most_used_edge = None
        most_explored_time = 0
        print(len(root_edges))
        for edge in root_edges:
            if edge.N > most_explored_time:
                most_explored_time = edge.N
                most_used_edge = edge

        #Pick the next move
        move = most_used_edge.move
        new_board = most_used_edge.to_node.go_board

        return new_board, move
