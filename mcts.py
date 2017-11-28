class MCTS():
    """Perform MCTS with a large number of simluations to determine the next move policy
    for a given board
    """
    def __init__(self, board, simluation_number = 1000):
        """Initialize the MCTS instance
        Args:
            simluation_number: number of simluations in MCTS before calculating a pi (next move policy)
        Fields:
            self.simluation_number_remaining: how many simluations are yet to be done
            self.root_node: the root node for MCTS simluations
        """
        self.simluation_number_remaining = simluation_number
        self.root_node = node(board, parent_edge = None, edges = [])

    def run_one_simluation(self):
        """Run one simluation within MCTS including select, expand leaf node and backup
        """
        pass

    def run_all_simulations(self):
        """Run the specified number of simluations according to simluation_number
        when initializing the object. Returns a policy pi for board's next move
        according to these simluations
        Returns:
            pi: the policy for next moves according to simluations, a vector with length
            board_dimension ** 2 + 1
        """
        pass
