class self_play():
    """Algorithm plays against itself till the game ends and produce a set of (board, policy, result)
    Used as training data for the res net.
    """
    def __init__(self, root_node, nn):
        """Initialize an instance of self play with a starting node
        Args:
            root_node: a node instance representing the starting board
            nn: instance of current AlphaGo0 model
        Fields:
            self.nn: instance of AlphaGo0 model used for this iteration of self play
            self.current_node: the current node during self play
            self.moves: track the history of the nodes played and their corresponding pi
                pi is the probabilty for next moves according to the MCTS simulations
        """
        self.nn = nn
        self.current_node = root_node
        self.moves = {}

    def play_one_move(self):
        """Use MCTS to calculate the pi for current node, 
           update self.moves and current node
           Return true if a move is made, return False if player decided to pass
           Returns:
                Boolean value indicating if a move was placed
        """
        pass

    def play_till_finish(self):
        """Play until the game reaches a final state (2 passes happen one after another)
        Returns:
            A set of (board, pi, result) ready to be used as training labels
        """
        pass