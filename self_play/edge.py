class Edge():
    def __init__(self, from_node, to_node, W, Q, N, P, move):
        """Initialize the edge representing each board move used in MCTS
        Args:
            from_board: parent node edge is associated with
            to_board: child node the edge is associated with
            W: tracks the sum of values of all explored children nodes
            Q: tracks the mean values of all explored children nodes
            N: the number of times the move has been visited
            P: probability assigned by the nn to make this move
            move: a tuple of (player, row, col) indicating what the move is, row = col = -1 if pass
        """
        self.from_node = from_node
        self.to_node = to_node
        self.W = W
        self.Q = Q
        self.N = N
        self.P = P
        self.move = move

    def __str__(self):
        return "NEW EDGE: W (sum of values of all explored children nodes): " + str(self.W) + " \n  " \
            + "Q (mean values of all explored children nodes): " + str(self.Q) + " \n  " \
            + "N (number of times the move has been visited): " + str(self.N) + " \n  " \
            + "P ( probability assigned by the nn to make this move): " + str(self.P) + " \n" \
            + "  Parent board:" + str(self.from_node.board.board_grid)  + " \n" \
            + "  Child board:" + str(self.to_node.board.board_grid) 


