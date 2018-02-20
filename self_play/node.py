class node():
    def __init__(self, go_board, parent_edge, edges, action_value, move_p_dist):
        """A node in the MCTS algorithm containing information about the board, 
        and its incoming and outcoming edges
        Args:
            go_board: the go board the node represents
            parent_edge: the edge that links to its parent node
            edges: the edges that link to its children nodes
        """
        self.go_board = go_board
        self.parent_edge = parent_edge
        self.edges = edges
        self.action_value = action_value
        self.move_p_dist = move_p_dist

    def get_edge_info(self):
        return str([str(edge) for edge in self.edges])

    def is_leaf(self):
        return self.edges == []

    def __str__(self):
        return "On Node \n Number of outgoing edges: " \
            + str(len(self.edges)) + "\n  action_value:" + str(self.action_value) \
            + "\n board looks like" + str(self.go_board.board_grid)
