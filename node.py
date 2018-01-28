class node():
    def __init__(self, go_board, parent_edge, edges, action_value):
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
