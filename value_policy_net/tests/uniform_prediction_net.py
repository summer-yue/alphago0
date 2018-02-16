# This is a fake class used for mcts and self playing testing
# nn in our unit tests uses UniformPredictionNet instances
# nn.predict() always returns the same probability for all available spot on the board
# that is compatibile with the go rules
from go import go_utils
import os

class UniformPredictionNet():
    """ Fake class used for mcts and self playing testing
    """
    def __init__(self, path_to_model = '/', go_board_dimension = 9):
        self.go_board_dimension = go_board_dimension
        self.path_to_model = path_to_model
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)

    def predict(self, board):
        """ Returns the available spot (by row and then by col) is 1 everything 
        Returns:
            p: the fake probability distribution of the next move according to current policy. including pass.
            v: the fake probability of winning from this board, always set to 0.2
        """
        prob = 0
        available_moves = []
        for r in range(self.go_board_dimension):
            for c in range(self.go_board_dimension):
                move = (r, c)
                can_move, _ = go_utils.make_move(board, move)
                if can_move:
                    available_moves.append(move)
        available_moves.append((-1, -1))

        if len(available_moves) > 0:
            prob = 1.0 / (len(available_moves))
        else:
            prob = 0

        p = []
        for move in available_moves:
            p.append((move, prob))

        return p, 0.2