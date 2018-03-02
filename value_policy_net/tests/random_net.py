# This class is used as a random benchmark evaluated against the neural net.
from game.go_utils import GoUtils
import os

class RandomNet():
    """ Fake class used for mcts and self playing testing
    This spits out moves randomly
    """
    def __init__(self, path_to_model = '/', board_dimension = 2):
        self.board_dimension = board_dimension
        self.path_to_model = path_to_model
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
        self.utils = GoUtils()

    def predict(self, board):
        """ Returns the available spot (by row and then by col) is 1 everything 
        Returns:
            p: the fake probability distribution of the next move according to current policy. including pass.
            v: the fake probability of winning from this board, always set to 0.2
        """
        prob = 0
        available_moves = []
        for r in range(self.board_dimension):
            for c in range(self.board_dimension):
                move = (r, c)
                available_moves.append(move)
        available_moves.append((-1, -1))

        prob = 1.0 / (len(available_moves))
        p = {}
        for move in available_moves:
            p[move] = prob

        return p, 0