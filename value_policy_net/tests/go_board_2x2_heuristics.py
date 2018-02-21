# This is a fake class used for mcts and self playing testing
# nn.predict() always returns uniform distribution for available moves
# nn.predict() returns the value 1=self cross corner, -1 opponent cross corner and 0 otherwise
# that is compatibile with the go rules
from go import go_utils
import os

class GoBoard2Heuristics():
    """ Fake class used for mcts and self playing testing
    """
    def __init__(self, path_to_model = '/', go_board_dimension = 2):
        self.go_board_dimension = go_board_dimension
        self.path_to_model = path_to_model
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)

    def count_stones(self, board_grid):
        count = 0
        value_sum = 0
        for r in range(self.go_board_dimension):
            for c in range(self.go_board_dimension):
                if abs(board_grid[r][c]) > 1e-3:
                    count += 1
                    value_sum += board_grid[r][c]
        return count, value_sum

    def almost_equal(self, a, b):
        return abs(a - b) < 1e-3

    def get_value(self, board):
        """returns the value 1=self cross corner, -1 opponent cross corner and 0 otherwise
        """
        board_grid = board.board_grid
        player = board.player
        stone_num, value_sum = self.count_stones(board_grid)
        if stone_num != 2:
            return 0

        if self.almost_equal(value_sum, 0):
            return 0

        if self.almost_equal(value_sum, 2): #both black
            #both corners
            if self.almost_equal(board_grid[0][0], 1) and self.almost_equal(board_grid[1][1], 1):
                return player
            if self.almost_equal(board_grid[0][1], 1) and self.almost_equal(board_grid[1][0], 1):
                return player

        elif  self.almost_equal(value_sum, -2): #both white
            #both corners
            if self.almost_equal(board_grid[0][0], -1) and self.almost_equal(board_grid[1][1], -1):
                return -player
            if self.almost_equal(board_grid[0][1], -1) and self.almost_equal(board_grid[1][0], -1):
                return -player

        return 0

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

        p = {}
        for move in available_moves:
            p[move] = prob

        return p, self.get_value(board)