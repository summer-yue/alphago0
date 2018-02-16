from self_play import self_play
from go import go_board
from value_policy_net import resnet

BLACK = 1
WHITE = -1

class AlphaGoZero():
    def __init__(self, model_path):
        self.model_path = model_path

    def train_nn(self):
        self.nn = resnet.ResNet(go_board_dimension = 5)

    def play_with_raw_nn(self, board):
        """Play a move with the raw res net
        Args:
            board: current board including the current player and stone distribution
        Returns:
            next_move: (row, col) indicating where the neural net would place the stone
        """
        pass

    def play_with_mcts(self, board):
        """Play a move with the res net and another round of Monte Carlo Tree Search
        Args:
            board: current board including the current player and stone distribution
        Returns:
            next_move: (row, col) indicating where the neural net with MCTS would place the stone
        """
        pass
        
if __name__ == '__main__':
    pass
    