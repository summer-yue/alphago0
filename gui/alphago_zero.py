from self_play import self_play
from go import go_board
from value_policy_net import resnet
from self_play import mcts

BLACK = 1
WHITE = -1

class AlphaGoZero():
    def __init__(self, model_path):
        self.model_path = model_path

    def train_nn(self, training_game_number = 500):
        """Training the resnet by self play using MCTS
        Args:
            training_game_number: number of self play games
        Returns:
            Nothing, but model_path/game_1 has the model trained
        """
        BATCH_SIZE = 100
        BLACK = 1 # black goes first
        batch_training_boards = []
        batch_training_labels_p = []
        batch_training_labels_v = []

        self.nn = resnet.ResNet(go_board_dimension = 5)

        for i in range(training_game_number):
            print("training game:", i+1)
            board = go_board.go_board(self.nn.go_board_dimension, BLACK, board_grid=None, game_history=None)
            ts = mcts.MCTS(board, self.nn)
            play = self_play.self_play(board, self.nn)

            training_boards, training_labels_p, training_labels_v = play.play_till_finish()
            batch_training_sample_size += len(training_labels_v)
            
            print("batch_training_sample_size:", batch_training_sample_size)
            if batch_training_sample_size < BATCH_SIZE:
                batch_training_boards += training_boards
                batch_training_labels_p += batch_training_labels_p
                batch_training_labels_v += training_labels_v
            else:
                model_path = self.model_path + '/batch_' + str(i)
                self.nn.train(batch_training_boards, batch_training_labels_p, batch_training_labels_v, model_path)
                batch_training_boards = training_boards
                batch_training_labels_p = batch_training_labels_p
                batch_training_labels_v = training_labels_v

    def play_with_raw_nn(self, board):
        """Play a move with the raw res net
        Args:
            board: current board including the current player and stone distribution
        Returns:
            next_move: (row, col) indicating where the neural net would place the stone
            winning_prob: probability of winning by playing this move acording to out neural net
        """
        p, winning_prob = self.nn.predict(board)
        move_index = argmax(p)
        if (move_index == board.board_dimension*board.board_dimension):
            (r, c) = (-1, -1)
        else:
            r = move_index / board.board_dimension
            c = move_index % board.board_dimension
        next_move = (r, c)
        return next_move, winning_prob

    def play_with_mcts(self, board):
        """Play a move with the res net and another round of Monte Carlo Tree Search
        Args:
            board: current board including the current player and stone distribution
        Returns:
            next_move: (row, col) indicating where the neural net with MCTS would place the stone
        """
        pass
        
if __name__ == '__main__':
    alphpago0 = AlphaGoZero(model_path="../models")
    alphpago0.train_nn()
    