from self_play import self_play
from go import go_board
from value_policy_net import resnet
from self_play import mcts
from pyprind import prog_bar
import numpy as np

import tensorflow as tf

BLACK = 1
WHITE = -1

class AlphaGoZero():
    def __init__(self, model_path, restored):
        """
        Args:
            model_path: path to the model to be restored from or save to
            restored: boolean indicating if we want to restore a saved model
        """
        self.model_path = model_path
        self.sess = tf.Session()
        with self.sess.as_default():
            self.nn = resnet.ResNet(go_board_dimension = 5, model_path = model_path, restored=restored)

    def train_nn(self, training_game_number = 1000):
        """Training the resnet by self play using MCTS
        With experience replay
        Args:
            training_game_number: number of self play games
        Returns:
            Nothing, but model_path/game_1 has the model trained
        """
        BATCH_SIZE = 100
        BUCKET_SIZE = 1000 # bucket size used in experience replay
        BLACK = 1 # black goes first
        batch_num = 0

        # batch_training_sample_size = 0
        bucket_training_boards = np.empty(0)
        bucket_training_labels_p = np.empty(0)
        bucket_training_labels_v = np.empty(0)

        batch_training_boards = np.empty(0)
        batch_training_labels_p = np.empty(0)
        batch_training_labels_v = np.empty(0)

        with self.sess.as_default():
            for game_num in prog_bar(range(training_game_number)):
                print("training game:", game_num+1)
                board = go_board.go_board(self.nn.go_board_dimension, BLACK, board_grid=None, game_history=None)
                ts = mcts.MCTS(board, self.nn)
                play = self_play.self_play(board, self.nn)
                training_boards, training_labels_p, training_labels_v = play.play_till_finish()
            
                # Fill the bucket with current game's boards, around 20
                if len(bucket_training_boards) == 0:
                    bucket_training_boards = training_boards
                if len(bucket_training_labels_p) == 0:
                    bucket_training_labels_p = training_labels_p
                if len(bucket_training_labels_v) == 0:
                    bucket_training_labels_v = training_labels_v
                bucket_training_boards = np.append(bucket_training_boards, training_boards, axis=0)
                bucket_training_labels_p = np.append(bucket_training_labels_p, training_labels_p, axis=0)
                bucket_training_labels_v = np.append(bucket_training_labels_v, training_labels_v, axis=0)

                # Remove from the front if bucket size exceeds the specified bucket size
                if len(bucket_training_labels_v) > BUCKET_SIZE:
                    deleted_indices = [i for i in range(len(bucket_training_labels_v) - BUCKET_SIZE)]
                    bucket_training_boards = np.delete(bucket_training_boards, deleted_indices)
                    bucket_training_labels_p = np.delete(bucket_training_labels_p, deleted_indices)
                    bucket_training_labels_v = np.delete(bucket_training_labels_v, deleted_indices)

                    # Take BATCH_SIZE number of random elements from the bucket and train
                    BUCKET_INDICES = [i for i in range(BUCKET_SIZE)]
                    batch_indices = np.random.choice(BUCKET_INDICES, BATCH_SIZE, replace=False)
                    batch_training_boards = np.take(bucket_training_boards, batch_indices, axis=0)
                    batch_training_labels_p = np.take(bucket_training_labels_p, batch_indices, axis=0)
                    batch_training_labels_v = np.take(bucket_training_labels_v, batch_indices, axis=0)
                    batch_num += 1
                    if batch_num%10 == 0: #Save every 10 batches
                        model_path = self.model_path + '/batch_' + str(batch_num)
                        self.nn.train(batch_training_boards, batch_training_labels_p, batch_training_labels_v, model_path)
                    else:
                        self.nn.train(batch_training_boards, batch_training_labels_p, batch_training_labels_v, model_path)

    def play_with_raw_nn(self, board):
        """Play a move with the raw res net
        Args:
            board: current board including the current player and stone distribution
        Returns:
            next_move: (row, col) indicating where the neural net would place the stone
            winning_prob: probability of winning by playing this move acording to out neural net
        """
        p, winning_prob = self.nn.predict(board)
        move_index = np.argmax(p)
        if (move_index == board.board_dimension*board.board_dimension):
            (r, c) = (-1, -1)
        else:
            r = int(move_index / board.board_dimension)
            c = int(move_index % board.board_dimension)
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
    alphpago0 = AlphaGoZero(model_path="../models", restored=False)
    alphpago0.train_nn()
    