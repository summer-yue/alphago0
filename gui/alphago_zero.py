import numpy as np
import operator
import tensorflow as tf

from pyprind import prog_bar

from game.go_board import GoBoard
from game.go_utils import GoUtils
from self_play.mcts import MCTS
from self_play.self_play import SelfPlay
from value_policy_net.resnet import ResNet

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
        self.utils = GoUtils()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.nn = ResNet(board_dimension = 5, model_path = model_path, restored=restored)

    def train_nn(self, training_game_number = 1000):
        """Training the resnet by self play using MCTS
        With experience replay
        Args:
            training_game_number: number of self play games
        Returns:
            Nothing, but model_path/game_1 has the model trained
        Notes:
            Training 1000 games, total board number seen = 1000 * 20 = 20,000
            After each game, 100 boards are sampled. Each board is used 5 times.
            Equivalent to 20,000 data over 5 epochs, 100,000 boards seen
            Fake dataset also had 100,000 data seen (achieved 96% test accuracy on 50 test boards for counting)
        """
        BATCH_SIZE = 100
        BUCKET_SIZE = 500 # bucket size used in experience replay
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
                board = GoBoard(self.nn.go_board_dimension, BLACK, board_grid=None, game_history=None)
                ts = MCTS(board, self.nn)
                play = SelfPlay(board, self.nn)
                training_boards, training_labels_p, training_labels_v = play.play_till_finish()
                #print("training_labels_p:", training_labels_p.shape)
            
                # Fill the bucket with current game's boards, around 20
                if len(bucket_training_boards) == 0:
                    bucket_training_boards = training_boards
                if len(bucket_training_labels_p) == 0:
                    bucket_training_labels_p = training_labels_p
                if len(bucket_training_labels_v) == 0:
                    bucket_training_labels_v = training_labels_v
                bucket_training_boards = np.append(bucket_training_boards, training_boards, axis=0)
                bucket_training_labels_p = np.append(bucket_training_labels_p, training_labels_p, axis=0)
                #print("bucket_training_labels_p:", bucket_training_labels_p.shape)
                bucket_training_labels_v = np.append(bucket_training_labels_v, training_labels_v, axis=0)

                # Remove from the front if bucket size exceeds the specified bucket size
                if len(bucket_training_labels_v) > BUCKET_SIZE:
                    deleted_indices = [i for i in range(len(bucket_training_labels_v) - BUCKET_SIZE)]
                    bucket_training_boards = np.delete(bucket_training_boards, deleted_indices, axis=0)
                    bucket_training_labels_p = np.delete(bucket_training_labels_p, deleted_indices, axis=0)
                    bucket_training_labels_v = np.delete(bucket_training_labels_v, deleted_indices, axis=0)
                    #print("bucket_training_labels_p:", bucket_training_labels_p.shape)
                    # Take BATCH_SIZE number of random elements from the bucket and train
                    BUCKET_INDICES = [i for i in range(BUCKET_SIZE)]
                    batch_indices = np.random.choice(BUCKET_INDICES, BATCH_SIZE, replace=False)
                    batch_training_boards = np.take(bucket_training_boards, batch_indices, axis=0)
                    batch_training_labels_p = np.take(bucket_training_labels_p, batch_indices, axis=0)
                    #print("batch_training_labels_p:", batch_training_labels_p.shape)
                    batch_training_labels_v = np.take(bucket_training_labels_v, batch_indices, axis=0)
                    batch_num += 1
                    if batch_num%10 == 0: #Save every 10 batches
                        model_path = self.model_path + '/batch_' + str(batch_num)
                        self.nn.train(batch_training_boards, batch_training_labels_p, batch_training_labels_v, model_path)
                    else:
                        self.nn.train(batch_training_boards, batch_training_labels_p, batch_training_labels_v)

    def play_with_raw_nn(self, board):
        """Play a move with the raw res net
        Args:
            board: current board including the current player and stone distribution
        Returns:
            next_move: (row, col) indicating where the neural net would place the stone
            winning_prob: probability of winning by playing this move acording to out neural net
        """
        potential_moves_policy, winning_prob = self.nn.predict(board)

        #print("policy is:", potential_moves_policy)
        found_move = False
        while not found_move:
            board_copy = board.copy()
            next_move = max(potential_moves_policy.items(), key=operator.itemgetter(1))[0]
            is_valid_move, new_board = self.utils.make_move(board_copy, next_move)

            #Only makes the move when the move is valid.
            if is_valid_move:
                found_move = True
            else:
                potential_moves_policy.pop(next_move)

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
    