import tensorflow as tf
import os
import random

class ResNet():
    """Go algorithm without human knowledge
    Original paper from: https://www.nature.com/articles/nature24270.pdf
    Using a res net and capability amplification with Monte Carlo Tree Search
    """
    def __init__(self, go_board_dimension = 5):
        """Initialize a supervised learning res net model
        Args:
            path_to_model: path to where the tf model locates
            go_board_dimension: dimension for the go board to learn. A regular go board is 19*19
                the default is 9*9 so it's convenient to train and run tests on.
        """
        self.go_board_dimension = go_board_dimension

        #Define the tensors
        x = tf.placeholder(tf.float32, [None, self.go_board_dimension, self.go_board_dimension, 3], name="input")
        self.nn = self.build_network(x) #Output tensor from the resnet

    def loss(p, v, z, pi, theta):
        c = tf.constant(1, dtype=float32, name="c")
        return tf.square(z-v) - tf.multiply(pi, tf.log(p)) + tf.multiply(c, tf.nn.l2_normalize(theta))

    def build_network(self, x):
        """ResNet structure TODO: @Ben make this prettier.
        Args:
            x: input as a tf placeholder of dimension board_dim*board_dim*3
        Returns:
            Z: output tensor of size 2
        """

        with tf.variable_scope("conv1", reuse=True) as scope:
            Z = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME")
            A = tf.nn.relu(Z, name="A")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv2") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv1/A"), filters=32, kernel_size=3, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + data_in
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv3") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv2/A"), filters=32, kernel_size=3, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("conv4") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv3/A"), filters=32, kernel_size=3, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + tf.get_variable("conv2/A")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("pool1") as scope:
            A = tf.layers.max_pooling2d(tf.get_variable("conv4/A"), pool_size=2, strides=2, padding="VALID")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv5") as scope:
            Z = tf.layers.conv2d(tf.get_variable("pool1/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("conv6") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv5/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + tf.get_variable("pool1/A")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv7") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv6/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("conv8") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv7/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + tf.get_variable("conv6/A")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("pool2") as scope:
            A = tf.layers.max_pooling2d(tf.get_variable("conv8/A"), pool_size=2, strides=2, padding="VALID")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv9") as scope:
            Z = tf.layers.conv2d(tf.get_variable("pool2/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("conv10") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv9/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + tf.get_variable("pool2/A")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv11") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv10/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("conv12") as scope:
            Z = tf.layers.conv2d(tf.get_variable("conv11/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + tf.get_variable("conv10/A")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("pool3") as scope:
            A = tf.layers.max_pooling2d(tf.get_variable("conv12/A"), pool_size=2, strides=2, padding="VALID")
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("fc") as scope:
            P = tf.contrib.layers.flatten(tf.get_variable("pool3/A"))
            P = tf.nn.relu(P)
            Z = tf.contrib.layers.fully_connected(P, 100)
            A = tf.nn.relu(Z)
            Z = tf.contrib.layers.fully_connected(A, 2)
            return Z

    def train(self, model_path, game_number=1000):
        """Train the res net model with results from each iteration of self play.
        Args:
            model_path: location where we want the final model to be saved
            game_number: number of self play games used on training
        Returns:
            None, but a model is saved at the path
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        BLACK = 1 # black goes first
        board = go_board(self.go_board_dimension, BLACK, board_grid=None, game_history=None)
        mcts = MCTS(board) # root has no parent edge

        play = self_play(board, mcts.root_node, self.nn)

        play.play_till_finished()

    def fake_train(self, model_path, training_data_num = 1000):
        """This function is ued for testing the resNet independent of the mcts and self play code.
        The goal is to teach the resNet to count the number of black and white stones on a board.
        """
        pass

    def generate_fake_data(self, training_data_num):
        """Generate fake boards and counts the number of black and white stones as labels.
        Args:
            training_data_num: the number of fake training data we want to generate
        Returns:
            Xs: a list of training boards
            Ys: a list of training labels, each label is a size 2 array indicating the count for black and white stones
        """
        Xs = []
        Ys = []

        options = [-1, 0, 1] #white empty black
        for i in training_data_num:
            black_stone_count = 0
            white_stone_count = 0

            board = [[random.choice(options) for c in range(self.go_board_dimension)] for r in range(self.go_board_dimension)]
            for r in range(self.go_board_dimension):
                for c in range(self.go_board_dimension):
                    if board[r][c] == -1:
                        white_stone_count += 1
                    elif board[r][c] == 1:
                        black_stone_count += 1
            Xs.append(board)
            Ys.append([black_stone_count, white_stone_count])
        return Xs, Ys

    def predict(self, board):
        """Given a board. predict (p,v) according to the current res net
        Args:
            board: current board including the current player and stone distribution
        Returns:
            p: the probability distribution of the next move according to current policy. including pass
            v: the probability of winning from this board.
        """
        pass

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

    def convert_to_one_hot_go_boards(self, original_board):
        """Convert the format of the go board from a dim by dim 2d array to a dim by dim by 3 3d array.
        This is used before feed the boards into the neural net.
        Args:
            original_board: a board_dimension x board_dimension array, each element can be -1 (white), 0 (empty) or 1 (black).
        Returns:
            flattend_board: a board_dimension x board_dimension array x 3 one hot vector
        """
        board_dim = len(original_board)
        return [[self.helper_convert_to_one_hot(original_board[r][c]) for c in range(board_dim)] for r in range(board_dim)]

    def helper_convert_to_one_hot(self, element):
        """ Transformation 1 -> [0,0,1]; 0->[0,1,0], -1 -> [-1,0,1]
        Args:
            element: number to be transformed into an array, has to be -1, 0 or 1
        Return:
            array of size 3 the element is transformed to
        """
        transformation = {
            -1: [1,0,0],
            0:  [0,1,0],
            1:  [0,0,1]
        }
        return transformation[element]

