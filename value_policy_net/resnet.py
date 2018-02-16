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

        #Define the tensors that compose the graph
        self.x = tf.placeholder(tf.float32, [None, self.go_board_dimension, self.go_board_dimension, 3], name="input")
        self.y = tf.placeholder(tf.float32, [None, 2], name="labels")
        self.y_ = self.build_network(self.x) #Output tensor from the resnet
        self.loss = tf.losses.absolute_difference(labels=self.y, predictions=self.y_)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,)
        self.train_op = self.optimizer.minimize(self.loss)

    def loss(p, v, z, pi, theta):
        c = tf.constant(1, dtype=float32, name="c")
        return tf.square(z-v) - tf.multiply(pi, tf.log(p)) + tf.multiply(c, tf.nn.l2_normalize(theta))

    def build_conv_layer(self, input_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=3, kernel_size=3, strides=1, padding="SAME")
            A = tf.nn.relu(Z, name="A")
            return A

    def build_res_layer(self, input_tensor, res_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=3, kernel_size=3, strides=1, padding="SAME")
            A = tf.nn.relu(Z)
            A = A + res_tensor
            return A

    def build_pooling_layer(self, input_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            A = tf.layers.max_pooling2d(input_tensor, pool_size=2, strides=2, padding="VALID")
            return A

    def build_network(self, x):
        """ResNet structure TODO: @Ben make this prettier.
        Args:
            x: input as a tf placeholder of dimension board_dim*board_dim*3
        Returns:
            Z: output tensor of size 2
        """

        A1 = self.build_conv_layer(input_tensor=x, varscope="conv1")
        A2 = self.build_res_layer(input_tensor=A1, res_tensor=x, varscope="conv2")
        A3 = self.build_conv_layer(input_tensor=A2, varscope="conv3")
        A4 = self.build_res_layer(input_tensor=A3, res_tensor=A2, varscope="conv4")
        Ap1 = self.build_pooling_layer(input_tensor=A4, varscope="pool1")
        A5 = self.build_conv_layer(input_tensor=Ap1, varscope="conv5")
        A6 = self.build_res_layer(input_tensor=A5, res_tensor=Ap1, varscope="conv6")
        A7 = self.build_conv_layer(input_tensor=A6, varscope="conv7")
        A8 = self.build_res_layer(input_tensor=A7, res_tensor=A6, varscope="conv8")
        Ap2 = self.build_pooling_layer(input_tensor=A8, varscope="pool2")

        with tf.variable_scope("fc") as scope:
            P = tf.contrib.layers.flatten(Ap2)
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

        play = self_play(board, mcts.root_node, self)

        play.play_till_finished()

    def fake_train(self, model_path, training_data_num = 1000):
        """This function is ued for testing the resNet independent of the mcts and self play code.
        The goal is to teach the resNet to count the number of black and white stones on a board.
        """
        pass

    def predict(self, board):
        """Given a board. predict (p,v) according to the current res net
        Args:
            board: current board including the current player and stone distribution
        Returns:
            p: the probability distribution of the next move according to current policy. including pass
            v: the probability of winning from this board.
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


