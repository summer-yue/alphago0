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
        self.yp = tf.placeholder(tf.float32, [None,  self.go_board_dimension*self.go_board_dimension + 1], name="labels_p")
        self.yv = tf.placeholder(tf.float32, [None, 1], name="labels_v")
        self.yp_, self.yv_, self.yp_logits, self.yv_logits = self.build_network(self.x) 
            
        # TODO change loss function and accuracy
        self.loss = tf.losses.absolute_difference(labels=self.yv, predictions=self.yv_) + tf.losses.absolute_difference(labels=self.yp, predictions=self.yp_)
        self.accuracy = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(self.yp, self.yp_),tf.equal(self.yv, self.yv_)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,)
        self.train_op = self.optimizer.minimize(self.loss)

    def build_conv_block(self, input_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=64, kernel_size=3, strides=1, padding="SAME")
            Z = tf.layers.batch_normalization(Z)
            A = tf.nn.relu(Z, name="A")
            return A

    def build_res_layer(self, input_tensor, res_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=64, kernel_size=3, strides=1, padding="SAME")
            Z = tf.layers.batch_normalization(Z)
            A = tf.nn.relu(Z)
            A = A + res_tensor
            return A

    def build_res_block(self, input_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            A1 = self.build_conv_block(input_tensor=input_tensor, varscope="conv1")
            A2 = self.build_res_layer(input_tensor=A1, res_tensor=input_tensor, varscope="res2")
            return A2

    def build_pooling_layer(self, input_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            A = tf.layers.max_pooling2d(input_tensor, pool_size=2, strides=2, padding="VALID")
            return A

    def build_head_conv_layer(self, input_tensor, varscope, filter):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=filter, kernel_size=1, strides=1, padding="SAME")
            Z = tf.layers.batch_normalization(Z)
            A = tf.nn.relu(Z, name="A")
            return A

    def build_network(self, x):
        """ResNet structure TODO: @Ben make this prettier.
        Args:
            x: input as a tf placeholder of dimension board_dim*board_dim*3
        Returns:
            p_logits, v_logits: the logits for policy and value
            P, V: output of policy and value heads
        """

        A = self.build_conv_block(input_tensor=x, varscope="conv1")

        for i in range(10):
            A = self.build_res_block(input_tensor=A, varscope="res" + str(i))

        #Policy head
        ph1 = self.build_head_conv_layer(A, "policy_head", filter=2)
        ph1 = tf.contrib.layers.flatten(ph1)
        p_logits = tf.contrib.layers.fully_connected(ph1, self.go_board_dimension*self.go_board_dimension+1)

        #Value head
        vh1 = self.build_head_conv_layer(A, "value_head", filter=1)
        vh2 = tf.contrib.layers.fully_connected(vh1, 256)
        vh2 = tf.nn.relu(vh2, name="vh2")
        vh2 = tf.contrib.layers.flatten(vh2)
        v_logits = tf.contrib.layers.fully_connected(vh2, 1)
        
        P = tf.nn.softmax(p_logits)
        V = tf.nn.tanh(v_logits)

        return P, V, p_logits, v_logits

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


