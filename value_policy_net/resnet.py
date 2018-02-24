import tensorflow as tf
import numpy as np
import os
import random

from go import go_board

class ResNet():
    """Go algorithm without human knowledge
    Original paper from: https://www.nature.com/articles/nature24270.pdf
    Using a res net and capability amplification with Monte Carlo Tree Search
    """
    def __init__(self, go_board_dimension = 5, model_path=None, restored=False):
        """Initialize a supervised learning res net model
        Args:
            go_board_dimension: dimension for the go board to learn. A regular go board is 19*19
                the default is 5*5 so it's convenient to train and run tests on.
            model_path: path to the model to be restored from or save to
            restored: boolean indicating if we want to restore a saved model
        """
        self.go_board_dimension = go_board_dimension

        #Define the tensors that compose the graph
        self.x = tf.placeholder(tf.float32, [None, self.go_board_dimension, self.go_board_dimension, 3], name="input")
        self.yp = tf.placeholder(tf.float32, [None,  self.go_board_dimension*self.go_board_dimension + 1], name="labels_p")
        self.yv = tf.placeholder(tf.float32, [None, 1], name="labels_v")
        self.yp_, self.yv_, self.yp_logits, self.yv_logits = self.build_network(self.x) 

        self.calc_loss()
        self.gradient = tf.gradients(self.loss, self.x)
        self.calc_accuracy()
        #self.optimizer = tf.train.MomentumOptimizer(1e-4, 0.9)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf.get_default_session()
        self.sess.run(tf.global_variables_initializer())

        if restored:
            saver = tf.train.Saver(max_to_keep=500)
            saver.restore(self.sess, model_path)

    def calc_accuracy(self):
        """Calculate the accuracy function for the fake value network
        used in fake_train testing
        Returns:
            The accuracy tensor
        """
        elements_0 = tf.equal(self.yv, 0)
        elements_0 = tf.cast(elements_0, tf.int32)
        predicted_right_count_zero = tf.less(tf.abs(self.yv - self.yv_), 0.3)
        predicted_right_count_zero = tf.cast(predicted_right_count_zero, tf.int32)
        predicted_right_count_zero = tf.multiply(elements_0, predicted_right_count_zero)
        predicted_right_count_zero = tf.reduce_sum(predicted_right_count_zero)

        elements_1 = tf.equal(self.yv, 1)
        elements_1 = tf.cast(elements_1, tf.int32)
        predicted_right_count_one = tf.less(tf.subtract(self.yv, self.yv_), 0.7)
        predicted_right_count_one = tf.cast(predicted_right_count_one, tf.int32)
        predicted_right_count_one = tf.multiply(elements_1, predicted_right_count_one)
        predicted_right_count_one = tf.reduce_sum(predicted_right_count_one)

        elements_neg1 = tf.equal(self.yv, -1)
        elements_neg1 = tf.cast(elements_neg1, tf.int32)
        predicted_right_count_neg_one = tf.less(tf.subtract(self.yv_, self.yv), 0.7)
        predicted_right_count_neg_one = tf.cast(predicted_right_count_neg_one, tf.int32)
        predicted_right_count_neg_one = tf.multiply(elements_neg1, predicted_right_count_neg_one)
        predicted_right_count_neg_one = tf.reduce_sum(predicted_right_count_neg_one)

        self.accuracy = (predicted_right_count_zero + predicted_right_count_one + predicted_right_count_neg_one)/tf.size(elements_0)

    def calc_loss(self):
        """Calculate the loss function for the policy-value network
        Returns:
            The loss tensor
        """
        value_loss = tf.losses.mean_squared_error(labels=self.yv, predictions=self.yv_)
        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.yp, logits=self.yp_logits)
        self.loss = value_loss + policy_loss

    def build_conv_block(self, input_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=64, kernel_size=3, strides=1, padding="SAME")
            # Z = tf.layers.batch_normalization(Z)
            A = tf.nn.relu(Z, name="A")
            return A

    def build_res_layer(self, input_tensor, res_tensor, varscope):
        with tf.variable_scope(varscope, reuse=tf.AUTO_REUSE) as scope:
            Z = tf.layers.conv2d(input_tensor, filters=64, kernel_size=3, strides=1, padding="SAME")
            Z = tf.layers.batch_normalization(Z)
            A = Z + res_tensor
            A = tf.nn.relu(A)
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

    def build_fc_network(self, x):
        v_logits = tf.contrib.layers.flatten(x)
        for i in range(4):
            v_logits = tf.contrib.layers.fully_connected(v_logits, 100)
        # v_logits = tf.contrib.layers.fully_connected(v_logits, 10)
        # v_logits = tf.nn.relu(v_logits)
        v_logits = tf.contrib.layers.fully_connected(v_logits, 1, activation_fn=None)
        V = tf.nn.tanh(v_logits)

        return V, V, v_logits, v_logits

    def build_network(self, x):
        """ResNet structure
        Args:
            x: input as a tf placeholder of dimension board_dim*board_dim*3
        Returns:
            p_logits, v_logits: the logits for policy and value
            P, V: output of policy and value heads
        """

        A = self.build_conv_block(input_tensor=x, varscope="conv1")

        for i in range(2):
            A = self.build_res_block(input_tensor=A, varscope="res" + str(i))

        #Policy head
        ph1 = self.build_head_conv_layer(A, "policy_head", filter=2)
        ph1 = tf.contrib.layers.flatten(ph1)
        p_logits = tf.contrib.layers.fully_connected(ph1, self.go_board_dimension*self.go_board_dimension+1, activation_fn=None)

        #Value head
        vh1 = self.build_head_conv_layer(A, "value_head", filter=1)
        vh2 = tf.contrib.layers.flatten(vh1)
        vh2 = tf.contrib.layers.fully_connected(vh2, 256)
        vh2 = tf.contrib.layers.fully_connected(vh2, 256)
        v_logits = tf.contrib.layers.fully_connected(vh2, 1, activation_fn=None)
        
        P = tf.nn.softmax(p_logits)
        V = tf.nn.tanh(v_logits)

        return P, V, p_logits, v_logits

    def train(self, training_boards, training_labels_p, training_labels_v, model_path = None):
        """Train the res net model with results from each iteration of self play.
        Args:
            model_path: location where we want the final model to be saved,
                None if we don't want to save the model
            training_boards: an array of board grids
            training_labels_p: an dim x dim + 1 array indicating the policy for current board
            training_labels_v: an array of results indicating who is the winner
        Returns:
            None, but a model is saved at the model_path
        """
        training_boards = np.array([self.convert_to_resnet_input(board) for board in training_boards])
        if model_path:
            saver = tf.train.Saver(max_to_keep=500)

        self.sess.run(
            self.train_op,
            feed_dict={self.x: training_boards, self.yp: training_labels_p, self.yv: training_labels_v}
        )
        if model_path:
            save_path = saver.save(self.sess, model_path)

    def generate_mini_batches(self, batch_size, train_data, train_labels_p, train_labels_v):
        """ Yield mini batches in tuples from the original dataset with a specified batch size
        Params: 
            batch_size: number of training data in a sample
            train_data: all of train boards after shuffling
            train_labels_p: all of train labels [None, dim x dim + 1] after shuffling
            train_labels_v: all of train labels [None, 1] after shuffling
        Return:
            A generator yielding each mini batch([batch_num, dim x dim], [batch_num, dim x dim + 1], [batch_num, 1])
        Notes:
            the last data not divisible by mini-batch is thrown away
        """
        print("Shuffling data...",)
        train_data_num = len(train_data)
        idx = np.random.permutation(train_data_num)
        train_data = train_data[idx]
        train_labels_p = train_labels_p[idx]
        train_labels_v = train_labels_v[idx]
        print("Done!")
        for i in range(int(train_data_num / batch_size)):
            start_slice_index = i * batch_size
            end_slice_index = (i + 1) * batch_size
            yield (train_data[start_slice_index:end_slice_index],
                   train_labels_p[start_slice_index:end_slice_index],
                   train_labels_v[start_slice_index:end_slice_index])

    def fake_train(self, model_path, training_data_num = 1000, test_data_num = 50):
        """This function is used for testing the resNet independent of the mcts and self play code.
        The goal is to teach the resNet to count the number of black and white stones on a board.
        This code is used in test only.
        """
        batch_size = 500
        epoch_num = 100
        fake_x, fake_yp, fake_yv = self.generate_fake_data(training_data_num)
        
        #split into batches
        for epoch in range(epoch_num):
            losses = []
            accuracies = []
            gradients = []
            for batch_data, batch_p, batch_v in self.generate_mini_batches(batch_size, fake_x, fake_yp, fake_yv):
                _, batch_loss, batch_acc, batch_gradient = self.sess.run(
                        [self.train_op, self.loss, self.accuracy, self.gradient],
                        feed_dict={self.x: batch_data, self.yp: batch_p, self.yv:batch_v}
                    )
                
                losses.append(batch_loss)
                accuracies.append(batch_acc)
              
            print("Loss for epoch {} is {}".format(epoch, np.mean(losses)))
            print("Accuracy for epoch {} is {}".format(epoch, np.mean(accuracies)))

        #Testing fake data
        #Achieved 96% test accuracy for counting, using 1000 training data, 500 batch size and 100 epochs
        print("Start testing")
        test_fake_x, test_fake_yp, test_fake_yv = self.generate_fake_data(test_data_num)
        test_loss, test_acc = self.sess.run(
            [self.loss, self.accuracy],
            feed_dict={self.x: test_fake_x, self.yp: test_fake_yp, self.yv:test_fake_yv}
        )
        print("For {} testing data points, test loss is {}, test accuracy is {}". format(test_data_num, np.mean(test_loss), test_acc))

    def predict(self, board):
        """Given a board. predict (p,v) according to the current res net
        Args:
            board: current board including the current player and stone distribution
            model_path: None if we used the model previously trained for this object, 
                otherwise restore the model from this path used in real time playing
        Returns:
            p_dist: the probability distribution dictionary of the next move according to current policy. including pass
            v: the probability of winning from this board.
        """

        p_dist = {}
        input_to_nn = self.convert_to_resnet_input(board)

        p = self.sess.run(self.yp_, feed_dict={self.x: [input_to_nn]})
        v = self.sess.run(self.yv_, feed_dict={self.x: [input_to_nn]})

        p = p[0]
        p_dist[(-1, -1)] = p[self.go_board_dimension**2]
        for r in range(self.go_board_dimension):
            for c in range(self.go_board_dimension):
                p_dist[(r, c)] = p[r * self.go_board_dimension + c]
        
        return p_dist, v

    def convert_to_resnet_input(self, original_board):
        converted_grid = np.array(self.convert_to_one_hot_go_boards(original_board.board_grid))   
        player_layer = np.ones((self.go_board_dimension , self.go_board_dimension , 1)) * original_board.player
        append_result = np.append(converted_grid, player_layer, axis = 2)
        return append_result

    def convert_to_one_hot_go_boards(self, original_board_grid):
        """Convert the format of the go board from a dim by dim 2d array to a dim by dim by 3 3d array.
        This is used before feed the boards into the neural net.
        Args:
            original_board_grid: a board_dimension x board_dimension array, each element can be -1 (white), 0 (empty) or 1 (black).
        Returns:
            flattend_board: a board_dimension x board_dimension array x 3 one hot vector
        """
        board_dim = len(original_board_grid)
        return [[self.helper_convert_to_one_hot(original_board_grid[r][c]) for c in range(board_dim)] for r in range(board_dim)]

    def helper_convert_to_one_hot(self, element):
        """ Transformation 1 -> [0,0,1]; 0->[0,1,0], -1 -> [-1,0,1]
        Args:
            element: number to be transformed into an array, has to be -1, 0 or 1
        Return:
            array of size 3 the element is transformed to
        """
        transformation = {
            -1: [1,0],
            0:  [0,0],
            1:  [0,1]
        }
        return transformation[element]

    def generate_fake_data(self, training_data_num):
        """Generate fake boards and counts the number of black and white stones as labels.
        Args:
            training_data_num: the number of fake training data we want to generate
        Returns:
            Xs: a list of training boards
            Ys: a list of training labels, each label is: 
            [a size 26 one hot arrayindicating the count the total number stones, layer indicating current player(1) or opponent(-1) has more stones,
                return 1 if they have the equal number of stones]
        """
        go_board_dimension = self.go_board_dimension
        Xs = []
        total_stone_count_vectors = []
        player_with_more_stones_all = [] #1 if current player has more stones, -1 otherwise

        options = [-1, 0, 1] #white empty black
        for i in range(training_data_num):
            black_stone_count = 0
            white_stone_count = 0

            player = random.choice([-1, 1])
            board_grid = [[random.choice(options) for c in range(go_board_dimension)] for r in range(go_board_dimension)]
            for r in range(go_board_dimension):
                for c in range(go_board_dimension):
                    if board_grid[r][c] == -1:
                        white_stone_count += 1
                    elif board_grid[r][c] == 1:
                        black_stone_count += 1
            board = go_board.go_board(go_board_dimension, player, board_grid)
            Xs.append(self.convert_to_resnet_input(board))

            total_stone_count = black_stone_count + white_stone_count
            total_stone_count_vector = [0]*(go_board_dimension*go_board_dimension+1)
            total_stone_count_vector[total_stone_count] = 1

            if player == 1:
                if black_stone_count > white_stone_count:
                    player_with_more_stones = float(1)
                elif black_stone_count < white_stone_count:
                    player_with_more_stones = float(-1)
                else:
                    player_with_more_stones = float(0)
            elif player == -1:
                if black_stone_count < white_stone_count:
                    player_with_more_stones = float(1)
                elif black_stone_count > white_stone_count:
                    player_with_more_stones = float(-1)
                else:
                    player_with_more_stones = float(0)
        
            total_stone_count_vectors.append(total_stone_count_vector)
            player_with_more_stones_all.append([float(player_with_more_stones)])

        return np.array(Xs), np.array(total_stone_count_vectors), np.array(player_with_more_stones_all)
