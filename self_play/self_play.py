import numpy as np
np.set_printoptions(threshold=np.nan)
from self_play.mcts import MCTS

class SelfPlay():
    """Algorithm plays against itself till the game ends and produce a set of (board, policy, result)
    Used as training data for the neural net.
    """
    def __init__(self, starting_board, nn, utils, simluation_number):
        """Initialize an instance of self play with a starting node
        Args:
            starting_board: a GameBoard instance representing the starting board
            nn: instance of current neural net model
            utils: GameUtils instance used during self play
            simluation_number: number of MCTS simulations needed to play one move
        Fields:
            self.nn: instance of neural net model used for this iteration of self play
            self.current_node: the current node during self play
            self.policies: track the history of the nodes played and their corresponding pi
                pi is the probabilty for next moves according to the MCTS simulations
        """
        self.utils = utils
        self.nn = nn
        self.simluation_number = simluation_number
        self.current_board = starting_board
        self.policies = np.empty(0)
        self.history_boards = np.empty(0) #Records all the board config played in this self play session

    def play_one_move(self):
        """Use MCTS to calculate the pi for current node, 
           update self.moves and current node
           Return true if a move is made, return False if player decided to pass
           Returns:
                True if the player passed, False otherwise
        """
        ts_instance = MCTS(self.current_board, self.nn, self.utils, self.simluation_number)
        new_board, move, policy = ts_instance.run_all_simulations(temp1 = 0.2, temp2 = 0.1, step_boundary=2)

        print("move is:", move)
        if len(self.policies) == 0:
            self.policies = policy
        else:
            self.policies = np.vstack((self.policies, policy))

        self.history_boards = np.append(self.history_boards, self.current_board) #Save current board to history
        self.current_board = new_board #Update current board to board after move

        return move == (-1, -1)

    def play_till_finish(self):
        """Play until the game reaches a final state (2 passes happen one after another)
        Returns:
            new_training_data: An array of board_grid ready to be used as training data
            self.policies: an array of result ready to be used as policy training labels
            new_training_labels_v: an array of result ready to be used as value training labels
        """
        move_num = 0
        #Cut the game if we played for too long
        while (not self.utils.is_game_finished(self.current_board)) and move_num <= self.current_board.board_dimension**2 * 2:
            self.play_one_move()
            move_num += 1
   
        boards_data = np.array([augment_board for history_board in self.history_boards for augment_board in history_board.generate_augmented_boards()])
        reversed_boards_data = [b.reverse_board_config() for b in boards_data]

        winner, _ = self.utils.evaluate_winner(self.current_board.board_grid)
        #corresponding winner for each history board from current perspective
        new_training_labels_v = np.array([[winner] if history_board.player != self.current_board.player else [-winner] \
            for history_board in self.history_boards])
        new_training_labels_v = np.repeat(new_training_labels_v, 5, axis=0)
        new_training_labels_v = np.append(new_training_labels_v, new_training_labels_v, axis=0)
        new_training_labels_p = np.repeat(np.array(self.policies), 5, axis=0)
        new_training_labels_p = np.append(new_training_labels_p, new_training_labels_p, axis=0)

        # print(self.policies)
        # print()
        print(new_training_labels_p)
        print("current board is ", self.current_board)
        print("a game is finished and winner is:", winner)
        print()
        print("self play 10 boards")
        for board in np.append(boards_data, reversed_boards_data):
            print(str(board))
        print()
        print("v labels")
        print(new_training_labels_v)
        print()
        print("p labels")
        print(new_training_labels_p)

        return np.append(boards_data, reversed_boards_data), new_training_labels_p, new_training_labels_v

