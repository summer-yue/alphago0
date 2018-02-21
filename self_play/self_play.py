from self_play import mcts
from go import go_utils_terminal
import numpy as np

class self_play():
    """Algorithm plays against itself till the game ends and produce a set of (board, policy, result)
    Used as training data for the res net.
    """
    def __init__(self, starting_board, nn):
        """Initialize an instance of self play with a starting node
        Args:
            starting_board: a go_board instance representing the starting board
            nn: instance of current AlphaGo0 model
        Fields:
            self.nn: instance of AlphaGo0 model used for this iteration of self play
            self.current_node: the current node during self play
            self.policies: track the history of the nodes played and their corresponding pi
                pi is the probabilty for next moves according to the MCTS simulations
        """
        self.nn = nn
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

        ts_instance = mcts.MCTS(self.current_board, self.nn)
        new_board, move, policy = ts_instance.run_all_simulations()

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
        passed_once = False
        game_over = False
        while not game_over:
            is_passed = self.play_one_move()
        
            if is_passed:
                if passed_once:
                    game_over = True
                else:
                    passed_once = True
            else:
                passed_once = False

        winner, _ = go_utils_terminal.evaluate_winner(self.current_board.board_grid)
        print("len(self.history_boards)", len(self.history_boards))
        new_training_labels_v = np.array([[winner]]*len(self.history_boards))

        print("a game is finished and winner is:", winner)
        print(self.policies)
        print(new_training_labels_v)
        return np.array([history_board for history_board in self.history_boards]), self.policies, new_training_labels_v

