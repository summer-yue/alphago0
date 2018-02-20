from self_play import mcts

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
        self.policies = []
        self.history_boards = [] #Records all the board config played in this self play session

    def play_one_move(self):
        """Use MCTS to calculate the pi for current node, 
           update self.moves and current node
           Return true if a move is made, return False if player decided to pass
           Returns:
                True if the player passed, False otherwise
        """

        ts_instance = mcts.MCTS(self.current_board, self.nn)
        print("Starting to run the simulations")
        new_board, move, policy = ts_instance.run_all_simulations()

        print("move is:", move)

        self.policies.append(policy)
        self.history_boards.append(self.current_board) #Save current board to history
        self.current_board = new_board #Update current board to board after move

        return move == (-1, -1)

    def play_till_finish(self):
        """Play until the game reaches a final state (2 passes happen one after another)
        Returns:
            new_training_data: A list of board_grid ready to be used as training data
            self.policies: a list of result ready to be used as policy training labels
            new_training_labels_v: a list of result ready to be used as value training labels
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

        winner = go_utils_terminal.evaluate_winner(current_board.board_grid)

        new_training_labels_v = [winner] * len(history_board.sboard_grid)

        return history_board.board_grid, self.policies, new_training_labels_v

