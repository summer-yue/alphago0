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
            self.moves: track the history of the nodes played and their corresponding pi
                pi is the probabilty for next moves according to the MCTS simulations
        """
        self.nn = nn
        self.current_board = starting_board
        self.moves = {}
        self.history_boards = [] #Records all the board config played in this self play session

    def find_most_likely_move(self, policy):
        """Return the move tuple that represents the best move according to a policy
        Args:
            policy: an array of tuples (move, probabilities) with length
            board_dimension ** 2 + 1 (pass)
        Return:
            best_move: (r, c) tuple representing the best nect move position
        """
        best_move = None
        highest_prob = 0

        for (move, p) in policy:
            if p > highest_prob:
                best_move = move
                highest_prob = p

        return best_move

    def play_one_move(self):
        """Use MCTS to calculate the pi for current node, 
           update self.moves and current node
           Return true if a move is made, return False if player decided to pass
           Returns:
                True if the player passed, False otherwise
        """

        ts_instance = mcts.MCTS(self.current_board, self.nn)
        next_move_policy = ts_instance.run_all_simulations()

        move = self.find_most_likely_move(next_move_policy)

        self.history_boards.append(self.current_board) #Save current board to history
        self.current_board = go_utils.make_move(self.current_board, move) #Update current board to board after move

        return move == (-1, -1)

    def play_till_finish(self):
        """Play until the game reaches a final state (2 passes happen one after another)
        Returns:
            new_training_data: A set of (board_grid, result) ready to be used as training labels
        """
        new_training_data = set()

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

        for history_board in history_boards:
            new_training_data.add(history_board.board_grid, winner)
        return new_training_data

