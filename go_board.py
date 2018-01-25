import copy 

class go_board():
    def __init__(self, board_dimension, player, board_grid = None, game_history = None):
        """Initialize a go board
        Args:
            board_dimension: the dimension of our go board
            player: 1 or -1 indicating black and white players
            board_grid: the original grid of the board, 2d array of 1 - black,
                -1: white and 0: not occupied
            game_history: the original order in which player played the game, a list of move tuples
                such as (-1, 4, 6), (1, -1, -1) means black passes a move
        """
        self.board_dimension = board_dimension
        self.player = player
        if board_grid != None:
            self.board_grid = board_grid
            self.game_history = game_history
            # TODO: check if game history matches the current board.
        else:
            self.board_grid = [[0 for i in range(board_dimension)] for y in range(board_dimension)]
            self.game_history = []

    def flip_player(self):
        """Update the player to the other player, 'b' to 'w' or 'w' to 'b'
        """
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def add_move_to_history(self, r, c):
        self.game_history.append((self.player, r, c))

    def get_last_position(self):
        (player, r, c) = self.game_history[-1]
        return [r,c]

    def copy(self):
        """Make a deep copy of the go board
        Returns:
            copy of the go board
        """
        copy_board_grid = copy.deepcopy(self.board_grid)
        copy_game_history = copy.deepcopy(self.game_history)
        return go_board(self.board_dimension, self.player, copy_board_grid, copy_game_history)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.board_dimension == other.board_dimension \
                and self.player == other.player \
                and self.board_grid == other.board_grid \
                and self.game_history == other.game_history
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __str__(self):
        """Define a more human friendly print for boards"""
        return str(self.board_dimension) + "x" + str(self.board_dimension) + " board\n" \
            + "with current player " + str(self.player) + "\n with current grid" \
            + str(self.board_grid) + "\n with game history" + str(self.game_history)
