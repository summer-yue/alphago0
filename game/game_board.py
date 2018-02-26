import copy

from abc import ABC, abstractmethod

class GameBoard(ABC):
    """The general class for a grid based two player game board.
    """
    def __init__(self, board_dimension, player, board_grid = [], game_history = None):
        """Initialize a game board
        Args:
            board_dimension: the dimension of our game board
            player: 1 (player who plays first) or -1 indicating the two players
            board_grid: the original grid of the board, 2d array of 1 - black,
                -1: white and 0: not occupied
            game_history: the original order in which player played the game, a list of move tuples
                (player, r, c) such as (-1, 4, 6), or (1, -1, -1) means black passes a move
        """
        self.board_dimension = board_dimension
        self.player = player
        if len(board_grid) > 0:
            self.board_grid = board_grid
            self.game_history = game_history
            # TODO: check if game history matches the current board.
        else:
            self.board_grid = [[0 for i in range(board_dimension)] for y in range(board_dimension)]
            self.game_history = []

    def flip_player(self):
        """Update the player to the other player, 'b' to 'w' or 'w' to 'b'
        The two players are represented by 1 (the one that moves first) and -1.
        """
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def generate_augmented_boards(self):
        """augment the training data using the flipped version and rotated version of the board itself
        """
        for i in range(4):
            # rotate counterclockwise
            new_board_grid = np.rot90(self.board_grid, i + 1)
            yield GameBoard(self.board_dimension, self.player, board_grid = new_board_grid, game_history = None)
                  
        # flip horizontally
        new_board_grid = np.fliplr(self.board_grid)
        yield GameBoard(self.board_dimension, self.player, board_grid = new_board_grid, game_history = None)

    def add_move_to_history(self, r, c):
        """Add move (r, c) to the game_history field of the class
        r is the row number ranged from 0 to self.board_dimension-1
        c is the col number ranged from 0 to self.board_dimension-1
        """
        self.game_history.append((self.player, r, c))

    def get_last_position(self):
        """Get the [r, c] position frmo the last moved,
        taken from the last slot on game_history
        Used for the GUIs to show the last moves
        """
        (player, r, c) = self.game_history[-1]
        return [r,c]

    def copy(self):
        """Make a deep copy of the board object
        Returns:
            copy of the board object
        """
        copy_board_grid = copy.deepcopy(self.board_grid)
        copy_game_history = copy.deepcopy(self.game_history)
        return GameBoard(self.board_dimension, self.player, copy_board_grid, copy_game_history)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__) or issubclass(other.__class__, self.__class__) or issubclass(self.__class__, other.__class__):
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
        return str(self.board_dimension) + "x" + str(self.board_dimension) + " game board\n" \
            + "with current player " + str(self.player) + "\n with current grid" \
            + str(self.board_grid) + "\n with game history" + str(self.game_history)
