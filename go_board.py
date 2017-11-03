class go_board():
    def __init__(self, board_dimension, player, board_grid = None, game_history = None):
        """Initialize a go board
        Args:
            board_dimension: the dimension of our go board
            player: 'b' or 'w' indicating black and white players
            board_grid: the original grid of the board, 2d array of 'b' - black,
                'w': white and '0': not occupied
            game_history: the original order in which player played the game, a list of move tuples
                such as ('w', 4, 6), ('b', -1, -1) means black passes a move
        """
        self.board_dimension = board_dimension
        self.player = player
        if board_grid != None:
            self.board_grid = board_grid
            self.game_history = game_history
            # TODO: check if game history matches the current board.
        else:
            self.board_grid = [['0' for i in range(board_dimension)] for y in range(board_dimension)]
            self.game_history = []