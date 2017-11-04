def is_valid_move(board, move):
    """Check if a move (row,col) on a board is valid
    Args:
        board: current board: including dimensionm grid, player and history
        move: (r, c) tuple indicating the position of the considered move
    """

    #Pass (-1, -1) is a valid move
    if is_move_pass(move):
        return True

    #Not valid if placed outside of a board
    if not is_move_in_board(move, board.board_dimension):
        return False

    return True

def pieces_captured(board, move):
    pass

def is_ko(board, move):
    pass

def is_move_pass(move):
    """Check it the move tuple means passs
    """
    return move == (-1, -1)

def is_move_in_board(move, board_dimension):
    """Check if a move is within the boundary of the Go board grid
    Args:
        move: (r, c) tuple indicating the position of the considered move
        board_dimension: the vertical and horizontal dimension of the Go board
    """
    (r,c) = move
    if r < 0 or c < 0 or r >= board_dimension or c >= board_dimension:
        return False
    return True