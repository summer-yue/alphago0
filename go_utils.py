from collections import deque

def make_move(board, move):
    """Make a move (row,col) on a go board
    Args:
        board: current board: including dimensionm grid, player and history
        move: (r, c) tuple indicating the position of the considered move
    Returns:
        new board config if the move was successfully placed
        None if move was invalid
    """

    #Pass (-1, -1) is a valid move
    if is_move_pass(move):
        board.game_history.append((board.player, -1, -1)) #Add a pass move to history
        board.flip_player() #The other player's turn
        return board

    #Not valid if placed outside of a board
    if not is_move_in_board(move, board.board_dimension):
        return None

    (r, c) = move
    #Invalid move if placed on top of another existing stone
    if board.board_grid[r][c] != 0:
        return None

    #Invalid move if placed in a spot that forms a piece that is completely surrounded
    #Try making the move on a copied board and check if anything illegal is detected

    return board

def find_pieces_in_group(position, board_grid):
    """For a potential move, find the surrounding pieces of the same player's stone
    that can form a group of stones
    Args:
        position: (r, c) tuple indicating the position of which piece we want to find the group for
        board_grid: 2d array representation of the board
    """
    queue = deque() #Track the frontier of positions to be visited
    group_members = set() #Stores the group members we already visited
    queue.append(position)

    while queue: #frontier is not empty
        current_pos = queue.popleft()
        if current_pos not in group_members:
            group_members.add(current_pos)

        #Iterate all the adjacent positions with the same color as current_pos and add to frontier
        #if we never visited it or have already added in the frontier
        for neighbor in find_adjacent_positions_with_same_color(current_pos, board_grid):
            if neighbor not in queue and neighbor not in group_members:
                queue.append(neighbor)
    return group_members

def find_adjacent_positions_with_same_color(position, board_grid):
    """Find the stones directly to the right, left, top or down of 
    a stone on a board. Return a list of them in the order of up, down, left, right.
    Args:
        position: (r, c) the position we're trying to find neighbors for
        board_grid: 2d array representation of the board
    Returns:
        an set of positions tuples that are immediately next to the original position with the same color
    """
    neighbors = set()
    (r, c) = position
    player = board_grid[r][c]
    board_dimension = len(board_grid)

    #top
    if r > 0 and board_grid[r - 1][c] == player:
        neighbors.add((r - 1, c))
    #bottom
    if r < board_dimension - 1 and board_grid[r + 1][c] == player:
        neighbors.add((r + 1, c))
    #left
    if c > 0 and board_grid[r][c - 1] == player:
        neighbors.add((r, c - 1))
    #right
    if c < board_dimension - 1 and board_grid[r][c + 1] == player:
        neighbors.add((r, c + 1))
    return neighbors

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