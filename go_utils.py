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

    #Invalid move because of Ko restrictions, this condition is checked before the liberty constraint
    if is_invalid_move_because_of_ko(board, move):
        return None

    #Invalid move if placed in a spot that forms a group of stones with no liberty
    #Try making the move on a copied board and check if anything illegal is detected
    board.board_grid[r][c] = board.player

    #Remove the stones captured because of this move
    #Remove the groups of the stones that belong to the opponent directly next to current move
    #top
    if r > 0 and board.board_grid[r - 1][c] == -board.player:
        board.board_grid = remove_pieces_if_no_liberty((r - 1, c), board.board_grid)
    #bottom
    if r < board.board_dimension - 1 and board.board_grid[r + 1][c] == -board.player:
        board.board_grid = remove_pieces_if_no_liberty((r + 1, c), board.board_grid)
    #left
    if c > 0 and board.board_grid[r][c - 1] == -board.player:
        board.board_grid = remove_pieces_if_no_liberty((r, c - 1), board.board_grid)
    #right
    if c < board.board_dimension - 1 and board.board_grid[r][c + 1] == -board.player:
        board.board_grid = remove_pieces_if_no_liberty((r, c + 1), board.board_grid)

    #Invalid move if current move would cause the current connected group to have 0 liberty
    if count_liberty_for_one_stone(board.board_grid, move) == 0:
        return None

    #After a move is successfully made, update the board to reflect that and return
    board.game_history = board.game_history + [(board.player, r, c)]
    board.player = -board.player
    return board

def remove_pieces_if_no_liberty(position, board_grid):
    """Look at the pieces that form the group of position
    If the group has no liberty, remove and return new grid
    Args:
        position: (r, c) tuple indicating the position of which piece we want to find the group for
        board_grid: 2d array representation of the board
    Returns:
        new_board_grid: the new grid after removal of elements
    """
    if count_liberty_for_one_stone(board_grid, position) == 0:
        pieces_in_group = find_pieces_in_group(position, board_grid)
    else:
        return board_grid
    for (r, c) in pieces_in_group:
        board_grid[r][c] = 0
    return board_grid

def find_pieces_in_group(position, board_grid):
    """For a potential move, find the surrounding pieces of the same player's stone
    that can form a group of stones
    Args:
        position: (r, c) tuple indicating the position of which piece we want to find the group for
        board_grid: 2d array representation of the board
    Returns:
        a set of position tuples inidcaitng the pieces in the same group, including the originally position
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

def find_adjacent_positions_with_opposite_color(position, board_grid):
    """Find the opposing stones directly to the right, left, top or down of 
    a stone on a board. Return a list of them in the order of up, down, left, right.
    Args:
        position: (r, c) the position we're trying to find neighbors for
        board_grid: 2d array representation of the board
    Returns:
        an set of positions tuples that are immediately next to the original position with the opposite color
    """
    neighbors = set()
    (r, c) = position
    player = board_grid[r][c]
    board_dimension = len(board_grid)

    #top
    if r > 0 and board_grid[r - 1][c] == -player:
        neighbors.add((r - 1, c))
    #bottom
    if r < board_dimension - 1 and board_grid[r + 1][c] == -player:
        neighbors.add((r + 1, c))
    #left
    if c > 0 and board_grid[r][c - 1] == -player:
        neighbors.add((r, c - 1))
    #right
    if c < board_dimension - 1 and board_grid[r][c + 1] == -player:
        neighbors.add((r, c + 1))
    return neighbors

def count_liberty_for_one_stone(board_grid, position):
    """Count the liberties associated with one stone on the board,
    in other words, the adjacent empty crosses
    Args:
        board_grid: 2d array representation of the board and stone distributions
        position: used to identify the stone that we're looking for liberties for
    Returns:
        The liberty number associated with the stone
    """
    (r, c) = position
    board_dimension = len(board_grid)
    liberty = 0

    #top
    if r > 0 and board_grid[r - 1][c] == 0:
        liberty += 1
    #bottom
    if r < board_dimension - 1 and board_grid[r + 1][c] == 0:
        liberty += 1
    #left
    if c > 0 and board_grid[r][c - 1] == 0:
        liberty += 1
    #right
    if c < board_dimension - 1 and board_grid[r][c + 1] == 0:
        liberty += 1

    return liberty

def count_liberty(board_grid, position):
    """Count the remaining liberties of the connected group of a particular position
    Args:
        board_grid: 2d array representation of the board and stone distributions
        position: used to identify the connected group that we're looking for liberties for
    Returns:
        The remaining liberty number as an integer
    """
    group = find_pieces_in_group(position, board_grid)
    total_liberties = 0
    for stone in group:
        total_liberties += count_liberty_for_one_stone(board_grid, stone)
    return total_liberties

def is_invalid_move_because_of_ko(board, move):
    """Detect if a move if invalid due to the ko condition
    1. the current stone is surrounded by opponents in all directions not on the border (no neighbor with the same color and no liberty)
    2. and the for all of the adjacent opponent stones, only one of them has no liberty after this move
    3. and the one stone from 2 is not connected to any other stones
    4. and the stone with no liberty from 2's position was played in the last move
    Args:
        board: current board config including whose turn it is
        move: (row, col) tuple indicating the the location of the current move
    Returns:
        Boolean value indicating if the move is invalid because it is a Ko invalid move
    """
    if find_adjacent_positions_with_same_color(move, board.board_grid) == set() and count_liberty(board.board_grid, move) == 0:
        #Condition one passes
        board_copy = board.copy()
        (r, c) = move
        #Place the stone temporarily on the board_copy board without considering any go rules
        board_copy.board_grid[r][c] = board_copy.player
        dead_neighbor_num = 0
        neighbor_opponent_dead_due_to_move = None
        for neighbor_opponent in find_adjacent_positions_with_opposite_color(move, board.board_grid):
            if count_liberty(board_copy.board_grid, neighbor_opponent) == 0:
                dead_neighbor_num += 1
                neighbor_opponent_dead_due_to_move = neighbor_opponent
        if (dead_neighbor_num == 1):
            #Condition 2 passes
            if len(find_pieces_in_group(neighbor_opponent_dead_due_to_move, board_copy.board_grid)) == 1:
                #Condition 3 passes
                last_move = board.game_history[len(board.game_history) - 1]
                #The logic for this check is due to the 3 element tuple format stored in history and the two element tuple in neighbor_opponent_dead_due_to_move
                if neighbor_opponent_dead_due_to_move[0] == last_move[1] and neighbor_opponent_dead_due_to_move[1] == last_move[2] :
                    #Condition 4 passes
                    return True
    return False

def is_move_pass(move):
    """Check it the move tuple means passs
    Args:
        move: tuple indicating the location of the move, or (-1, -1) inidcating a pass
    Returns:
        a boolean value indiating if the move is a pass
    """
    return move == (-1, -1)

def is_move_in_board(move, board_dimension):
    """Check if a move is within the boundary of the Go board grid
    Args:
        move: (r, c) tuple indicating the position of the considered move
        board_dimension: the vertical and horizontal dimension of the Go board
    Returns:
        boolean value indicating if the move is inside of the board range
    """
    (r,c) = move
    if r < 0 or c < 0 or r >= board_dimension or c >= board_dimension:
        return False
    return True