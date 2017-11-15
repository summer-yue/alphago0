#This file contains the Go related utility functions at the terminal state of the game
#For example, deciding what pieces and captured, deciding the winner and counting scores.
from collections import deque

BLACK = 1
WHITE = -1

def evaluate_winner(board):
    """Evaluate who is the winner of the board configuration
    Args:
        board: current board: including dimensionm grid, player and history
    Returns:
        player who won the game. 1: black or -1: white
    """
    new_board = remove_captured_stones(board)
    black_score = 0
    white_score = 0

    #Count the territories that don't have stones on them
    for (player, empty_pieces_positions) in find_connected_empty_pieces(new_board):
        if (player == BLACK):
            black_score += len(empty_pieces_positions)
        if (player == WHITE):
            white_score += len(empty_pieces_positions)
    
    #Count the remaining stones for each side
    black_score += count_stones(board.board_grid, player=BLACK)
    white_score += count_stones(board.board_grid, player=WHITE)

    if black_score > white_score:
        return BLACK
    else:
        return WHITE

def find_connected_empty_pieces(board_grid):
    """Find the groups of connected empty pieces on the board and infer whose territory they are
    using depth first search
    Args:
        board_grid: 2d array representation of the board
    Returns:
        a list of tuple, tuple's first element indicates which player's territory the empty pieces belong to
        second element is a list of tuples indicating these empty pieces' locations(1, [(2,2),(3,3)])
    """
    connected_empty_pieces_and_player = [] #Value to be returned
    border_stone_nums = {}
    border_stone_nums[BLACK] = 0
    border_stone_nums[WHITE] = 0 #This tracks the distribution of stones bordering this empty territory

    visited = {} #A dictorionary of tuples tracking if the positions are already added into a group
    current_group = []

    #Initialize visited, all empty spaces is set to false because we haven't seen any of them
    for i in range(len(board_grid)):
        for j in range(len(board_grid[0])):
            if board_grid[i][j] == 0:
                visited[(i, j)] = False

    current_position = get_next_empty_space_to_visit(visited)
    #Outer loop makes sure that all 
    while current_position != None:
        
        frontier = deque()
        frontier.append(current_position)
        while (len(frontier) > 0): #frontier is not empty
            current_position = frontier.popleft()  
            current_group.append(current_position)
            visited[current_position] = True

            #Go to its neighbors, if it's a stone, update border_stone_nums, otherwise add to frontier
            for neighbor in find_direct_neighbors(current_position, board_grid):
                (neighbor_r, neighbor_c) = neighbor
                if board_grid[neighbor_r][neighbor_c] == 0:
                    if visited[neighbor] == False and neighbor not in frontier:
                        frontier.append(neighbor)
                else:
                    border_stone_nums[board_grid[neighbor_r][neighbor_c]] += 1

        #Put the group we found in the returning array
        if border_stone_nums[BLACK] > 0 and border_stone_nums[WHITE] == 0:
            connected_empty_pieces_and_player.append((BLACK, current_group))
        elif border_stone_nums[WHITE] > 0 and border_stone_nums[BLACK] == 0: 
            connected_empty_pieces_and_player.append((WHITE, current_group)) 
        else:
            connected_empty_pieces_and_player.append((0, current_group))
        current_group = []
        border_stone_nums[BLACK] = 0
        border_stone_nums[WHITE] = 0
        current_position = get_next_empty_space_to_visit(visited)

    return connected_empty_pieces_and_player

def get_next_empty_space_to_visit(visited):
    """Return the first position not visited. None if all elements are visited
    Args:
        visited: a dictionary position->boolean indicating if the position has been visited
    Returns:
        the first position not visited. None if all elements are visited
    """
    for position in visited:
        if not visited[position]:
            return position
    return None

def remove_captured_stones(board_grid):
    """Remove the captured/dead stones from the Go board and return a new board
    Args:
        board_grid: 2d array representation of the board
    Returns:
        new_board with the removed stones
    TODO
    """
    pass

def count_stones(board_grid, player):
    """Count the total number of stones on the board that belong to a certain player
    Returns:
        integer representing the total number of stones of this particular player
    """
    count = 0
    for i in range(len(board_grid)):
        for j in range(len(board_grid[0])):
            if board_grid[i][j] == player:
                count += 1
    return count

def find_direct_neighbors(position, board_grid):
    """Find the stones directly to the right, left, top or down of 
    a stone on a board. Return a list of them in the order of up, down, left, right.
    Args:
        position: (r, c) the position we're trying to find neighbors for
        board_grid: 2d array representation of the board
    Returns:
        an set of positions tuples that are immediately next to the original position
    """
    neighbors = set()
    (r, c) = position
    player = board_grid[r][c]
    board_dimension = len(board_grid)

    #top
    if r > 0:
        neighbors.add((r - 1, c))
    #bottom
    if r < board_dimension - 1:
        neighbors.add((r + 1, c))
    #left
    if c > 0:
        neighbors.add((r, c - 1))
    #right
    if c < board_dimension - 1:
        neighbors.add((r, c + 1))
    return neighbors
