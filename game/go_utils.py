from collections import deque

BLACK = 1
WHITE = -1

from game.game_utils import GameUtils

class GoUtils(GameUtils):
    """The go specific utility class.
    The current board contains the current player and other board infomration defined in go_board.
    The utilities are static functions called by GoUtils.function_name()
    """
    def __init__(self):
        pass

    def is_valid_move(self, board, move):
        """Check if a potential move for the go game is valid.
        Args:
            board: current board as a go_board object
            move: (r, c) tuple indicating the position of the considered move
        Returns:
            boolean variable indicating if the go move is valid.
        """

        #Pass (-1, -1) is a valid move
        if GoUtils._is_move_pass(move):
            return True

        board_copy = board.copy()
        #Not valid if placed outside of a board
        if not GoUtils._is_move_in_board(move, board_copy.board_dimension):
            return False

        (r, c) = move
        #Invalid move if placed on top of another existing stone
        if board_copy.board_grid[r][c] != 0:
            return False

        #Invalid move because of Ko restrictions, this condition is checked before the liberty constraint
        if GoUtils._is_invalid_move_because_of_ko(board_copy, move):
            return False

        #Invalid move if placed in a spot that forms a group of stones with no liberty
        board_copy.board_grid[r][c] = board_copy.player
        #Remove the groups of the stones that belong to the opponent directly next to current move
        #top
        if r > 0 and board_copy.board_grid[r - 1][c] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r - 1, c), board_copy.board_grid)
        #bottom
        if r < board_copy.board_dimension - 1 and board_copy.board_grid[r + 1][c] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r + 1, c), board_copy.board_grid)
        #left
        if c > 0 and board_copy.board_grid[r][c - 1] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r, c - 1), board_copy.board_grid)
        #right
        if c < board_copy.board_dimension - 1 and board_copy.board_grid[r][c + 1] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r, c + 1), board_copy.board_grid)

        #Invalid move if current move would cause the current connected group to have 0 liberty
        if GoUtils._count_liberty(board_copy.board_grid, move) == 0:
            return False
        return True

    def make_move(self, board, move):
        """Make a move (row,col) on a go board
        Args:
            board: current board: including dimension grid, player and history
            move: (r, c) tuple indicating the position of the considered move
        Returns:
            A tuple indicating board config and if the move was valid (boolean value)
            new board config if the move was successfully placed
            old config if board is not updated
        """

        board_copy = board.copy()
        #Pass (-1, -1) is a valid move
        if GoUtils._is_move_pass(move):
            board_copy.add_move_to_history(-1, -1) #Add a pass move to history
            board_copy.flip_player() #The other player's turn
            return True, board_copy

        #Not valid if placed outside of a board
        if not GoUtils._is_move_in_board(move, board_copy.board_dimension):
            return False, board

        (r, c) = move

        #Invalid move if placed on top of another existing stone
        if board_copy.board_grid[r][c] != 0:
            return False, board

        #Invalid move because of Ko restrictions, this condition is checked before the liberty constraint
        if GoUtils._is_invalid_move_because_of_ko(board_copy, move):
            #print("Invalid because of Ko")
            return False, board

        #Invalid move if placed in a spot that forms a group of stones with no liberty
        board_copy.board_grid[r][c] = board_copy.player

        #Remove the stones captured because of this move
        #Remove the groups of the stones that belong to the opponent directly next to current move
        #top
        if r > 0 and board_copy.board_grid[r - 1][c] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r - 1, c), board_copy.board_grid)
        #bottom
        if r < board_copy.board_dimension - 1 and board_copy.board_grid[r + 1][c] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r + 1, c), board_copy.board_grid)
        #left
        if c > 0 and board_copy.board_grid[r][c - 1] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r, c - 1), board_copy.board_grid)
        #right
        if c < board_copy.board_dimension - 1 and board_copy.board_grid[r][c + 1] == -board_copy.player:
            board_copy.board_grid = GoUtils._remove_pieces_if_no_liberty((r, c + 1), board_copy.board_grid)

        #Invalid move if current move would cause the current connected group to have 0 liberty

        if GoUtils._count_liberty(board_copy.board_grid, move) == 0:
            return False, board
        
        #After a move is successfully made, update the board to reflect that and return
        board_copy.add_move_to_history(r, c)
        board_copy.flip_player()

        return True, board_copy

    def evaluate_winner(self, board_grid):
        """Evaluate who is the winner of the board configuration
        Args:
            board_grid: 2d array representation of the board
        Returns:
            player who won the game. 1: black or -1: white
            Absolute difference in game score
        """
        new_board = GoUtils._remove_captured_stones(board_grid)
        black_score = 0
        white_score = 0

        #Count the territories that don't have stones on them
        for (player, empty_pieces_positions) in GoUtils._find_connected_empty_pieces(new_board):
            if (player == BLACK):
                black_score += len(empty_pieces_positions)
            if (player == WHITE):
                white_score += len(empty_pieces_positions)
        
        #Count the remaining stones for each side
        black_score += GoUtils._count_stones(board_grid, player=BLACK)
        white_score += GoUtils._count_stones(board_grid, player=WHITE)

        # print("black score is " + str(black_score))
        # print("white score is " + str(white_score))
        if black_score > white_score:
            return BLACK, abs(black_score - white_score)
        else:
            return WHITE, abs(black_score - white_score)

    def is_game_finished(self, board):
        """Check if the go game is finished by looking at its game history
        The game is finished if the last two actions were both pass
        Args:
            board: current board as a go_board object
        Returns:
            Boolean variable indicating if the game is finished
        """
        if len(board.game_history) < 2:
            return False

        (_, r1, c1) = board.game_history[-1]
        (_, r2, c2) = board.game_history[-2]
        last_move = (r1, c1)
        second_to_last_move = (r2, c2)
        return (last_move == (-1, -1) and second_to_last_move == (-1, -1))

    @staticmethod
    def _remove_pieces_if_no_liberty(position, board_grid):
        """Look at the pieces that form the group of position
        If the group has no liberty, remove and return new grid
        Args:
            position: (r, c) tuple indicating the position of which piece we want to find the group for
            board_grid: 2d array representation of the board
        Returns:
            new_board_grid: the new grid after removal of elements
        """
        if GoUtils._count_liberty(board_grid, position) == 0:
            pieces_in_group = GoUtils._find_pieces_in_group(position, board_grid)
        else:
            return board_grid
        for (r, c) in pieces_in_group:
            board_grid[r][c] = 0
        return board_grid

    @staticmethod
    def _find_pieces_in_group(position, board_grid):
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
            for neighbor in GoUtils._find_adjacent_positions_with_same_color(current_pos, board_grid):
                if neighbor not in queue and neighbor not in group_members:
                    queue.append(neighbor)
        return group_members

    @staticmethod
    def _find_adjacent_positions_with_same_color(position, board_grid, current_player=None):
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
        if current_player != None:
            player = current_player
        else:
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

    @staticmethod
    def _find_adjacent_positions_with_opposite_color(position, board_grid):
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

    @staticmethod
    def _count_liberty_for_one_stone(board_grid, position):
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

    @staticmethod
    def _count_liberty(board_grid, position):
        """Count the remaining liberties of the connected group of a particular position
        Args:
            board_grid: 2d array representation of the board and stone distributions
            position: used to identify the connected group that we're looking for liberties for
        Returns:
            The remaining liberty number as an integer
        """
        group = GoUtils._find_pieces_in_group(position, board_grid)
        total_liberties = 0
        for stone in group:
            total_liberties += GoUtils._count_liberty_for_one_stone(board_grid, stone)
        return total_liberties

    @staticmethod
    def _is_invalid_move_because_of_ko(board, move):
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
        if GoUtils._find_adjacent_positions_with_same_color(move, board.board_grid, board.player) == set() and GoUtils._count_liberty(board.board_grid, move) == 0:
            #Condition one passes
            board_copy = board.copy()
            (r, c) = move
            #Place the stone temporarily on the board_copy board without considering any go rules
            board_copy.board_grid[r][c] = board_copy.player
            dead_neighbor_num = 0
            neighbor_opponent_dead_due_to_move = None

            for neighbor_opponent in GoUtils._find_adjacent_positions_with_opposite_color(move, board_copy.board_grid):
                if GoUtils._count_liberty(board_copy.board_grid, neighbor_opponent) == 0:
                    dead_neighbor_num += 1
                    neighbor_opponent_dead_due_to_move = neighbor_opponent
            if (dead_neighbor_num == 1):
                #Condition 2 passes
                if len(GoUtils._find_pieces_in_group(neighbor_opponent_dead_due_to_move, board_copy.board_grid)) == 1:
                    #Condition 3 passes
                    last_move = board.game_history[len(board.game_history) - 1]
                    #The logic for this check is due to the 3 element tuple format stored in history and the two element tuple in neighbor_opponent_dead_due_to_move
                    if neighbor_opponent_dead_due_to_move[0] == last_move[1] and neighbor_opponent_dead_due_to_move[1] == last_move[2] :
                        #Condition 4 passes
                        return True
        return False

    @staticmethod
    def _is_move_pass(move):
        """Check it the move tuple means passs
        Args:
            move: tuple indicating the location of the move, or (-1, -1) inidcating a pass
        Returns:
            a boolean value indiating if the move is a pass
        """
        return move == (-1, -1)

    @staticmethod
    def _is_move_in_board(move, board_dimension):
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

    @staticmethod
    def _find_connected_empty_pieces(board_grid):
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

        current_position = GoUtils._get_next_empty_space_to_visit(visited)
        #Outer loop makes sure that all 
        while current_position != None:
            
            frontier = deque()
            frontier.append(current_position)
            while (len(frontier) > 0): #frontier is not empty
                current_position = frontier.popleft()  
                current_group.append(current_position)
                visited[current_position] = True

                #Go to its neighbors, if it's a stone, update border_stone_nums, otherwise add to frontier
                for neighbor in GoUtils._find_direct_neighbors(current_position, board_grid):
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
            current_position = GoUtils._get_next_empty_space_to_visit(visited)

        return connected_empty_pieces_and_player

    @staticmethod
    def _get_next_empty_space_to_visit(visited):
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

    @staticmethod
    def _remove_captured_stones(board_grid):
        """Remove the captured/dead stones from the Go board and return a new board
        Args:
            board_grid: 2d array representation of the board
        Returns:
            new_board with the removed stones
        TODO implement
        """
        return board_grid

    @staticmethod
    def _count_stones(board_grid, player):
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

    @staticmethod
    def _find_direct_neighbors(position, board_grid):
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

