import pygame
import go_board as gb
import go_utils
import go_utils_terminal
from pygame.locals import *

BOARD_DIM = 2 # Define an x by x board

# Define colors
BLACK  = (0, 0, 0)
WHITE  = (245, 245, 245)
RED    = (133, 42, 44)
YELLOW = (208, 176, 144)
GREEN  = (26, 81, 79)

PLAYER_BLACK = 1
PLAYER_WHITE = -1
EMPTY = 0

# Define grid globals
WIDTH = 20 # Width of each square on the board
MARGIN = 1 # How thick the lines are
PADDING = 20 # Distance between board and border of the window
DOT = 4 # Number of dots
BOARD = (WIDTH + MARGIN) * (BOARD_DIM - 1) + MARGIN # Actual width for the board
GAME_WIDTH = BOARD + PADDING * 2
GAME_HIGHT = GAME_WIDTH + 100

class Go:
    def __init__(self):
        self.go_board = gb.go_board(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        pygame.init()
        pygame.font.init()
        self._display_surf = pygame.display.set_mode((GAME_WIDTH,GAME_HIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

        pygame.display.set_caption('Go')

        self._running = True
        self._playing = False
        self._win = False
        self.lastPosition = [-1,-1]

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            if self.mouse_in_botton(pos):
                if not self._playing:
                    self.start()
                else:
                    self.surrender()
                    self.go_board.flip_player()

            elif self._playing:
                c = (pos[0] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)
                r = (pos[1] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)

                if 0 <= r < BOARD_DIM and 0 <= c < BOARD_DIM:
                    _, self.go_board = go_utils.make_move(board=self.go_board, move=(r, c))
                    self.print_winner()
                    self.lastPosition = self.go_board.get_last_position()
             
        # print(self.go_board)
        # print()
    
    def on_render(self):
        self.render_go_piece()
        self.render_last_position()
        self.render_game_info()
        self.render_button()
        pygame.display.update()

    def on_cleanup(self):
        pygame.quit()


    def on_execute(self):   
        while( self._running ):
            self.go_board_init()
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()

    def start(self):
        self._playing = True
        self.lastPosition = [-1,-1]
        self.go_board = gb.go_board(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        self._win = False

    def surrender(self):
        self._playing = False
        self._win = True

    def go_board_init(self):
        self._display_surf.fill(YELLOW)
        # Draw black background rect for game area
        pygame.draw.rect(self._display_surf, BLACK,
                         [PADDING,
                          PADDING,
                          BOARD,
                          BOARD]) 

        # Draw the grid
        for row in range(BOARD_DIM - 1):
            for column in range(BOARD_DIM - 1):
                pygame.draw.rect(self._display_surf, YELLOW,
                                 [(MARGIN + WIDTH) * column + MARGIN + PADDING,
                                  (MARGIN + WIDTH) * row + MARGIN + PADDING,
                                  WIDTH,
                                  WIDTH])

        # dots
        # points = [(3,3),(11,3),(3,11),(11,11),(7,7)]
        # for point in points:
        #     pygame.draw.rect(self._display_surf, BLACK,
        #                     (PADDING + point[0] * (MARGIN + WIDTH) - DOT // 2,
        #                      PADDING + point[1] * (MARGIN + WIDTH) - DOT // 2,
        #                      DOT,
        #                      DOT),0)


    def mouse_in_botton(self,pos):
        """ Check if mouse is in the button and return a boolean value
        """
        if GAME_WIDTH // 2 - 50 <= pos[0] <= GAME_WIDTH // 2 + 50 and GAME_HIGHT - 50 <= pos[1] <= GAME_HIGHT - 20:
           return True
        return False

    def render_button(self):
        color = GREEN if not self._playing else RED
        info = "Start" if not self._playing else "Surrender"

        pygame.draw.rect(self._display_surf, color, 
                         (GAME_WIDTH // 2 - 50, GAME_HIGHT - 50, 100, 30))

        info_font = pygame.font.SysFont('Helvetica', 18)
        text = info_font.render(info, True, WHITE)
        textRect = text.get_rect()
        textRect.centerx = GAME_WIDTH // 2
        textRect.centery = GAME_HIGHT - 35
        self._display_surf.blit(text, textRect)

    def render_game_info(self):
        #current player color
        color = BLACK if self.go_board.player == PLAYER_BLACK else WHITE
        center = (GAME_WIDTH // 2 - 60, BOARD + 60)
        radius = 12
        pygame.draw.circle(self._display_surf, color, center, radius, 0)

        info = "You Win" if self._win else "Your Turn"
        info_font = pygame.font.SysFont('Helvetica', 24)
        text = info_font.render(info, True, BLACK)
        textRect = text.get_rect()
        textRect.centerx = self._display_surf.get_rect().centerx + 20
        textRect.centery = center[1]
        self._display_surf.blit(text, textRect)

    def render_go_piece(self):
        """ Render the Go stones on the board according to self.go_board
        """
        # print('rendering go pieces')
        # print(self.go_board)
        for r in range(BOARD_DIM):
            for c in range(BOARD_DIM):
                center = ((MARGIN + WIDTH) * c + MARGIN + PADDING,
                          (MARGIN + WIDTH) * r + MARGIN + PADDING)
                if self.go_board.board_grid[r][c] != EMPTY:
                    color = BLACK if self.go_board.board_grid[r][c] == PLAYER_BLACK else WHITE
                    pygame.draw.circle(self._display_surf, color,
                                       center,
                                       WIDTH // 2 - MARGIN,
                                       0)


    def render_last_position(self):
        """ Render a red rectangle around the last position
        """
        if self.lastPosition[0] > 0 and self.lastPosition[1] > 0:
            pygame.draw.rect(self._display_surf,RED,
                             ((MARGIN + WIDTH) * self.lastPosition[0] + (MARGIN + WIDTH) // 2, 
                              (MARGIN + WIDTH) * self.lastPosition[1] + (MARGIN + WIDTH) // 2, 
                              (MARGIN + WIDTH), 
                              (MARGIN + WIDTH)),1)

    def print_winner(self):
        winner, winning_by_points = go_utils_terminal.evaluate_winner(self.go_board.board_grid)
        if winner == PLAYER_BLACK:
            print ("Black wins by " + str(winning_by_points))
        else:
            print ("White wins by " + str(winning_by_points))

if __name__ == "__main__" :
    go = Go()
    go.on_execute()