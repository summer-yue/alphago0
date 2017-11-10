# AlphaGo Zero Clone in Tensorflow (In Progress)
Reproducing the AlphaGo Zero Algorithms to play the game of Go without human knowledge.
Original Paper: https://www.nature.com/articles/nature24270.pdf

# Components
Training Residual Net <br />
Self Play <br />
Monte Carlo Tree Search <br />

## Step 1a:
Implement Go rules to detect invalid moves (go_utils.py)
## Step 1b:
Implement Go rules to decide the winner for the game according to the Go board (go_utils.py)
## Step 1c:
Construct a res net structure in tensorflow that trains based on labels (p, v) for a board. (alphgo_zero.py)
## Step 1d:
This can be done in a separate thread of our work.
Implement the game GUI(gui.py)

Implement the game GUI(gui.py)

The GUI displays how a game was played.

The first step of this GUI is human vs human mode (mode can be fed in as a parameter to start the GUI), initialized with a board object. When the user clicks on a cross on the board, it tries to make a move for a current user. Show some sort of warning message if the move is invalid. (This is known by make_move function returning None). Otherwise, if make_move is successful, make the move on the board by updating the board and the current player based on board_grid and board_player in the go_board object. The logic is implemented in the go_board class itself and go_utils. This is only to show the display.

Use Tkinter to display the current board. Inputs can be either a whole board configuration + last move made (used to mark the most recent move with a red dot), or simply the difference from the previous board configs as a list [(row_num1, col_num1, current_player1), (row_num2, col_num2, current_player2), ... ] + the last move made.

The second step of this GUI is to enable human vs machine mode, initialize with a initial board object. Assume that human makes the first move after the initialization of the board. The machine moves are made by the MCTS process.

The last step of this GUI is machine vs machine mode.

## Step 2a:
This depends on the completion of 1a and 1b but can start with 1a simultaneously.
Implement and test the Monte Carlo Tree Search Simulations to predict a next move.(mcts.py)

## Step 2b:
Implement self play (self_play.py)

## Step 3a:
Train the res net to implement raw play (alphago_zero.py)

## Step 3b:
Implement play with MCTS (alphago_zero.py)

# Authors
* [**Summer Yue**](https://github.com/yutingyue514)
* [**Ben Greenberg**](https://github.com/anchorwatt)
* [**Troy Wang**](https://github.com/TroyTianzhengWang)
