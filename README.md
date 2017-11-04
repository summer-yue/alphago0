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
