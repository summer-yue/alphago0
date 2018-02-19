# AlphaGo Zero Clone in Tensorflow (In Progress)
Reproducing the AlphaGo Zero Algorithms to play the game of Go without human knowledge. We stick to all the Go rules (including Ko), but due to our computation resources constraint, we reduced the board size, which can be modified with the go_board_dimension parameter passed into the residual network.

Original Paper: https://www.nature.com/articles/nature24270.pdf
Python 3 is required to run this code.

# Components
Residual Net <br />
Self Play <br />
Monte Carlo Tree Search <br />
GUI implemented in PyGame <br />

# Instruction
1. Clone the repository
2. Install all packages required
3. For human-human mode, python go_gui.py
4. For human-machine mode, python human_machine_gui.py

# Authors
* [**Summer Yue**](https://github.com/yutingyue514)
* [**Ben Greenberg**](https://github.com/anchorwatt)
* [**Troy Wang**](https://github.com/TroyTianzhengWang)
