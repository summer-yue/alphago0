from game.go_board import GoBoard
from game.go_utils import GoUtils
from gui.alphago_zero import AlphaGoZero
from self_play.mcts import MCTS
from self_play.self_play import SelfPlay
from value_policy_net.tests.uniform_prediction_net import UniformPredictionNet
from value_policy_net.tests.random_net import RandomNet

import numpy as np
import random

PLAYER_BLACK = 1
PLAYER_WHITE = -1
BOARD_DIM = 5

def ai_vs_mcts(nn_batch, mcts_simulation_num, game_num):
    """ Play ai against mcts only and calculate the ai's winning rate
    Args:
        nn_batch: the batch number for the version of ResNet used
        mcts_simulation_num: simluation
    Returns:
        percentage of games when AI beats MCTS
    """
    uniform_net = UniformPredictionNet(path_to_model = '/', board_dimension = BOARD_DIM)
    utils = GoUtils()
    count_nn_winning = 0
    count_mcts_winning = 0
    alphago0 = AlphaGoZero(model_path="../models/batch_" + str(nn_batch), restored=True)
   
    for i in range(game_num):
        print()
        print("game number ", i)
        game_over = False
        board = GoBoard(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        while not game_over:
            #NN plays black 
            if board.player == PLAYER_BLACK:
                print("MCTS plays")
                mcts_play_instance = MCTS(board, uniform_net, utils, simluation_number=mcts_simulation_num)
                move = mcts_play_instance.run_simulations_without_noise()
                # mcts_play_instance = MCTS(board, uniform_net, utils, simluation_number = mcts_simulation_num)
                # move = mcts_play_instance.run_simulations_without_noise()
            else:
                
                print("AlphaGo Zero plays")
                move = alphago0.play_with_mcts(board, simulation_number=mcts_simulation_num)
               
                # p, _ = uniform_net.predict(board)
                # move = random.choice([move for move in p.keys() if p[move] > 0])

            print("\t move is", move)

            _, board = utils.make_move(board=board, move=move)

            if utils.is_game_finished(board):
                game_over = True
                winner, winning_by_points = utils.evaluate_winner(board.board_grid)
                if winner == 1:
                    count_nn_winning += 1
                elif winner == -1:
                    count_mcts_winning += 1
                print("winner is ", winner)
                print("winning by points", winning_by_points)
                print(board)

    return count_nn_winning, count_mcts_winning

batch = 970
print("For nn trained with {} batches VS MCTS simluations 100 playing 10 games, the winning ratio is".format(batch))
print(ai_vs_mcts(nn_batch=batch, mcts_simulation_num=300, game_num=10))