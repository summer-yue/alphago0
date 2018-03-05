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

def nn_vs_random(nn_batch, game_num):
    """ Play raw neural net against random play
    Args:
        nn_batch: the batch number for the version of ResNet used, save in the models folder
        game_num: number of games played
    Returns:
        black winning counts, white winning counts
    """
    uniform_net = UniformPredictionNet(path_to_model = '/', board_dimension = BOARD_DIM)
    utils = GoUtils()
    count_nn_winning = 0
    count_random_winning = 0
    alphago0 = AlphaGoZero(model_path="../models/batch_" + str(nn_batch), restored=True)
   
    for i in range(game_num):
        print()
        print("game number ", i)
        game_over = False
        board = GoBoard(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        while not game_over:
            #Raw NN plays black 
            if board.player == PLAYER_BLACK:
                print("Raw NN plays")
                move, _ = alphago0.play_with_raw_nn(board)
            else:
                print("Random plays")
                p, _ = uniform_net.predict(board)
                move = random.choice([move for move in p.keys() if p[move] > 0])

            print("\t move is", move)

            _, board = utils.make_move(board=board, move=move)

            if utils.is_game_finished(board) or len(board.game_history) > BOARD_DIM**2*2:
                game_over = True
                winner, winning_by_points = utils.evaluate_winner(board.board_grid)
                if winning_by_points > 0:
                    if winner == 1:
                        count_nn_winning += 1
                    elif winner == -1:
                        count_random_winning += 1
                print("winner is ", winner)
                print("winning by points", winning_by_points)
                print(board)

    return count_nn_winning, count_random_winning

def ai_vs_random(nn_batch, ai_simulation_num, game_num):
    """ Play ai against random play
    Args:
        nn_batch: the batch number for the version of ResNet used, save in the models folder
        ai_simulation_num: simulation number used in AlphaGo
        game_num: number of games played
    Returns:
        percentage of games when AI beats MCTS
    """
    uniform_net = UniformPredictionNet(path_to_model = '/', board_dimension = BOARD_DIM)
    utils = GoUtils()
    count_nn_winning = 0
    count_random_winning = 0
    alphago0 = AlphaGoZero(model_path="../models/batch_" + str(nn_batch), restored=True)
   
    for i in range(game_num):
        print()
        print("game number ", i)
        game_over = False
        board = GoBoard(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        while not game_over:
            #AlphaGo with MCTS plays black 
            if board.player == PLAYER_BLACK:
                print("AlphaGo Zero plays")
                move = alphago0.play_with_mcts(board, simulation_number=mcts_simulation_num)
            else:
                print("Random plays")
                p, _ = uniform_net.predict(board)
                move = random.choice([move for move in p.keys() if p[move] > 0])
                
            print("\t move is", move)

            _, board = utils.make_move(board=board, move=move)

            if utils.is_game_finished(board) or len(board.game_history) > BOARD_DIM**2*2:
                game_over = True
                winner, winning_by_points = utils.evaluate_winner(board.board_grid)
                if winning_by_points > 0:
                    if winner == 1:
                        count_nn_winning += 1
                    elif winner == -1:
                        count_random_winning += 1
                print("winner is ", winner)
                print("winning by points", winning_by_points)
                print(board)

    return count_nn_winning, count_random_winning

def ai_vs_mcts(nn_batch, ai_simulation_num, mcts_simulation_num, game_num):
    """ Play ai against mcts (with uniform heuristic) only and calculate the ai's winning rate
    Args:
        nn_batch: the batch number for the version of ResNet used, save in the models folder
        ai_simulation_num: simulation number used in AlphaGo
        mcts_simulation_num: simluation number used in MCTS
        game_num: number of games played
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
            #AlphaGo with MCTS plays black 
            if board.player == PLAYER_BLACK:
                print("AlphaGo Zero plays")
                move = alphago0.play_with_mcts(board, simulation_number=mcts_simulation_num)
            else:
                print("MCTS plays")
                mcts_play_instance = MCTS(board, uniform_net, utils, simluation_number=mcts_simulation_num)
                move = mcts_play_instance.run_simulations_without_noise()
                
            print("\t move is", move)

            _, board = utils.make_move(board=board, move=move)

            if utils.is_game_finished(board) or len(board.game_history) > BOARD_DIM**2*2:
                game_over = True
                winner, winning_by_points = utils.evaluate_winner(board.board_grid)
                if winning_by_points > 0:
                    if winner == 1:
                        count_nn_winning += 1
                    elif winner == -1:
                        count_mcts_winning += 1
                print("winner is ", winner)
                print("winning by points", winning_by_points)
                print(board)

    return count_nn_winning, count_mcts_winning

batch = 1920 #Last model saved
game_num = 100
# mcts_simulation_num = 5000
# ai_simulation_num = 300
# print("For nn trained with {} batches VS MCTS simluations 100 playing {} games, the winning ratio is".format(batch, game_num))
# print(ai_vs_mcts(nn_batch=batch, ai_simulation_num=ai_simulation_num, mcts_simulation_num=100, game_num=game_num))

print("For nn trained with {} batches VS random playing {} games, the winning ratio is".format(batch, game_num))
print(nn_vs_random(nn_batch=batch, game_num=game_num))



