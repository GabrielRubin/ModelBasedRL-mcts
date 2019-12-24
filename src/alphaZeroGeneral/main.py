from Coach import Coach
from Arena import Arena
from MCTS import MCTS
from MCTS_2 import UCT2
from utils import *

import os, sys, inspect
import numpy as np
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from boardGames.othello.OthelloGame import OthelloGame as Game
from boardGames.othello.pytorch.NNet import NNetWrapper as nn
from boardGames.othello.OthelloPredictor import OthelloPredictor
from boardGames.othello.OthelloPlayers import RandomPlayer, GreedyOthelloPlayer, HumanOthelloPlayer

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('temp','checkpoint_43.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

class AgentTest:
    def __init__(self, name:str, playFunc):
        self.name     = name
        self.playFunc = playFunc

def TrainNewNet():
    pred = OthelloPredictor(6, 'trainedModels/othello/pred_othello_073.pth', 100000)
    g = Game(6, predictor=pred)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

def PitNetworks(gameCount:int):
    pred = OthelloPredictor(6, 'trainedModels/othello/pred_othello_087.pth', 100000)
    g_pred    = Game(6, predictor=pred)
    g_regular = Game(6)
    nnet1 = nn(g_regular)
    nnet2 = nn(g_regular)

    #nnet1.load_checkpoint('AlphaZeroModels', 'predictor_87_ep93.pth.tar')
    nnet1.load_checkpoint('AlphaZeroModels', 'predictor_87_ep131.pth.tar')
    nnet2.load_checkpoint('AlphaZeroModels', 'pretrained_ep153.pth.tar')

    mcts1 = MCTS(g_regular, nnet1, args)
    mcts2 = MCTS(g_regular, nnet2, args)

    print('PITTING AGAINST PREVIOUS VERSION')
    arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),
                  lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), g_regular)
    p1wins, p2wins, draws = arena.playGames(gameCount)

    print('P1/P2 WINS : %d / %d ; DRAWS : %d' % (p1wins, p2wins, draws))

def PitAgents(agent1, agent2, boardSize:int, gameCount:int):
    game = Game(boardSize)
    print('(P1 = {0}) vs. (P2 = {1})'.format(agent1.name, agent2.name))
    arena = Arena(agent1.playFunc, agent2.playFunc, game)
    p1wins, p2wins, draws = arena.playGames(gameCount)

    print('P1/P2 WINS : %d / %d ; DRAWS : %d' % (p1wins, p2wins, draws))

def AlphaZeroAgent(agentName:str, boardSize:int, netName:str, predictorPath=None):
    if predictorPath is not None:
        pred = OthelloPredictor(boardSize, predictorPath, 100000)
        game = Game(boardSize, predictor=pred)
    else:
        game = Game(boardSize)
    nnet = nn(game)
    nnet.load_checkpoint('alphaZeroModels', netName)
    mcts = MCTS(game, nnet, args)
    return AgentTest(agentName, lambda x: np.argmax(mcts.getActionProb(x, temp=0)))

def UCTAgent(agentName:str, boardSize:int, ec:int=0.25, rollouts:int=100):
    game = Game(boardSize)
    uct  = UCT2(game, ec, rollouts)
    return AgentTest("{0}_ec={1}_r={2}".format(agentName, ec, rollouts), uct.GetAction)

def GreedyAgent(agentName:str, boardSize:int):
    game = Game(boardSize)
    gp = GreedyOthelloPlayer(game)
    return AgentTest(agentName, gp.play)

if __name__=="__main__":

    #sys.setrecursionlimit(100000)
    TrainNewNet()
    #PitNetworks(gameCount=50)
    
    #bSize = 6
    #alphaZero     = AlphaZeroAgent("Alpha Zero", bSize, 'pretrained_ep153.pth.tar')
    #alphaZeroPred = AlphaZeroAgent("Alpha Zero Pred", bSize, 'predictor_87_ep93.pth.tar', 
    #                                predictorPath='trainedModels/othello/pred_othello_087.pth')
    #azpTrainOnly  = AlphaZeroAgent("Alpha Zero Pred (train only)", bSize, 'predictor_87_ep93.pth.tar')
    #uct100        = UCTAgent("UCT Agent", bSize, ec=0.25, rollouts=100)
    #uct1000       = UCTAgent("UCT Agent", bSize, ec=0.25, rollouts=1000)
    #greedy        = GreedyAgent("Greedy Agent", bSize)

    #PitAgents(uct100, greedy, bSize, gameCount=100)