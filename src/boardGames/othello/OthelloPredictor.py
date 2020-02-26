import operator
import numpy as np
from cachetools import cachedmethod, LFUCache
from cachetools.keys import hashkey
#from ML import StatePredictor
from ml.state_predictor_model import OthelloStatePredictor

def CustomCacheKey(*args, strRepr="", **kwargs):
    key = hashkey(strRepr)
    return key

class OthelloPredictor:

    def __init__(self, boardSize:int, predictorPath:str, cacheSize:int):
        self.boardSize = boardSize
        self.predictor = OthelloStatePredictor.from_file(predictorPath)
        #self.predictor.cpu()
        self.predictionCache = LFUCache(maxsize=cacheSize)

    @staticmethod
    def GetOthelloBoardInMyFormat(othelloBoard):
        othelloBoard1 = np.copy(othelloBoard)
        othelloBoard2 = np.copy(othelloBoard)
        othelloBoard1[othelloBoard1 < 0] = 0 #neat trick!
        othelloBoard2[othelloBoard2 > 0] = 0
        othelloBoard2 *= -1

        finalBoard = np.append(othelloBoard1.flatten(), othelloBoard2.flatten())

        return finalBoard

    '''
    @staticmethod
    def GetOthelloActionInMyFormat(boardSize, actionArray):

        board = np.array([[0 for i in range(0, boardSize)] for j in range(0, boardSize)])
        if 1 in actionArray:
            return np.append(actionArray, board)
        else:
            return np.append(board, actionArray)
    '''

    @staticmethod
    def GetOriginalOthelloBoard(boardSize, board):
        for i in range(len(board)):
            val = board[i]
            if val < 0 or val >= 1:
                board[i] = 1
        othelloBoard1 = np.array(board[:boardSize*boardSize])
        othelloBoard2 = np.array(board[boardSize*boardSize:])
        othelloBoard2 *= -1
        final_board = othelloBoard1 + othelloBoard2

        #DEBUG
        #for i in final_board:
        #    if abs(i) > 1:
        #        print("bizarre board!")

        final_board = final_board.reshape(boardSize, boardSize)
        return final_board

    @staticmethod
    def GetBoardWithAction(boardSize, action, player):
        board = np.array([[0 for i in range(0, boardSize)] for j in range(0, boardSize)])
        if(int(action/boardSize) >= boardSize): # NO MOVES AVAILABLE
            return board
        move  = (int(action/boardSize), action%boardSize)
        board[move] = player
        return board

    @cachedmethod(operator.attrgetter('predictionCache'), key=CustomCacheKey)
    def GetNextState(self, player, othelloBoard, othelloAction, strRepr=""):
        boardData     = self.GetOthelloBoardInMyFormat(othelloBoard)
        othelloAction = self.GetBoardWithAction(self.boardSize, othelloAction, player)
        actionData    = self.GetOthelloBoardInMyFormat(othelloAction)

        nextBoard = self.predictor.get_next_state(boardData.tolist(), actionData.tolist())

        return self.GetOriginalOthelloBoard(self.boardSize, nextBoard)

    def GetNextStates(self, player, board, actions, rnd):

        def get_action_value(a):
            if sum(a) == 0:
                return 36
            else:
                return a.index(1)

        def get_player_action_space(board, player):
            if player == 1:
                return board[:int((len(board)*0.5).__round__())]
            else:
                return board[int((len(board)*0.5).__round__()):]
        
        boardData  = [board for i in range(len(actions))]
        actionData = [self.GetOthelloBoardInMyFormat(self.GetBoardWithAction(self.boardSize, get_action_value(get_player_action_space(action, player)), player)).tolist() for action in actions]
        nextBoards = self.predictor.get_next_states(board, boardData, actionData, rnd)

        return [self.GetOriginalOthelloBoard(self.boardSize, nextBoard) for nextBoard in nextBoards]

    def GetNextStates2(self, player, board, actions, simulator):

        def get_action_value(a):
            if sum(a) == 0:
                return 36
            else:
                return a.index(1)

        boardData  = [board for i in range(len(actions))]
        actionData = [self.GetOthelloBoardInMyFormat(self.GetBoardWithAction(self.boardSize, get_action_value(action), player)).tolist() for action in actions]
        nextBoards = self.predictor.get_next_states2(board, boardData, actionData, simulator, player)

        return [self.GetOriginalOthelloBoard(self.boardSize, nextBoard) for nextBoard in nextBoards]

