import operator
import numpy as np
from cachetools import cachedmethod, LFUCache
from cachetools.keys import hashkey
from ML import StatePredictor

def CustomCacheKey(*args, strRepr="", **kwargs):
    key = hashkey(strRepr)
    return key

class OthelloPredictor:

    def __init__(self, boardSize:int, predictorPath:str, cacheSize:int):
        self.boardSize = boardSize
        self.predictor = StatePredictor.FromModel(predictorPath, autoFormat=False)
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

        othelloBoard1 = board[:boardSize*boardSize]
        othelloBoard2 = board[boardSize*boardSize:]
        othelloBoard2 *= -1
        finalBoard    = othelloBoard1 + othelloBoard2
        finalBoard    = finalBoard.reshape(boardSize, boardSize, 1)
        return finalBoard

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
        actionData    = self.GetOthelloActionInMyFormat(self.boardSize, othelloAction)
        totalData     = np.append(boardData, actionData)

        nextBoard = self.predictor.GetPrediction2(totalData)

        return self.GetOriginalOthelloBoard(self.boardSize, nextBoard)

