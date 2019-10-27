import random
import numpy as np
from boardGames.othello.OthelloGame import OthelloGame
from boardGames.othello.OthelloPredictor import OthelloPredictor
from boardGames.game_simulator_base import GameSimulator

class OthelloSimulator(GameSimulator):
    def __init__(self, predictor:OthelloPredictor=None):
        self.game = OthelloGame(6, predictor=predictor)

    def actions(self, state, player:int):
        tempState = np.copy(state)
        return self.game.getValidMoves2(tempState, player)
    
    def result(self, state, player, action):
        tempState = np.copy(state)
        return self.game.getNextState(tempState, player, action, True)

    def terminal_test(self, state):
        return self.game.getGameEnded(state, 1) != 0

    def utility(self, state, player):
        tempState = np.copy(state)
        winner = self.game.getGameEnded(tempState, 1)
        if player == winner:
            return 1
        return -1

    def max_turns(self, state):
        return 100

    def get_state_data(self, state):
        return OthelloPredictor.GetOthelloBoardInMyFormat(state).tolist()

    def get_action_data(self, state, player:int, action):
        actionInBoardFormat = OthelloPredictor.GetBoardWithAction(6, action, player)
        return OthelloPredictor.GetOthelloBoardInMyFormat(actionInBoardFormat).tolist()

    def get_initial_state(self, starting_player:int):
        return self.game.getInitBoard()

    @staticmethod
    def get_state_data_len(board_dim):
        return board_dim * board_dim * 2

    @staticmethod
    def get_action_data_len(board_dim):
        return board_dim * board_dim * 2

'''
class OthelloRollout:
    def __init__(self, player1, player2, predictor=None):
        self.game = OthelloGame(6, predictor=predictor)
        self.player1 = player1
        self.player2 = player2

    @staticmethod
    def Clamp01(value):
        return min(max(value, 0), 1)

    def GetRolloutDataWithSymetry(self):
        players = [self.player1, None, self.player2]
        curPlayer = (1 - 2 * int(random.random() * 2))
        board = self.game.getInitBoard()
        gameData = []
        while self.game.getGameEnded(board, curPlayer)==0:
            pastBoard        = np.copy(board)
            pastplayer       = curPlayer
            action           = players[curPlayer+1].play(self.game.getCanonicalForm(board, curPlayer))
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            boardData        = OthelloPredictor.GetOthelloBoardInMyFormat(pastBoard)
            actionData       = OthelloPredictor.GetOthelloActionInMyFormat(6, OthelloPredictor.GetBoardWithAction(6, action, pastplayer))
            nextBoardData    = OthelloPredictor.GetOthelloBoardInMyFormat(board) 
            totalData        = np.append(boardData, actionData)
            totalData        = np.append(totalData, nextBoardData)
            gameData.append(totalData)

    def GetRolloutData(self):
        players = [self.player1, None, self.player2]
        curPlayer = (1 - 2 * int(random.random() * 2))
        board = self.game.getInitBoard()
        gameData = []
        while self.game.getGameEnded(board, curPlayer)==0:
            pastBoard        = np.copy(board)
            pastplayer       = curPlayer
            action           = players[curPlayer+1].play(self.game.getCanonicalForm(board, curPlayer))
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            boardData        = OthelloPredictor.GetOthelloBoardInMyFormat(pastBoard)
            actionData       = OthelloPredictor.GetOthelloActionInMyFormat(6, OthelloPredictor.GetBoardWithAction(6, action, pastplayer))
            nextBoardData    = OthelloPredictor.GetOthelloBoardInMyFormat(board) 
            totalData        = np.append(boardData, actionData)
            totalData        = np.append(totalData, nextBoardData)
            gameData.append(totalData)
'''