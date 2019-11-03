import pickle
from boardGames.hex.hex import HexGameState, HexSimulator, PutPieceAction, HexSimulatorPredictor
from boardGames.game_simulator_base import GameSimulator

class HexGameSimulatorWrapper(GameSimulator):
    def __init__(self, simulator:HexSimulator):
        self.simulator = simulator

    def is_approximate_simulator(self):
        return isinstance(self.simulator, HexSimulatorPredictor)

    def actions(self, state:HexGameState, player:int):
        return state.get_possible_actions(player)

    def result(self, state:HexGameState, player, action:PutPieceAction):
        return self.simulator.apply_action(pickle.loads(pickle.dumps(state)), action), -player

    def results(self, states, player, actions, rnd):
        return self.simulator.apply_actions(states, actions, rnd), -player

    def result_debug(self, state, player:int, action):
        result = self.simulator.apply_action(pickle.loads(pickle.dumps(state)), action)
        real_result = HexSimulator(self.simulator.board_size).apply_action(pickle.loads(pickle.dumps(state)), action)
        is_correct       = 1
        result_data      = self.get_state_data(result)
        real_result_data = self.get_state_data(real_result)
        for i in range(len(result_data)):
            if result_data[i] != real_result_data[i]:
                is_correct = 0
                break
        return result, -player, is_correct

    def terminal_test(self, state:HexGameState):
        return state.winner != 0

    def utility(self, state:HexGameState, player):
        if state.winner == 0:
            return 0
        if player == state.winner:
            return 1
        return -1

    def max_turns(self, state:HexGameState):
        return state.board_size * state.board_size

    def get_state_data(self, state:HexGameState):
        return state.get_data()

    def get_action_data(self, state:HexGameState, player, action:PutPieceAction):
        return action.get_data(state)

    def get_initial_state(self, starting_player):
        return HexGameState(self.simulator.board_size, starting_player)

    def get_state_from_data(self, last_state, data):
        return HexGameState.from_data(data, last_state.board_size, last_state.turn + 1, last_state.current_player * -1)

    @staticmethod
    def get_state_data_len(board_dim):
        return board_dim * board_dim * 2

    @staticmethod
    def get_action_data_len(board_dim):
        return board_dim * board_dim * 2

#TODO: COMPLETE THIS WRAPPER IMPLEMENTATION FOR ALPHA ZERO INTEGRATION:
'''
class HexAlphaZeroWrapper(Game):
    def __init__(self, n=9):
        self.n   = n
        self.hex = HexGameState(n, 0)
        
    def getInitBoard(self):
        return np.array([[0 for i in range(0, self.n)] for j in range(0, self.n)])

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n * self.n

    def getNextState(self, board, player, action):
        state = HexGameState.FromReducedData(board)
        action = PutPieceAction(player, (action%self.n, int(action/self.n)))
        simulator = HexSimulator()
        state = simulator.ApplyAction(state, action)
        return state.board

    def getValidMoves(self, board, player):
        valids = [0 for i in range(0, self.getActionSize())]
        state  = HexGameState.FromReducedData(board)
        available_positions = state.GetAvailablePositions()
        for x, y in available_positions:
            valids[x + y * self.n] = 1
        return valids

    def getGameEnded(self, board, player):
        state = HexGameState.FromReducedData(board)
        winner = state.GetWinner()
        if winner == 0:
            return 0
        if winner == player:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        return player*board

    def getSymmetries(self, board, pi):
        result = []
        piBoard = np.reshape(pi[:-1], (self.n, self.n))
        boardFlipRL = np.fliplr(board)
        piFlipRL    = np.fliplr(piBoard)
        boardFlipUD = np.flipud(board)
        piFlipUD    = np.flipud(piBoard)
        result += [(board, list(piBoard.ravel())) + [pi[-1]]]
        result += [(boardFlipRL, list(piFlipRL.ravel())) + [pi[-1]]]
        result += [(boardFlipUD, list(piFlipUD.ravel())) + [pi[-1]]]
        result += [(np.flipud(boardFlipRL), list(np.flipud(piFlipRL).ravel())) + [pi[-1]]]
        return result

    def stringRepresentation(self, board):
        return board.tostring()
'''