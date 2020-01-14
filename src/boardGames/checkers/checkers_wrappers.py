import pickle
from boardGames.game_simulator_base import GameSimulator
from boardGames.checkers.checkers import CheckersGameState, CheckersSimulator, CheckersSimulatorPredictor, \
     CheckersMove, CheckerPiece

class CheckersGameSimulatorWrapper(GameSimulator):
    def __init__(self, simulator:CheckersSimulator):
        self.simulator = simulator
        
    def is_approximate_simulator(self):
        return isinstance(self.simulator, CheckersSimulatorPredictor)

    def actions(self, state:CheckersGameState, player:int):
        return state.get_possible_moves(player)

    def result(self, state:CheckersGameState, player, action:CheckersMove):
        result = self.simulator.apply_action(pickle.loads(pickle.dumps(state)), action)
        return result, result.current_player 

    def results(self, states, player, actions, rnd):
        return self.simulator.apply_actions(pickle.loads(pickle.dumps(states)), actions, rnd), -player

    def terminal_test(self, state:CheckersGameState):
        return state.get_winner() != 0

    def utility(self, state:CheckersGameState, player:int):
        winner = state.get_winner()
        if winner == 0:
            return 0
        if player == winner:
            return 1
        return -1

    def max_turns(self, state:CheckersGameState):
        return 150

    def get_state_data(self, state:CheckersGameState):
        return state.get_data()

    def get_action_data(self, state:CheckersGameState, player, action:CheckersMove):
        return action.get_data(state)

    def get_initial_state(self, starting_player):
        return CheckersGameState.get_initial_board(starting_player)

    def get_state_from_data(self, last_state, data):
        pass

    @staticmethod
    def get_state_data_len(board_dim):
        pass

    @staticmethod
    def get_action_data_len(board_dim):
        pass

    