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
        if isinstance(self.simulator, CheckersSimulatorPredictor):
            sim = CheckersSimulator()
            real_result = sim.apply_action(pickle.loads(pickle.dumps(state)), action)
            return result, real_result.current_player
        return result, result.current_player

    def checkers_results(self, real_state, real_actions, data_states, player, actions, rnd):
        results        = self.simulator.predictor.get_next_states_checkers(data_states[0], data_states, actions, rnd)
        sim            = CheckersSimulator()
        curr_players   = [sim.apply_action(pickle.loads(pickle.dumps(real_state)), real_actions[i]).current_player for i in range(len(real_actions)) if results[i] is not None]
        results        = [results[i] for i in range(len(results)) if results[i] is not None]
        return results, curr_players

    def results(self, state, player, actions, rnd):
        return self.simulator.apply_actions(state, actions, rnd), -player

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
        return 100

    def get_state_data(self, state:CheckersGameState):
        return state.get_data()

    def get_action_data(self, state:CheckersGameState, player, action:CheckersMove):
        return action.get_data()

    def get_initial_state(self, starting_player):
        return CheckersGameState.get_initial_board(starting_player)

    def get_state_from_data(self, last_state, data):
        return CheckersGameState.from_data(data, last_state.current_player)

    @staticmethod
    def get_state_data_len(board_dim):
        return board_dim * 2

    @staticmethod
    def get_action_data_len(board_dim):
        return board_dim * 2

    def result_debug(self, state, player:int, action):
        tempSimulator    = CheckersSimulator()
        result           = self.simulator.apply_action(pickle.loads(pickle.dumps(state)), action)
        real_result      = tempSimulator.apply_action(pickle.loads(pickle.dumps(state)), action)
        is_correct       = 1
        result_data      = self.get_state_data(result)
        real_result_data = self.get_state_data(real_result)
        for i in range(len(result_data)):
            if result_data[i] != real_result_data[i]:
                is_correct = 0
                break
        return result, -player, is_correct
    