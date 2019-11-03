import random
from boardGames.game_simulator_base import GameSimulator
from mcts import NoveltyUCT

class RandomAgent:
    def __init__(self, simulator:GameSimulator):
        self.simulator = simulator

    def choose_action(self, state, current_player):
        valid_actions = self.simulator.actions(state, current_player)
        return valid_actions[int(random.random() * len(valid_actions))]

class GameTrailBase:
    def __init__(self, simulator:GameSimulator, player1=None, player2=None):
        self.simulator = simulator
        if player1 is None:
            player1 = RandomAgent(self.simulator)
        if player2 is None:
            player2 = RandomAgent(self.simulator)
        #                -1        0       1
        self.players = [player2, None, player1]

    def _get_random_player(self):
        return 1 - 2 * int(random.random() * 2)

    def _do_rollout(self, starting_player):
        current_player  = starting_player
        current_state   = self.simulator.get_initial_state(starting_player)
        max_turns       = self.simulator.max_turns(current_state)
        for _ in range(max_turns):
            action = self.players[current_player+1].choose_action(current_state, current_player)
            current_state, current_player = \
                self.simulator.result(current_state, current_player, action)
            if self.simulator.terminal_test(current_state):
                break
        winner = self.simulator.utility(current_state, 1)
        return winner

    def do_rollouts(self, game_count:int, randomize_starting_player=False):
        half_games = (game_count * 0.5).__round__()
        starting_player = 0
        for game in range(game_count):
            if randomize_starting_player:
                starting_player = self._get_random_player()
            else:
                if game < half_games:
                    starting_player = 1
                else:
                    starting_player = -1
            yield self._do_rollout(starting_player)

class DataCollectTrial(GameTrailBase):
    def _do_rollout(self, starting_player):
        data = []
        current_player  = starting_player
        current_state   = self.simulator.get_initial_state(starting_player)
        max_turns       = self.simulator.max_turns(current_state)
        for _ in range(max_turns):
            action = self.players[current_player+1].choose_action(current_state, current_player)
            curr_state_data = self.simulator.get_state_data(current_state)
            action_data = self.simulator.get_action_data(current_state, current_player, action)
            current_state, current_player = \
                self.simulator.result(current_state, current_player, action)
            next_state_data = self.simulator.get_state_data(current_state)
            data.append(curr_state_data + action_data + next_state_data)
            if self.simulator.terminal_test(current_state):
                break
        winner = self.simulator.utility(current_state, 1)
        return data, winner

class DataCollectWithInvalidRolloutCount(GameTrailBase):
    def _do_rollout(self, starting_player):
        current_player  = starting_player
        current_state   = self.simulator.get_initial_state(starting_player)
        max_turns       = self.simulator.max_turns(current_state)
        total_invalid_rollouts = 0
        for _ in range(max_turns):
            action = self.players[current_player+1].choose_action(current_state, current_player)
            if isinstance(self.players[current_player+1], NoveltyUCT):
                total_invalid_rollouts += self.players[current_player+1].invalid_rollouts
            current_state, current_player = \
                self.simulator.result(current_state, current_player, action)
            if self.simulator.terminal_test(current_state):
                break
        winner = self.simulator.utility(current_state, 1)
        return winner, total_invalid_rollouts

class DataCollectWithSimCategory(GameTrailBase):
    def _do_rollout(self, starting_player):
        data = []
        current_player      = starting_player
        current_state       = self.simulator.get_initial_state(starting_player)
        max_turns           = self.simulator.max_turns(current_state)
        for _ in range(max_turns):
            action = self.players[current_player+1].choose_action(current_state, current_player)
            curr_state_data = self.simulator.get_state_data(current_state)
            action_data = self.simulator.get_action_data(current_state, current_player, action)
            current_state, current_player, is_correct = \
                self.simulator.result_debug(current_state, current_player, action)
            next_state_data = self.simulator.get_state_data(current_state)
            data.append(curr_state_data + action_data + next_state_data + [is_correct])
            if self.simulator.terminal_test(current_state):
                break
        winner = self.simulator.utility(current_state, 1)
        return data, winner
