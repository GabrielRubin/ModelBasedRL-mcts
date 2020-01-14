import math
import random
import pickle
import time
from typing import List
import numpy as np
from boardGames.game_simulator_base import GameSimulator
from ml.rnd_model import _RandomNetworkDistillation
from ml.state_predictor_model import _StatePredictor

class TreeNode:
    def __init__(self, state, action, player:int, parent, possible_actions:List,
                 n_value=1, utility=0):
        self.state     = state
        self.action    = action
        self.player    = player
        self.parent    = parent
        self.q_value   = 0
        self.n_value   = n_value
        self.utility   = utility
        self.children  = []
        self.to_expand = possible_actions

    def expand(self, index=-1):
        if index == -1:
            index = int(random.random() * len(self.to_expand))
        action = self.to_expand[index]
        self.to_expand.remove(action)
        return action

    def get_state(self): #RETURNS A COPY OF THE STATE
        return pickle.loads(pickle.dumps(self.state))

    def __eq__(self, other):
        if other is None:
            if self is None:
                return True
            return False
        return self.__dict__ == other.__dict__

class MCTS:
    def __init__(self, simulator:GameSimulator, ec=0.25, rollouts=100):
        self.simulator         = simulator
        self.rollouts          = rollouts
        self.ec                = ec
        self.valid_decisions   = 0
        self.invalid_decisions = 0

    def choose_action(self, state, current_player):
        possible_actions = self.simulator.actions(state, current_player)
        if len(possible_actions) == 1:
            return possible_actions[0]
        return self.get_action(pickle.loads(pickle.dumps(state)), current_player)

    def create_root_node(self, state, player):
        return TreeNode(state=state,
                        action=None,
                        player=player,
                        parent=None,
                        possible_actions=self.simulator.actions(state, player))

    def get_action(self, state, player:int):
        #start = time.time()
        root_node     = self.create_root_node(state, player)
        rollout_count = 0
        while rollout_count < self.rollouts:
            selected_node = self.tree_policy(root_node)
            reward        = self.rollout_policy(selected_node.get_state(), selected_node.player)
            self.backup(selected_node, -reward)
            rollout_count += 1

        best = self.best_child(root_node, 0)
        #end = time.time()
        #print(end - start)
        return best.action

    def tree_policy(self, node:TreeNode):
        while node.utility == 0:
            if node.to_expand:
                return self.expand(node)
            node = self.best_child(node, self.ec)
        return node

    def expand(self, node:TreeNode):
        action = node.expand()
        next_state, next_player = self.simulator.result(node.get_state(), node.player, action)
        child_node = self.create_child_node(node, action, next_state, next_player)
        node.children.append(child_node)
        return child_node

    def create_child_node(self, parent_node, action, next_state, next_player):
        child_node = TreeNode(state=next_state,
                              action=action,
                              player=next_player,
                              parent=parent_node,
                              possible_actions=self.simulator.actions(next_state, next_player),
                              utility=self.simulator.utility(next_state, next_player))
        return child_node

    def best_child(self, node:TreeNode, exp_value:float):
        return max(node.children, key=lambda child: float(child.q_value)/float(child.n_value))

    def rollout_policy(self, state, player:int):
        current_player = player
        for _ in range(self.simulator.max_turns(state)):
            if self.simulator.terminal_test(state):
                break
            action = self.get_random_action(state, current_player)
            state, current_player = self.simulator.result(state, current_player, action)
        return self.simulator.utility(state, player)

    def get_random_action(self, state, player:int):
        valid_actions = self.simulator.actions(state, player)
        return valid_actions[int(random.random() * len(valid_actions))]

    def backup(self, node:TreeNode, reward:int):
        while node is not None:
            node.n_value += 1
            node.q_value += reward
            if node.parent and node.parent.player != node.player:
                reward *= -1
            node = node.parent

class UCT(MCTS):
    def _ucb_1(self, node:TreeNode, child_node:TreeNode, exp_value:float):
        exploitation = float(child_node.q_value) / max(float(child_node.n_value), 1)
        exploration  = exp_value * \
            math.sqrt((2 * math.log(float(node.n_value))) / max(float(child_node.n_value), 1))
        return exploitation + exploration

    def best_child(self, node:TreeNode, exp_value:int):
        return max(node.children, key=lambda child:self._ucb_1(node, child, exp_value))

class NoveltyTreeNode(TreeNode):
    def __init__(self, *args, novelty=0, **kwargs):
        self.novelty = novelty
        self.is_valid = True
        super().__init__(*args, **kwargs)
    @classmethod
    def from_tree_node(cls, tree_node:TreeNode):
        return cls(state=tree_node.state,
                   action=tree_node.action,
                   player=tree_node.player,
                   parent=tree_node.parent,
                   possible_actions=tree_node.to_expand,
                   utility=tree_node.utility)

class NoveltyUCT(UCT):
    def __init__(self, *args, rnd:_RandomNetworkDistillation, 
                 novelty_bonus=1, simulation_rerolls=10, **kwargs):
        self.rnd = rnd
        self.novelty_bonus = novelty_bonus
        self.simulation_rerolls = simulation_rerolls
        self.invalid_rollouts = 0
        super().__init__(*args, **kwargs)
    '''
    def get_action(self, state, player:int):
        #start = time.time()
        root_node     = self.create_root_node(state, player)
        rollout_count = 0
        while rollout_count < self.rollouts:
            selected_node = self.tree_policy(root_node)
            if selected_node.is_valid:
                reward = self.rollout_policy(selected_node.get_state(), selected_node.player)
                self.backup(selected_node, -reward)
                rollout_count += 1
            else:
                self.backup(selected_node, 0)

        best = self.best_child(root_node, 0)
        #end = time.time()
        #print(end - start)
        return best.action

    def _calculate_child_score(self, node:TreeNode, child_node:TreeNode, exp_value:float):
        #return self._score_test_A(node, child_node, exp_value)
        return self._score_test_B(node, child_node, exp_value)

    def _score_test_A(self, node:TreeNode, child_node:TreeNode, exp_value:float):
        ucb_score = self._ucb_1(node, child_node, exp_value)
        if exp_value == 0:
            return ucb_score
        if node.novelty == 0:
            #print("ZERO NOVELTY!")
            node.novelty = 0.0001
        novelty_score  = child_node.novelty / node.novelty
        novelty_factor = (novelty_score * self.novelty_bonus)
        final_score    = ucb_score - novelty_factor
        return final_score

    def _score_test_B(self, node:TreeNode, child_node:TreeNode, exp_value:float):
        ucb_score = self._ucb_1(node, child_node, exp_value)
        if not child_node.is_valid:
            if exp_value == 0:
                return random.uniform(-0.5, 0.5)
            return -1
        return ucb_score
    '''
    def rollout_policy(self, state, player:int):
        current_player = player
        for i in range(self.simulator.max_turns(state)):
            if self.simulator.terminal_test(state):
                break
            state_data = self.simulator.get_state_data(state)
            actions = [self.simulator.get_action_data(state, current_player, action) for action in self.simulator.actions(state, current_player)]
            if len(actions) > 10:
                actions = random.sample(actions, 10)
            states  = [pickle.loads(pickle.dumps(state_data)) for i in range(len(actions))]
            states, current_player = self.simulator.results(states, current_player, actions, self.rnd)
            if not states:
                break
            state_data = states[int(random.random() * len(states))]
            state = self.simulator.get_state_from_data(state, state_data)

        return self.simulator.utility(state, player)
    '''

    def create_root_node(self, state, player):
        return NoveltyTreeNode.from_tree_node(super().create_root_node(state, player))

    def create_child_node(self, parent_node, action, next_state, next_player):
        child_node = NoveltyTreeNode.from_tree_node(
            super().create_child_node(parent_node, action, next_state, next_player)
        )
        parent_state = parent_node.get_state()
        child_node.is_valid = self.rnd.is_transition_valid(
            [np.append(np.array(self.simulator.get_state_data(parent_state)) \
                     - np.array(self.simulator.get_state_data(next_state)),
                       np.array(self.simulator.get_action_data(parent_state, parent_node.player, action)))]
        )
        #DEBUG
        #self._debug_novelty(parent_node.get_state(), action, next_state, novelty)
        #child_node.novelty = novelty
        return child_node

    max_false_novelty = 0
    def _debug_novelty(self, prev_state, action, predicted_state, novelty):
        next_state = self.simulator.simulator._debug_apply_action(prev_state, action)
        are_equal  = _StatePredictor.compare_states(next_state.get_data(), predicted_state.get_data())
        if not are_equal:
            print(novelty)
        else:
            if novelty > NoveltyUCT.max_false_novelty:
                NoveltyUCT.max_false_novelty = novelty
                #print("NEW FALSE POSITIVE! = {0}".format(NoveltyUCT.max_false_novelty))

    def best_child(self, node:TreeNode, exp_value:int):
        return max(node.children,
                   key=lambda child:self._calculate_child_score(node, child, exp_value))
    '''
