import math
import random
import pickle
from typing import List
import numpy as np
from boardGames.game_simulator_base import GameSimulator

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
        return self.get_action(pickle.loads(pickle.dumps(state)), current_player)

    def create_root_node(self, state, player):
        return TreeNode(state=state,
                        action=None,
                        player=player,
                        parent=None,
                        possible_actions=self.simulator.actions(state, player))

    def get_action(self, state, player:int):
        root_node     = self.create_root_node(state, player)
        rollout_count = 0
        while rollout_count < self.rollouts:
            selected_node = self.tree_policy(root_node)
            reward        = self.rollout_policy(selected_node.get_state(), selected_node.player)
            self.backup(selected_node, -reward)
            rollout_count += 1

        best = self.best_child(root_node, 0)
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
        return max(node.children, key=lambda child: child.q_value/child.n_value)

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
            node = node.parent
            reward *= -1

class UCT(MCTS):
    def _ucb_1(self, node:TreeNode, child_node:TreeNode, exp_value:float):
        exploitation = float(child_node.q_value) / max(float(child_node.n_value), 1)
        exploration  = exp_value * \
            math.sqrt((2 * math.log(node.n_value)) / max(float(child_node.n_value), 1))
        return exploitation + exploration

    def best_child(self, node:TreeNode, exp_value:int):
        return max(node.children, key=lambda child:self._ucb_1(node, child, exp_value))

class NoveltyTreeNode(TreeNode):
    def __init__(self, *args, novelty=0, **kwargs):
        self.novelty = novelty
        super().__init__(*args, **kwargs)
    @classmethod
    def from_tree_node(cls, tree_node:TreeNode):
        return cls(state=tree_node.state,
                   action=tree_node.action,
                   player=tree_node.player,
                   parent=tree_node.parent,
                   possible_actions=tree_node.toExpand,
                   utility=tree_node.utility)

class NoveltyUCT(UCT):
    def __init__(self, *args, rnd=0, novelty_bonus=1, **kwargs):
        self.rnd = rnd
        self.novelty_bonus = novelty_bonus
        super().__init__(*args, **kwargs)

    def _calculate_child_score(self, node:TreeNode, child_node:TreeNode, exp_value:float):
        ucb_score = self._ucb_1(node, child_node, exp_value)
        if node.novelty == 0:
            print("ZERO NOVELTY!")
            node.novelty = 0.0001
        novelty_score  = child_node.novelty / node.novelty
        novelty_factor = (novelty_score * self.novelty_bonus)
        final_score    = ucb_score - novelty_factor
        return final_score

    def create_root_node(self, state, player):
        return NoveltyTreeNode.from_tree_node(super().create_root_node(state, player))

    def create_child_node(self, parent_node, action, next_state, next_player):
        child_node = NoveltyTreeNode.from_tree_node(
            super().create_child_node(parent_node, action, next_state, next_player)
        )
        novelty, _ = self.rnd.get_novelty(
            np.array(self.simulator.get_state_data(parent_node.get_state())) \
            - np.array(self.simulator.get_state_data(next_state))
        )
        child_node.novelty = novelty
        return child_node

    def best_child(self, node:TreeNode, expValue:int):
        return max(node.children,
                   key=lambda child:self._calculate_child_score(node, child, expValue))

    def backup(self, node:TreeNode, reward:int):
        novelty = node.novelty
        while node is not None:
            node.n_value += 1
            node.q_value += reward
            node = node.parent
            reward *= -1
            if node is not None:
                node.novelty += novelty
