import numpy as np
import math
import random
import pickle

class TreeNode:

    def __init__(self, board, action:int, player:int, parent, possibleActions, nValue=1, utility=0):
        self.board    = board
        self.action   = action
        self.player   = player
        self.parent   = parent
        self.qValue   = 0
        self.nValue   = nValue
        self.utility  = utility
        self.children = []
        self.toExpand = possibleActions
    
    def Expand(self, index=-1):
        if index == -1:
            index = int(random.random() * len(self.toExpand))
        action = self.toExpand[index]
        self.toExpand.remove(action)
        return action

    def GetBoard(self, game):
        return pickle.loads(pickle.dumps(self.board))

    def __eq__(self, other):
        if other is None:
            if self is None:
                return True
            return False
        return self.__dict__ == other.__dict__

class MCTS2:

    def __init__(self, game, ec:int=0.25, rollouts:int = 100):
        self.game     = game
        self.rollouts = rollouts
        self.ec       = ec
        self.validDecisions   = 0
        self.invalidDecisions = 0

    def GetAction(self, board):
        rootNode = TreeNode(board=board,
                            action=None,
                            player=1,
                            parent=None,
                            possibleActions=self.game.getValidMoves2(board, 1))

        rolloutCount = 0
        while rolloutCount < self.rollouts:
            selectedNode = self.TreePolicy(rootNode)
            reward       = self.RolloutPolicy(selectedNode.GetBoard(self.game), selectedNode.player)
            self.BackUp(selectedNode, -reward)
            rolloutCount += 1

        best = self.BestChild(rootNode, 0)
        return best.action
    
    def TreePolicy(self, node:TreeNode):
        while node.utility == 0:
            if node.toExpand:
                return self.Expand(node)
            node = self.BestChild(node, self.ec)
        return node
    
    def Expand(self, node:TreeNode):
        action    = node.Expand()
        nextBoard, nextPlayer = self.game.getNextState(node.GetBoard(self.game), node.player, action)
        childNode = TreeNode(board=nextBoard,
                             action=action,
                             player=nextPlayer,
                             parent=node,
                             possibleActions=self.game.getValidMoves2(nextBoard, nextPlayer))
        node.children.append(childNode)
        return childNode

    def BestChild(self, node:TreeNode, expValue:float):
        return max(node.children, key=lambda child: child.qValue/child.nValue)

    def RolloutPolicy(self, board, player:int):
        utility = self.Utility(board, player)
        while utility == 0:
            action = self.GetRandomAction(board, player)
            board, player = self.game.getNextState(board, player, action)
            utility   = self.Utility(board, player)
        return utility

    def GetRandomAction(self, board, player):
        validActions = self.game.getValidMoves2(board, player)
        return validActions[int(random.random() * len(validActions))]

    def Utility(self, board, player:int):
        return self.game.getGameEnded(board, player)

    def BackUp(self, node:TreeNode, reward:int):
        while node is not None:
            node.nValue += 1
            node.qValue += reward
            reward *= -1
            node = node.parent

class UCT2(MCTS2):

  def UCB1(self, node:TreeNode, childNode:TreeNode, expValue:float):
    evaluationPart  = float(childNode.qValue) / max(float(childNode.nValue), 1)
    explorationPart = expValue * math.sqrt( (2 * math.log(node.nValue)) / max(float(childNode.nValue), 1) )
    return evaluationPart + explorationPart

  def BestChild(self, node:TreeNode, expValue:int):
    return max(node.children, key=lambda child:self.UCB1(node, child, expValue))