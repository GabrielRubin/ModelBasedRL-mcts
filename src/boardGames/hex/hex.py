import random
from typing import Tuple, List
from boardGames.hex.hex_wqu import WQuickUnion, HexDirection
from ML import StatePredictor

class HexGameState:
    NEIGHBORS = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]
    def __init__(self, board_size:int, starting_player:int):
        self.board          = [[0 for i in range(0, board_size)] for j in range(0, board_size)]
        self.player1_union  = WQuickUnion(board_size, HexDirection.HORIZONTAL)
        self.player2_union  = WQuickUnion(board_size, HexDirection.VERTICAL)
        self.current_player = starting_player
        self.turn           = 0
        self.winner         = 0
        self.board_size     = board_size

    #TODO: Create state from Alpha Zero representation (simple board - not one hot)
    '''
    @classmethod
    def FromReducedData(cls, data:List[int], boardSize:int, turn:int, currPlayer:int):
        """
        CREATES AN INSTANCE FROM A REDUCED DATA FORMAT (LIKE A REGULAR BOARD 2D-ARRAY, BUT FLATTEN)
        """
        state = cls(boardSize, currPlayer)
        state.turn = turn
        
        for i, value in enumerate(data):
        if value != 0:
            position = (i%boardSize, int(i/boardSize))
            state.board[position[0]][position[1]] = value
            state.UpdateUnion(value, position)

        return state
    '''

    @classmethod
    def from_data(cls, data:List[int], board_size:int, turn:int, current_player:int):
        """
        CREATES AN INSTANCE FROM OUR DATA FORMAT (ONE-HOT-BOARD - 2 BOARDS, ONE FOR EACH PLAYER)
        """
        board_dim = board_size*board_size
        data_player1 = data[:board_dim]
        data_player2 = data[board_dim:board_dim*2]
        state = cls(board_size, current_player)
        state.turn = turn

        def update_board(player_data, player):
            for i, value in enumerate(player_data):
                if value == 1:
                    position = (i%board_size, int(i/board_size))
                    state.board[position[0]][position[1]] = player
                    state.update_union(player, position)

        update_board(data_player1,  1)
        update_board(data_player2, -1)
        return state

    def put_piece(self, player:int, position:Tuple[int,int]):
        self.board[position[0]][position[1]] = player
        self.update_union(player, position)
        self.turn += 1
        self.current_player *= -1

    def update_union(self, player:int, position:Tuple[int, int]):
        connections = self.get_connections(player, position)
        if player == 1:
            player_union = self.player1_union
        else:
            player_union = self.player2_union
        for other_pos in connections:
            player_union.connect_points(position, other_pos)
        if player_union.check_win():
            self.winner = player

    def get_connections(self, player:int, position:Tuple[int,int]):
        selected_positions = []
        for pos in HexGameState.NEIGHBORS:
            x = position[0] + pos[0]
            y = position[1] + pos[1]
            if x < 0 or y < 0 or \
                x > len(self.board[0])-1 or y > len(self.board[1])-1:
                continue
            if self.board[x][y] == player:
                selected_positions.append((x, y))
        return selected_positions

    def get_available_positions(self):
        available_positions = []
        for y in range(0, len(self.board[1])):
            for x in range(0, len(self.board[0])):
                if self.board[x][y] == 0:
                    available_positions.append((x, y))
        #assert availablePositions
        return available_positions

    def get_available_positions_count(self):
        count = 0
        for y in range(0, len(self.board[1])):
            for x in range(0, len(self.board[0])):
                if self.board[x][y] == 0:
                    count += 1
        return count

    def get_possible_actions(self, player:int):
        available_positions = self.get_available_positions()
        return [PutPieceAction(player, position) for position in available_positions]

    def get_random_action(self, player:int):
        available_positions = self.get_available_positions()
        selected_position = available_positions[int(random.random() * len(available_positions))]
        return PutPieceAction(player, selected_position)

    def get_data(self):
        def select_player(value, player):
            if value == player:
                return 1
            return 0
        dataP1 = []
        dataP2 = []
        for y in range(0, len(self.board[1])):
            for x in range(0, len(self.board[0])):
                dataP1.append(select_player(self.board[x][y],  1))
                dataP2.append(select_player(self.board[x][y], -1))
        return dataP1 + dataP2

class PutPieceAction:
    def __init__(self, player:int, position:Tuple[int,int]):
        #assert len(position) == 2
        self.player   = player
        self.position = position

    def get_data(self, state:HexGameState):
        dataP1 = [0 for i in range(0, len(state.board[0]) * len(state.board[1]))]
        dataP2 = [0 for i in range(0, len(state.board[0]) * len(state.board[1]))]
        if self.player == 1:
            dataP1[self.position[0] + self.position[1] * len(state.board[0])] = 1
        else:
            dataP2[self.position[0] + self.position[1] * len(state.board[0])] = 1
        return dataP1 + dataP2

class HexSimulator:
    def __init__(self, board_size:int):
        self.board_size = board_size

    def apply_action(self, state:HexGameState, action:PutPieceAction):
        #assert state.board[self.position[0]][self.position[1]] != self.player #TEST
        state.put_piece(action.player, action.position)
        return state

    def undo_action(self, state:HexGameState, action:PutPieceAction):
        #assert state.board[action.position[0], action.position[1]] == action.player #TEST
        state.board[action.position[0], action.position[1]] = 0
        state.turn -= 1
        state.current_player *= -1
        return state

class HexSimulatorPredictor(HexSimulator):
    def __init__(self, predictor:StatePredictor):
        self.predictor = predictor

    def ApplyAction(self, state:HexGameState, action:PutPieceAction):
        prediction = self.predictor.GetPrediction(state.GetData(), action.GetData(state))
        return HexGameState.from_data(prediction, state.board_size, state.turn + 1, state.current_player * -1)