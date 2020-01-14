import numpy as np

class CheckerPiece:

    #PIECE STRUCTURE: [ X, Y, Type, O1, O2, O3, O4, E1, E2, E3, E4 ]
    #t = piece type: -1, -2, 1, 2 (where -2 and 2 are super pieces)
    #open = Open pieces, where 00 = UP LEFT, 01 = UP RIGHT, 02 = DOWN LEFT, 03 = DOWN RIGHT
    #capture = Captures available, where 00 = UP LEFT, 01 = UP RIGHT, 02 = DOWN LEFT, 03 = DOWN RIGHT

    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.piece_type = t
        self.open    = [0, 0, 0, 0]
        self.capture = [0, 0, 0, 0]

    def is_open(self):
        return sum(self.open) > 0
    
    def is_capture(self):
        return sum(self.capture) > 0

class CheckersGameState:
    #P1 = black (1), P2 = white (-1)

    checker_directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    __checker_board_size = 8

    def __init__(self, current_player:int):
        self.current_player = current_player
        self.p1_pieces = []
        self.p2_pieces = []
        self.lookup_board = self.create_lookup_board()

    @classmethod
    def get_initial_board(cls, starting_player:int):
        instance = cls(starting_player)
        instance.p1_pieces = [
            CheckerPiece(1, 0, 1),
            CheckerPiece(3, 0, 1),
            CheckerPiece(5, 0, 1),
            CheckerPiece(7, 0, 1),
            CheckerPiece(0, 1, 1),
            CheckerPiece(2, 1, 1),
            CheckerPiece(4, 1, 1),
            CheckerPiece(6, 1, 1),
            CheckerPiece(1, 2, 1),
            CheckerPiece(3, 2, 1),
            CheckerPiece(5, 2, 1),
            CheckerPiece(7, 2, 1)
        ]
        instance.p2_pieces = [
            CheckerPiece(0, 5, -1),
            CheckerPiece(2, 5, -1),
            CheckerPiece(4, 5, -1),
            CheckerPiece(6, 5, -1),
            CheckerPiece(1, 6, -1),
            CheckerPiece(3, 6, -1),
            CheckerPiece(5, 6, -1),
            CheckerPiece(7, 6, -1),
            CheckerPiece(0, 7, -1),
            CheckerPiece(2, 7, -1),
            CheckerPiece(4, 7, -1),
            CheckerPiece(6, 7, -1)
        ]
        instance.update_board()
        return instance

    @classmethod
    def from_lookup_board(cls, lookup_board, current_player):
        instance = cls(current_player)
        for y in range(0, CheckersGameState.__checker_board_size):
            for x in range(0, CheckersGameState.__checker_board_size):
                board_content = lookup_board[x][y]
                if board_content != 0:
                    player = np.sign(board_content)
                    piece  = CheckerPiece(x, y, board_content)
                    if player == 1:
                        instance.p1_pieces.append(piece)
                    else:
                        instance.p2_pieces.append(piece)
        instance.update_board()
        instance.lookup_board = lookup_board

    def create_lookup_board(self):
        lookup_board = [[0 for i in range(0, CheckersGameState.__checker_board_size)] for j in range(0, CheckersGameState.__checker_board_size)]
        for p in self.p1_pieces:
            assert lookup_board[p.x][p.y] == 0
            lookup_board[p.x][p.y] = np.sign(p.piece_type)
        for p in self.p2_pieces:
            assert lookup_board[p.x][p.y] == 0
            lookup_board[p.x][p.y] = np.sign(p.piece_type)
        return lookup_board

    def update_board(self):
        self.lookup_board = self.create_lookup_board()
        for p in self.p1_pieces:
            self.update_piece(p)
        for p in self.p2_pieces:
            self.update_piece(p)

    def update_turn(self, last_move):
        #keep the current player if he can make a capture 'combo' with the current piece
        if last_move.is_capture:
            piece = self.get_piece(self.current_player, last_move.get_result_position())
            if piece.is_capture():
                return
        self.current_player *= -1

    def update_piece(self, p:CheckerPiece):
        p.open = [0, 0, 0, 0]
        p.capture = [0, 0, 0, 0]
        if np.abs(p.piece_type) > 1:
            #SUPER PIECE
            for i in range(0, 4):
                self.update_piece_stats(p, i)
        else:
            if p.piece_type > 0:
                for i in range(2, 4):
                    self.update_piece_stats(p, i)
            else:
                for i in range(0, 2):
                    self.update_piece_stats(p, i)

    def update_piece_stats(self, p:CheckerPiece, dir_index):
        d = CheckersGameState.checker_directions[dir_index]
        new_pos = (p.x + d[0], p.y + d[1])
        if CheckersGameState.is_position_valid(new_pos):
            player = np.sign(p.piece_type)
            board_content = self.lookup_board[new_pos[0]][new_pos[1]]
            if board_content != 0 and board_content != player:
                temp_dir = CheckersGameState.checker_directions[dir_index]
                capture_pos = (new_pos[0] + temp_dir[0], new_pos[1] + temp_dir[1])
                if CheckersGameState.is_position_valid(capture_pos) and self.lookup_board[capture_pos[0]][capture_pos[1]] == 0:
                    p.capture[dir_index] = 1
            elif board_content == 0:
                p.open[dir_index] = 1
            
    @staticmethod
    def is_position_valid(pos):
        if pos[0] >= CheckersGameState.__checker_board_size or pos[0] < 0:
            return False
        if pos[1] >= CheckersGameState.__checker_board_size or pos[1] < 0:
            return False
        return True

    def get_possible_moves(self, playerCode:int):
        if playerCode == 1:
            pieces = self.p1_pieces
        else:
            pieces = self.p2_pieces
        possible_moves = []
        for p in pieces:
            if p.is_capture():
                for i, c in enumerate(p.capture):
                    if c == 1:
                        possible_moves.append(CheckersMove(p, i, True))
        if possible_moves:
            return possible_moves
        for p in pieces:
            if p.is_open():
                for i, c in enumerate(p.open):
                    if c == 1:
                        possible_moves.append(CheckersMove(p, i, False))
        return possible_moves

    def get_winner(self):
        if not self.p1_pieces:
            return -1
        if not self.p2_pieces:
            return 1
        p1_over = True
        p2_over = True
        for p in self.p1_pieces:
            if p.is_capture() or p.is_open():
                p1_over = False
                break
        for p in self.p2_pieces:
            if p.is_capture() or p.is_open():
                p2_over = False
                break
        assert (p1_over and p2_over) is False
        if p1_over:
            return -1
        if p2_over:
            return 1
        return 0

    def get_piece(self, player:int, position):
        if player == 1:
            for p in self.p1_pieces:
                if p.x == position[0] and p.y == position[1]:
                    return p
        else:
            for p in self.p2_pieces:
                if p.x == position[0] and p.y == position[1]:
                    return p

    def move_piece(self, move):
        p = self.get_piece(move.player, move.position)
        p.x, p.y = move.get_result_position()
        if move.is_capture:
            enemy_p = self.get_piece(-move.player, move.get_capture_position())
            if move.player == 1:
                self.p2_pieces.remove(enemy_p)
            else:
                self.p1_pieces.remove(enemy_p)
        if (move.player == 1 and p.y == CheckersGameState.__checker_board_size-1) or \
           (move.player == -1 and p.y == 0):
            p.piece_type *= 2

        self.update_board()
        self.update_turn(move)

    def get_data(self):
        pass

class CheckersMove:
    def __init__(self, piece:CheckerPiece, dir_index:int, is_capture:bool):
        self.player = np.sign(piece.piece_type)
        self.position = (piece.x, piece.y)
        self.dir_index = dir_index
        self.is_capture = is_capture

    def get_result_position(self):
        d = CheckersGameState.checker_directions[self.dir_index]
        if self.is_capture:
            return (self.position[0] + d[0] * 2, self.position[1] + d[1] * 2)
        return (self.position[0] + d[0], self.position[1] + d[1])

    def get_capture_position(self):
        if self.is_capture is False:
            return (0, 0)
        d = CheckersGameState.checker_directions[self.dir_index]
        return (self.position[0] + d[0], self.position[1] + d[1])

    def get_data(self):
        pass

class CheckersSimulator:
    def apply_action(self, state:CheckersGameState, move:CheckersMove):
        state.move_piece(move)
        return state

class CheckersSimulatorPredictor(CheckersSimulator):
    def apply_action(self, state:CheckersGameState, move:CheckersMove):
        pass