import pygame
import random
import numpy as np
import pygame.draw as draw
import checkers
import pickle
from checkers import CheckersGameState, CheckerSimulator

__resolution     = (800, 800)
__board_dark     = (42, 44, 43)
__board_clear    = (55, 65, 64)
__p1_piece_color = (220, 53, 34)
__p2_piece_color = (217, 203, 158)

#------------------------------------#
space_dim = (__resolution[0] / 8, __resolution[1] / 8)
piece_dim = (space_dim[0] * 0.4).__round__()
#------------------------------------#

def Show(states:CheckersGameState):
    pygame.init()
    screen = pygame.display.set_mode(__resolution)
    clock = pygame.time.Clock()

    background = pygame.Surface(screen.get_size())

    running = True
    curr_game_state = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    curr_game_state -= 1
                elif event.key == pygame.K_RIGHT:
                    curr_game_state += 1
                    curr_game_state = curr_game_state % len(states)
    
        background.fill((0,0,0))
        DrawBoard(background)
        DrawPieces(background, states[curr_game_state])
        screen.blit(background, (0,0))
        pygame.display.update()
        clock.tick(60)
    pygame.quit()

def DrawPieces(background, state):
    for p in state.p1_pieces:
        position = ((p.x * space_dim[0] + piece_dim * 1.25).__round__(), (p.y * space_dim[1] + piece_dim * 1.25).__round__())
        draw.circle(background, __p1_piece_color, position, piece_dim)
    for p in state.p2_pieces:
        position = ((p.x * space_dim[0] + piece_dim * 1.25).__round__(), (p.y * space_dim[1] + piece_dim * 1.25).__round__())
        draw.circle(background, __p2_piece_color, position, piece_dim)

def DrawBoard(background):
    use_clear = False
    for y in range(0, 8):
        use_clear = not use_clear
        for x in range(0, 8):
            if use_clear:
                color = __board_clear
            else:
                color = __board_dark
            draw.rect(background, color, (space_dim[0] * x, space_dim[1] * y, space_dim[0], space_dim[1]))
            use_clear = not use_clear

if __name__ == '__main__':
    current_state = CheckersGameState.get_initial_board(1)
    states        = []
    simulator     = CheckerSimulator()
    turn = 0
    while True:
        #Show([current_state])
        states.append(pickle.loads(pickle.dumps(current_state)))
        winner = current_state.get_winner()
        if winner != 0:
            break
        possible_moves = current_state.get_possible_moves(current_state.current_player)
        if not possible_moves:
            winner = current_state.get_winner()
        selected_move  = random.choice(possible_moves)
        current_state  = simulator.apply_action(current_state, selected_move)
        turn += 1
        print(turn)
    Show(states)