import multiprocessing as mp
from board_game_tester import BoardGameTester
from mcts import UCT, NoveltyUCT
from boardGames.hex.hex import HexGameState, PutPieceAction, HexSimulatorPredictor, HexSimulator
from boardGames.hex.hex_wrappers import HexGameSimulatorWrapper
from boardGames.othello.OthelloWrappers import OthelloSimulator, OthelloPredictor
from ML import StateNoveltyPredictor, StatePredictor

HEX_BOARD_SIZE  = 7
HEX_STATE_SIZE  = HexGameSimulatorWrapper.get_state_data_len(HEX_BOARD_SIZE)
HEX_ACTION_SIZE = HexGameSimulatorWrapper.get_action_data_len(HEX_BOARD_SIZE)
HEX_TEST_FOLDER = "tests/tests_hex/"
HEX_PROX_MODEL  = "HexModel_{0}x{0}".format(HEX_BOARD_SIZE)

OTHELLO_BOARD_SIZE  = 6
OTHELLO_STATE_SIZE  = OthelloSimulator.get_state_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_ACTION_SIZE = OthelloSimulator.get_action_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_TEST_FOLDER = "tests/tests_othello/"
OTHELLO_PROX_MODEL  = "OthelloModel_{0}x{0}".format(OTHELLO_BOARD_SIZE)

def get_predictor_hex():
    return StatePredictor(dimensions=[HEX_STATE_SIZE + HEX_ACTION_SIZE,
                                      HEX_STATE_SIZE],
                          learningRate=1e-4)

def get_novelty_predictor_hex():
    return StateNoveltyPredictor(dimensions=[HEX_STATE_SIZE,
                                             HEX_STATE_SIZE * 6,
                                             HEX_STATE_SIZE * 6,
                                             HEX_STATE_SIZE],
                                 learningRate=1e-4)

def get_predictor_othello():
    return StatePredictor(dimensions=[OTHELLO_STATE_SIZE + OTHELLO_ACTION_SIZE,
                                      OTHELLO_STATE_SIZE + OTHELLO_ACTION_SIZE * 6,
                                      OTHELLO_STATE_SIZE + OTHELLO_ACTION_SIZE * 6,
                                      OTHELLO_STATE_SIZE + OTHELLO_ACTION_SIZE * 6,
                                      OTHELLO_STATE_SIZE],
                          learningRate=1e-4)

def get_novelty_predictor_othello():
    return StateNoveltyPredictor(dimensions=[OTHELLO_STATE_SIZE,
                                             OTHELLO_STATE_SIZE * 6,
                                             OTHELLO_STATE_SIZE * 6,
                                             OTHELLO_STATE_SIZE],
                                 learningRate=1e-4)

def create_rnd_mcts(simulator, test_folder, rndName, noveltyBonus, rollouts=100):
    return NoveltyUCT(simulator=simulator,
                      rnd=StateNoveltyPredictor.FromModel(test_folder + rndName), rollouts=rollouts, noveltyBonus=noveltyBonus)

def create_regular_mcts(simulator, rollouts=100):
    return UCT(simulator=simulator, rollouts=rollouts)

# FOR WINDOWS:
if __name__ == '__main__':
    mp.freeze_support()

    HEX_TESTER = BoardGameTester(game_simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)),
                                 board_size = HEX_BOARD_SIZE,
                                 test_folder = HEX_TEST_FOLDER,
                                 aproximate_game_simulator=None)
                                 #aproximate_game_simulator=HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(predictor=StatePredictor.FromModel(HEX_TEST_FOLDER + HEX_PROX_MODEL))))

    OTHELLO_TESTER = BoardGameTester(game_simulator=OthelloSimulator(), 
                                     board_size = OTHELLO_BOARD_SIZE,
                                     test_folder = OTHELLO_TEST_FOLDER,
                                     aproximate_game_simulator=None)
                                     #aproximate_game_simulator=OthelloSimulator(StatePredictor.FromModel(OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL)))

    HEX_TESTER.play_games_async(game_count=10000, process_count=8)
    OTHELLO_TESTER.play_games_async(game_count=10000, process_count=8)
    #HEX_TESTER.play_games_async(player1=create_regular_mcts(simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)), rollouts=100))
    #OTHELLO_TESTER.play_games_async(player1=create_regular_mcts(simulator=OthelloSimulator(), rollouts=100))

    #HEX_TESTER.create_dataset("Hex_speed_test_100", 100, True)
    #OTHELLO_TESTER.create_dataset("Othello_test_10", 10, True)
    #OTHELLO_TESTER.create_dataset("Othello_train_1-000", 1000, True)
    #OTHELLO_TESTER.create_dataset("Othello_test_10000_3", 100000, True)
    #OTHELLO_TESTER.create_predictor(get_predictor_othello(), name=OTHELLO_PROX_MODEL, train_file_name="Othello_train_1-000", batch_size=10000, epoch_count=10)
    #OTHELLO_TESTER.test_predictor(name=OTHELLO_PROX_MODEL, test_file_name="Othello_train_10-000")