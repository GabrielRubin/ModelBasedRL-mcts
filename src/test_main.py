from board_game_tester import BoardGameTester
from mcts import UCT, NoveltyUCT
from boardGames.hex.hex import HexGameState, PutPieceAction, HexSimulator, HexSimulatorPredictor
from boardGames.hex.hex_wrappers import HexGameSimulatorWrapper
from boardGames.othello.OthelloWrappers import OthelloSimulator, OthelloPredictor
from ml.state_predictor_model import HexStatePredictor, OthelloStatePredictor
from ml.rnd_model import HexRND, OthelloRND

HEX_BOARD_SIZE  = 7
HEX_STATE_SIZE  = HexGameSimulatorWrapper.get_state_data_len(HEX_BOARD_SIZE)
HEX_ACTION_SIZE = HexGameSimulatorWrapper.get_action_data_len(HEX_BOARD_SIZE)
HEX_TEST_FOLDER = "tests/tests_hex/"
HEX_PROX_MODEL  = "HexModel_{0}x{0}".format(HEX_BOARD_SIZE)
HEX_RND_MODEL   = "Hex_rnd_Model_{0}x{0}".format(HEX_BOARD_SIZE)

OTHELLO_BOARD_SIZE  = 6
OTHELLO_STATE_SIZE  = OthelloSimulator.get_state_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_ACTION_SIZE = OthelloSimulator.get_action_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_TEST_FOLDER = "tests/tests_othello/"
OTHELLO_PROX_MODEL  = "OthelloModel_{0}x{0}".format(OTHELLO_BOARD_SIZE)
OTHELLO_RND_MODEL   = "Hex_rnd_Model_{0}x{0}".format(HEX_BOARD_SIZE)

def get_hex_sim():
    return HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE))

def get_hex_approx_sim():
    return HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL)))

def get_othello_sim():
    return OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000))

def create_mcts(simulator, rollouts=100):
    return UCT(simulator=simulator, rollouts=rollouts)

def create_rnd_mcts(simulator, rnd_name, novelty_bonus=1, rollouts=100):
    return NoveltyUCT(simulator=simulator,
                      rnd=HexRND.from_file(HEX_TEST_FOLDER + rnd_name), rollouts=rollouts, novelty_bonus=novelty_bonus)

if __name__ == '__main__':
    # FOR WINDOWS:
    #mp.freeze_support()
    #FOR UBUNTU DEBUG:
    #mp.set_start_method('spawn') #beware! this explodes memory!

    HEX_TESTER = BoardGameTester(game_simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)),
                                 board_size = HEX_BOARD_SIZE,
                                 test_folder = HEX_TEST_FOLDER,
                                 aproximate_game_simulator=HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                                                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL))))

    OTHELLO_TESTER = BoardGameTester(game_simulator=OthelloSimulator(), 
                                     board_size = OTHELLO_BOARD_SIZE,
                                     test_folder = OTHELLO_TEST_FOLDER,
                                     aproximate_game_simulator=None)#OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000)))

    #HEX_TESTER.play_games_async(game_count=10000, process_count=8)
    #OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=get_othello_sim()), game_count=100)

    #OTHELLO_TESTER.create_approx_dataset("Othello_fake_train_1-000", rollout_count=1000)
    #OTHELLO_TESTER.create_rnd_predictor(OthelloRND(OTHELLO_STATE_SIZE), name=OTHELLO_RND_MODEL, train_file_name="Othello_train_1-000", batch_size=10000, epoch_count=250)
    
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_train_10-000")
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_train_1-000")
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_fake_train_1-000")

    #HEX_TESTER.play_games_async(player1=create_rnd_mcts(simulator=get_hex_approx_sim(), rnd_name=HEX_RND_MODEL, rollouts=100),
    #                            player2=create_mcts(simulator=get_hex_sim(), rollouts=100), game_count=400)

    #HEX_TESTER.play_games_async(player1=create_mcts(simulator=get_hex_approx_sim(), rollouts=100),
    #                            player2=create_mcts(simulator=get_hex_sim(), rollouts=100), game_count=400)

    #OTHELLO_TESTER.play_games_async(player1=create_regular_mcts(simulator=OthelloSimulator(), rollouts=100))

    #HEX_TESTER.create_dataset("hex_train_17", rollout_count=17, override=True)
    #HEX_TESTER.create_predictor(HexStatePredictor(HEX_STATE_SIZE, HEX_ACTION_SIZE), name=HEX_PROX_MODEL, train_file_name="hex_train_17", epoch_count=500, batch_size=100000)
    #HEX_TESTER.test_predictor(predictor_cls=HexStatePredictor, name=HEX_PROX_MODEL, test_file_name="hex_train_1000")

    #HEX_TESTER.create_dataset("hex_rnd_train_100")
    #HEX_TESTER.create_dataset("hex_rnd_diff_train_100")
    #HEX_TESTER.create_approx_dataset("hex_fake_train_100")

    #HEX_TESTER.create_rnd_predictor(HexRND(HEX_STATE_SIZE), name=HEX_RND_MODEL, train_file_name="hex_rnd_train_100", epoch_count=250, batch_size=10000)

    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL, test_file_name="hex_rnd_diff_train_100")
    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL, test_file_name="hex_rnd_train_100")
    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL, test_file_name="hex_fake_train_100")

    #OTHELLO_TESTER.create_dataset("othello_test_10", 10, True)
    #HEX_TESTER.create_dataset("hex_test_10", 10, True)

    #HEX_TESTER.create_dataset("hex_train_100", 100, True)
    #HEX_TESTER.create_dataset("hex_train_1000", 1000, True)
    #HEX_TESTER.create_dataset("hex_test_100000", 100000, True)

    #HEX_TESTER.create_dataset("Hex_speed_test_100", 100, True)
    #OTHELLO_TESTER.create_dataset("Othello_train_1-000", 1000, True)
    #OTHELLO_TESTER.create_dataset("Othello_test_10000_3", 100000, True)

    OTHELLO_TESTER.create_predictor(OthelloStatePredictor(OTHELLO_STATE_SIZE, OTHELLO_ACTION_SIZE), name=OTHELLO_PROX_MODEL, train_file_name="Othello_train_1-000", batch_size=1000, epoch_count=500)
    OTHELLO_TESTER.test_predictor(predictor_cls=OthelloStatePredictor,name=OTHELLO_PROX_MODEL, test_file_name="Othello_train_10-000")
    
    #HEX_TESTER.create_predictor(HexStatePredictor(HEX_STATE_SIZE, HEX_ACTION_SIZE), name=HEX_PROX_MODEL, train_file_name="hex_train_100", epoch_count=250)
    #HEX_TESTER.test_predictor(predictor_cls=HexStatePredictor, name=HEX_PROX_MODEL, test_file_name="hex_train_1000")
