import multiprocessing as mp
from board_game_tester import BoardGameTester
from mcts import UCT, NoveltyUCT
from boardGames.hex.hex import HexGameState, PutPieceAction, HexSimulator, HexSimulatorPredictor
from boardGames.hex.hex_wrappers import HexGameSimulatorWrapper
from boardGames.othello.OthelloWrappers import OthelloSimulator, OthelloPredictor
from boardGames.checkers.checkers_wrappers import CheckersGameSimulatorWrapper
from boardGames.checkers.checkers import CheckersSimulator, CheckersSimulatorPredictor
from ml.state_predictor_model import HexStatePredictor, OthelloStatePredictor
from ml.rnd_model import HexRND, OthelloRND
from ml.transition_classifier import HexTransitionClassifier

#TEST CHECKERS
from boardGames.game_trial import DebugTrial
from boardGames.checkers.checker_viewer import Show

HEX_BOARD_SIZE    = 7
HEX_STATE_SIZE    = HexGameSimulatorWrapper.get_state_data_len(HEX_BOARD_SIZE)
HEX_ACTION_SIZE   = HexGameSimulatorWrapper.get_action_data_len(HEX_BOARD_SIZE)
HEX_TEST_FOLDER   = "tests/tests_hex/"
HEX_PROX_MODEL    = "HexModel_{0}x{0}".format(HEX_BOARD_SIZE)
HEX_RND_MODEL     = "Hex_rnd_Model_{0}x{0}".format(HEX_BOARD_SIZE)
HEX_RND_MODEL_2   = "Hex_rnd_Model_{0}x{0}_2".format(HEX_BOARD_SIZE)
HEX_PROX_MODEL_15 = "HexModel_{0}x{0}_015".format(HEX_BOARD_SIZE)
HEX_PROX_MODEL_45 = "HexModel_{0}x{0}_045".format(HEX_BOARD_SIZE)
HEX_PROX_MODEL_75 = "HexModel_{0}x{0}_075".format(HEX_BOARD_SIZE)

OTHELLO_BOARD_SIZE  = 6
OTHELLO_STATE_SIZE  = OthelloSimulator.get_state_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_ACTION_SIZE = OthelloSimulator.get_action_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_TEST_FOLDER = "tests/tests_othello/"
OTHELLO_PROX_MODEL  = "OthelloModel_{0}x{0}__acc_73".format(OTHELLO_BOARD_SIZE)
OTHELLO_RND_MODEL   = "Hex_rnd_Model_{0}x{0}".format(HEX_BOARD_SIZE)

def get_hex_sim():
    return HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE))

def get_hex_approx_sim():
    return HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL)))

def get_hex_approx_sim_15():
    return HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL_15)))

def get_hex_approx_sim_45():
    return HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL_45)))

def get_hex_approx_sim_75():
    return HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL_75)))

def get_othello_sim():
    return OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000))

def create_mcts(simulator, rollouts=100):
    return UCT(simulator=simulator, rollouts=rollouts)

def create_rnd_mcts(simulator, rnd_name, novelty_bonus=1, rollouts=100):
    return NoveltyUCT(simulator=simulator,
                      rnd=HexRND.from_file(HEX_TEST_FOLDER + rnd_name), rollouts=rollouts, novelty_bonus=novelty_bonus)

def test_checkers_trial():
    simulator = CheckersGameSimulatorWrapper(simulator=CheckersSimulator());
    trial = DebugTrial(simulator=simulator, player1=create_mcts(simulator, rollouts=1000))
    for states, winner in trial.do_rollouts(5):
        print("Winner = {0}".format(winner))
        Show(states)

if __name__ == '__main__':
    # FOR WINDOWS:
    mp.freeze_support()
    #FOR UBUNTU DEBUG:
    #mp.set_start_method('spawn')

    HEX_TESTER = BoardGameTester(game_simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)),
                                 board_size = HEX_BOARD_SIZE,
                                 test_folder = HEX_TEST_FOLDER,
                                 aproximate_game_simulator=HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                                                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL))))

    HEX_TESTER_15 = BoardGameTester(game_simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)),
                                 board_size = HEX_BOARD_SIZE,
                                 test_folder = HEX_TEST_FOLDER,
                                 aproximate_game_simulator=HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                                                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL_15))))

    HEX_TESTER_45 = BoardGameTester(game_simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)),
                                 board_size = HEX_BOARD_SIZE,
                                 test_folder = HEX_TEST_FOLDER,
                                 aproximate_game_simulator=HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                                                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL_45))))

    HEX_TESTER_75 = BoardGameTester(game_simulator=HexGameSimulatorWrapper(simulator=HexSimulator(HEX_BOARD_SIZE)),
                                 board_size = HEX_BOARD_SIZE,
                                 test_folder = HEX_TEST_FOLDER,
                                 aproximate_game_simulator=HexGameSimulatorWrapper(simulator=HexSimulatorPredictor(HEX_BOARD_SIZE, 
                                                                                   predictor=HexStatePredictor.from_file(file_name=HEX_TEST_FOLDER+HEX_PROX_MODEL_75))))

    OTHELLO_TESTER = BoardGameTester(game_simulator=OthelloSimulator(), 
                                     board_size = OTHELLO_BOARD_SIZE,
                                     test_folder = OTHELLO_TEST_FOLDER,
                                     aproximate_game_simulator=OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000)))

    #HEX_TESTER.play_games(player1=create_mcts(simulator=get_hex_sim(), rollouts=100),
    #                      player2=None, game_count=100)

    test_checkers_trial()

    #HEX_TESTER.play_games_async(game_count=10000, process_count=8)
    #OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=get_othello_sim()), game_count=100)

    #OTHELLO_TESTER.create_rnd_predictor(OthelloRND(OTHELLO_STATE_SIZE), name=OTHELLO_RND_MODEL, train_file_name="Othello_train_1-000", batch_size=10000, epoch_count=250)
    #OTHELLO_TESTER.create_approx_dataset("Othello_fake_train_1-000", rollout_count=1000)
    
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_train_10-000")
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_train_1-000")
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_fake_train_1-000")

    #OTHELLO_TESTER.create_mixed_dataset("othello_mixed_test_1000", rollout_count=1000)
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="othello_mixed_test_1000")

    #HEX_TESTER.create_rnd_predictor(HexRND(HEX_STATE_SIZE), name=HEX_RND_MODEL, train_file_name="hex_rnd_train_100", epoch_count=250, batch_size=10000)
    #HEX_TESTER.play_games_async(player1=create_rnd_mcts(simulator=get_hex_approx_sim(), rnd_name=HEX_RND_MODEL, rollouts=100),
    #                            player2=create_mcts(simulator=get_hex_sim(), rollouts=100), game_count=100, process_count=4)

    #HEX_TESTER_15.play_games_async(player1=create_rnd_mcts(simulator=get_hex_approx_sim_15(), rnd_name=HEX_RND_MODEL, rollouts=100),
    #                               player2=create_mcts(simulator=get_hex_sim(), rollouts=100), game_count=200, process_count=4)

    #HEX_TESTER_15.play_games_async(player1=create_rnd_mcts(simulator=get_hex_approx_sim_15(), rnd_name=HEX_RND_MODEL, rollouts=1000),
    #                            player2=create_mcts(simulator=get_hex_sim(), rollouts=1000), game_count=200, process_count=4)

    #HEX_TESTER.play_games_async(player1=create_mcts(simulator=get_hex_approx_sim(), rollouts=1000),
    #                            player2=create_mcts(simulator=get_hex_sim(), rollouts=1000), game_count=400, process_count=6)

    #HEX_TESTER.play_games_async(player1=create_mcts(simulator=get_hex_approx_sim(), rollouts=100),
    #                            player2=create_mcts(simulator=get_hex_sim(), rollouts=100), game_count=400, process_count=6)
 
    #OTHELLO_TESTER.play_games(player1=create_mcts(simulator=OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000)), rollouts=100))
                                    #player2=create_rnd_mcts(simulator=OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000)),
                                    #                        rnd_name='')

    #HEX_TESTER.create_dataset("hex_train_17", rollout_count=17, override=True)
    #HEX_TESTER.create_predictor(HexStatePredictor(HEX_STATE_SIZE, HEX_ACTION_SIZE), name=HEX_PROX_MODEL, train_file_name="hex_train_17", epoch_count=500, batch_size=100000)
    #HEX_TESTER.test_predictor(predictor_cls=HexStatePredictor, name=HEX_PROX_MODEL, test_file_name="hex_train_1000")

    #HEX_TESTER.create_dataset("hex_rnd_train_100")
    #HEX_TESTER.create_dataset("hex_rnd_diff_train_100")
    #HEX_TESTER.create_approx_dataset("hex_fake_train_100")

    #HEX_TESTER.create_rnd_predictor_2(HexRND(HEX_STATE_SIZE), name=HEX_RND_MODEL_2, train_file_name="hex_rnd_train_100", epoch_count=250, batch_size=10000)

    #OTHELLO_TESTER.create_dataset("othello_test_10", 10, True)
    #HEX_TESTER.create_dataset("hex_test_10", 10, True)

    #trainSize = 25
    #HEX_TESTER.create_dataset("hex_train_{0}".format(trainSize), trainSize, True)
    #HEX_TESTER.create_dataset("hex_train_1000", 1000, True)
    #HEX_TESTER.create_dataset("hex_test_100000", 100000, True)

    #HEX_TESTER.create_dataset("Hex_speed_test_100", 100, True)
    #OTHELLO_TESTER.create_dataset("Othello_train_1-000", 1000, True)
    #OTHELLO_TESTER.create_dataset("Othello_test_10000_3", 100000, True)

    # latest enabled
    # OTHELLO_TESTER.create_predictor(OthelloStatePredictor(OTHELLO_STATE_SIZE, OTHELLO_ACTION_SIZE), name=OTHELLO_PROX_MODEL, train_file_name="Othello_train_1-000", batch_size=1000, epoch_count=500)
    # OTHELLO_TESTER.test_predictor(predictor_cls=OthelloStatePredictor,name=OTHELLO_PROX_MODEL, test_file_name="Othello_train_10-000")
    
    #HEX_TESTER.create_predictor(HexStatePredictor(HEX_STATE_SIZE, HEX_ACTION_SIZE), name=HEX_PROX_MODEL, train_file_name="hex_train_{0}".format(trainSize), epoch_count=250)
    #HEX_TESTER.test_predictor(predictor_cls=HexStatePredictor, name=HEX_PROX_MODEL, test_file_name="hex_train_1000")

    #HEX_TESTER.create_mixed_dataset("mixed_test_1", rollout_count=1)
    #HEX_TESTER.create_mixed_dataset("mixed_test_100", rollout_count=100)
    #HEX_TESTER.create_mixed_dataset("mixed_test2_1000", rollout_count=1000)
    #HEX_TESTER.create_transition_classifier(classifier=HexTransitionClassifier(HEX_STATE_SIZE), name="test_01", train_file_name="mixed_test_100", batch_size=10000, epoch_count=500)
    #HEX_TESTER.test_transition_classifier(HexTransitionClassifier, name="test_01", test_file_name="mixed_test2_1000")
    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL, test_file_name="mixed_test2_1000")
    #HEX_TESTER.test_rnd_predictor_2(HexRND, name=HEX_RND_MODEL_2, test_file_name="mixed_test2_1000")