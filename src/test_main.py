import multiprocessing as mp
from board_game_tester import BoardGameTester
from mcts import UCT, NoveltyUCT, NoveltyUCTB, NoveltyUCTC
from boardGames.hex.hex import HexGameState, PutPieceAction, HexSimulator, HexSimulatorPredictor
from boardGames.hex.hex_wrappers import HexGameSimulatorWrapper
from boardGames.othello.OthelloWrappers import OthelloSimulator, OthelloPredictor
from boardGames.checkers.checkers_wrappers import CheckersGameSimulatorWrapper
from boardGames.checkers.checkers import CheckersSimulator, CheckersSimulatorPredictor
from ml.state_predictor_model import HexStatePredictor, OthelloStatePredictor, CheckersStatePredictor
from ml.rnd_model import HexRND, OthelloRND, CheckersRND
from ml.transition_classifier import HexTransitionClassifier

#TEST CHECKERS
#from boardGames.game_trial import DebugTrial
#from boardGames.checkers.checker_viewer import Show

SAVE_FILE = "tests/results.txt"

HEX_BOARD_SIZE    = 7
HEX_STATE_SIZE    = HexGameSimulatorWrapper.get_state_data_len(HEX_BOARD_SIZE)
HEX_ACTION_SIZE   = HexGameSimulatorWrapper.get_action_data_len(HEX_BOARD_SIZE)
HEX_TEST_FOLDER   = "tests/tests_hex/"
HEX_PROX_MODEL    = "HexModel_{0}x{0}".format(HEX_BOARD_SIZE)
HEX_RND_MODEL     = "Hex_rnd_Model_{0}x{0}".format(HEX_BOARD_SIZE)
HEX_RND_MODEL_2   = "Hex_rnd_Model_{0}x{0}_2".format(HEX_BOARD_SIZE)
HEX_RND_MODEL_F   = "Hex_rnd"
HEX_PROX_MODEL_15 = "HexModel_{0}x{0}_015".format(HEX_BOARD_SIZE)
HEX_PROX_MODEL_45 = "HexModel_{0}x{0}_045".format(HEX_BOARD_SIZE)
HEX_PROX_MODEL_75 = "HexModel_{0}x{0}_075".format(HEX_BOARD_SIZE)

OTHELLO_BOARD_SIZE     = 6
OTHELLO_STATE_SIZE     = OthelloSimulator.get_state_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_ACTION_SIZE    = OthelloSimulator.get_action_data_len(OTHELLO_BOARD_SIZE)
OTHELLO_TEST_FOLDER    = "tests/tests_othello/"
OTHELLO_PROX_MODEL     = "pred_othello_073"
OTHELLO_PROX_MODEL_95  = "pred_othello_095"
OTHELLO_PROX_MODEL_925  = "pred_othello_0925"
OTHELLO_PROX_MODEL_85  = "pred_othello_087"
OTHELLO_PROX_MODEL_75  = "new_pred_othello_073"
OTHELLO_PROX_MODEL_45  = "pred_othello_045"
OTHELLO_PROX_MODEL_15  = "pred_othello_015"
OTHELLO_RND_MODEL      = "Othello_rnd_Model_5"

CHECKERS_BOARD_SIZE    = 8
CHECKERS_TEST_FOLDER   = "tests/tests_checkers/"
CHECKERS_PROX_MODEL_80 = "CheckersModel_80_2"
CHECKERS_PROX_MODEL_45 = "CheckersModel_45_2"
CHECKERS_PROX_MODEL_15 = "CheckersModel_15_2"
CHECKERS_PROX_MODEL_1 = "CheckersModel_1"
CHECKERS_RND_MODEL     = "CheckersRND_1"

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

def create_rnd_mcts(uct_class, simulator, rnd_name, novelty_bonus=1, rollouts=100):
    return uct_class(simulator=simulator,
                     rnd=HexRND.from_file(HEX_TEST_FOLDER + rnd_name), rollouts=rollouts, novelty_bonus=novelty_bonus)

def create_rnd_mcts_othello(uct_class, simulator, rnd_name, novelty_bonus=1, rollouts=100):
    return uct_class(simulator=simulator,
                      rnd=OthelloRND.from_file(OTHELLO_TEST_FOLDER + rnd_name), rollouts=rollouts, novelty_bonus=novelty_bonus)

def create_rnd_mcts_checkers(uct_class, simulator, rnd_name, novelty_bonus=1, rollouts=100):
    return uct_class(simulator=simulator,
                    rnd=CheckersRND.from_file(CHECKERS_TEST_FOLDER + rnd_name), rollouts=rollouts, novelty_bonus=novelty_bonus)


#def test_checkers_trial():
#    #simulator = CheckersGameSimulatorWrapper(simulator=CheckersSimulator());
#    trial = DebugTrial(simulator=CHECKERS_TESTER.game_simulator, player1=create_mcts(CHECKERS_TESTER.aprox_simulator, rollouts=100), player2=create_rnd_mcts_checkers(simulator=CHECKERS_TESTER.aprox_simulator, rnd_name=CHECKERS_RND_MODEL, rollouts=100))
#    for states, winner, winner_type in trial.do_rollouts(5):
#        print("Winner = {0} - Type = {1}".format(winner, winner_type))
#        Show(states)

if __name__ == '__main__':
    # FOR WINDOWS:
    mp.freeze_support()
    #FOR UBUNTU DEBUG:
    mp.set_start_method('spawn')

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
                                     aproximate_game_simulator=None)

    CHECKERS_TESTER = BoardGameTester(game_simulator=CheckersGameSimulatorWrapper(simulator=CheckersSimulator()),
                                      board_size=CHECKERS_BOARD_SIZE,
                                      test_folder=CHECKERS_TEST_FOLDER,
                                      aproximate_game_simulator=None)

    #CHECKERS_TESTER.create_dataset("train_100-000", rollout_count=100000, override=True)
    #CHECKERS_TESTER.create_dataset("train_1-000", rollout_count=1000, override=True)
    #CHECKERS_TESTER.create_dataset("train_100", rollout_count=100, override=True)
    #CHECKERS_TESTER.create_predictor(CheckersStatePredictor(8*8*2, 8*8*2), name=CHECKERS_PROX_MODEL_80, train_file_name="train_1-000", epoch_count=70, batch_size=1000)
    #CHECKERS_TESTER.test_predictor(predictor_cls=CheckersStatePredictor, name=CHECKERS_PROX_MODEL_80, test_file_name="train_10-000")

    #OTHELLO_TESTER.create_predictor(OthelloStatePredictor(OTHELLO_STATE_SIZE, OTHELLO_ACTION_SIZE), name=OTHELLO_PROX_MODEL, train_file_name="Othello_train_1-000", batch_size=1000, epoch_count=12)

    #CHECKERS_TESTER.create_mixed_dataset("checkers_mixed_test", 1000, process_count=5)
    #CHECKERS_TESTER.create_rnd_predictor(CheckersRND(8*8*2), name=CHECKERS_RND_MODEL, train_file_name="train_10-000", batch_size=1500, epoch_count=200)
    #CHECKERS_TESTER.test_rnd_predictor(CheckersRND, name=CHECKERS_RND_MODEL, test_file_name="checkers_mixed_test")

    #checkers_sim_80 = CheckersGameSimulatorWrapper(simulator=CheckersSimulatorPredictor(predictor=CheckersStatePredictor.from_file(file_name=CHECKERS_TEST_FOLDER+CHECKERS_PROX_MODEL_80)))
    #checkers_sim_45 = CheckersGameSimulatorWrapper(simulator=CheckersSimulatorPredictor(predictor=CheckersStatePredictor.from_file(file_name=CHECKERS_TEST_FOLDER+CHECKERS_PROX_MODEL_45)))
    #checkers_sim_15 = CheckersGameSimulatorWrapper(simulator=CheckersSimulatorPredictor(predictor=CheckersStatePredictor.from_file(file_name=CHECKERS_TEST_FOLDER+CHECKERS_PROX_MODEL_15)))
    #checkers_sim_reg = CheckersGameSimulatorWrapper(simulator=CheckersSimulator())

    #CHECKERS_TESTER.play_games_async(player1=create_rnd_mcts_checkers(simulator=checkers_sim_80, rnd_name=CHECKERS_RND_MODEL, rollouts=100),
    #                                player2=create_rnd_mcts_checkers(simulator=checkers_sim_80, rnd_name=CHECKERS_RND_MODEL, rollouts=100),
    #                                game_count=30, process_count=5)

    #CHECKERS_TESTER.play_games_async(player1=create_rnd_mcts_checkers(simulator=checkers_sim_45, rnd_name=CHECKERS_RND_MODEL, rollouts=100),
    #                                 player2=create_rnd_mcts_checkers(simulator=checkers_sim_45, rnd_name=CHECKERS_RND_MODEL, rollouts=100),
    #                                 game_count=30, process_count=5)

    #CHECKERS_TESTER.play_games_async(player1=create_rnd_mcts_checkers(simulator=checkers_sim_15, rnd_name=CHECKERS_RND_MODEL, rollouts=100),
    #                                player2=create_rnd_mcts_checkers(simulator=checkers_sim_15, rnd_name=CHECKERS_RND_MODEL, rollouts=100),
    #                                game_count=30, process_count=5)

    #CHECKERS_TESTER.play_games_async(player1=create_mcts(simulator=checkers_sim_reg, rollouts=100),
    #                                player2=create_mcts(simulator=checkers_sim_45, rollouts=100),
    #                                game_count=100, process_count=5)

    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL, test_file_name="mixed_test2_1000")

    #HEX_TESTER.play_games(player1=create_mcts(simulator=get_hex_sim(), rollouts=100),
    #                      player2=None, game_count=100)

    #test_checkers_trial()

    #HEX_TESTER.play_games_async(game_count=10000, process_count=8)
    #OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=get_othello_sim()), game_count=100)

    #OTHELLO_TESTER.create_rnd_predictor(OthelloRND(OTHELLO_STATE_SIZE), name=OTHELLO_RND_MODEL, train_file_name="Othello_train_10-000", batch_size=1000, epoch_count=200)
    #OTHELLO_TESTER.create_approx_dataset("Othello_fake_train_1-000", rollout_count=1000)
    
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_train_10-000")
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_train_1-000")
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="Othello_fake_train_1-000")

    #OTHELLO_TESTER.create_mixed_dataset("othello_mixed_test_1000", rollout_count=1000)
    #OTHELLO_TESTER.test_rnd_predictor(OthelloRND, name=OTHELLO_RND_MODEL, test_file_name="othello_mixed_test_1000_2")

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
 
    #OTHELLO_TESTER.play_games_async(player1=create_rnd_mcts_othello(simulator=OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000)), rnd_name=OTHELLO_RND_MODEL, rollouts=100),
    #                                player2=create_mcts(simulator=OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL, 50000)), rollouts=100),
    #                                game_count=100, process_count=5)


    #othello_015 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_15, 50000))
    #othello_045 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_45, 50000))
    #othello_075 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_75, 50000))
    #othello_085 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_85, 50000))
    #othello_095 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_95, 50000))
    #othello_0925 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_925, 50000))

    #OTHELLO_TESTER.play_games_async(player1=create_rnd_mcts_othello(simulator=othello_015, rnd_name=OTHELLO_RND_MODEL, rollouts=100),
    #                                player2=create_rnd_mcts_othello(simulator=othello_015, rnd_name=OTHELLO_RND_MODEL, rollouts=100),
    #                                game_count=30, process_count=5)

    #OTHELLO_TESTER.play_games_async(player1=create_rnd_mcts_othello(simulator=othello_045, rnd_name=OTHELLO_RND_MODEL, rollouts=100),
    #                                player2=create_mcts(simulator=othello_045, rollouts=100),
    #                                game_count=20, process_count=5)

    #OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=OthelloSimulator(), rollouts=100),
    #                                player2=create_mcts(simulator=othello_075, rollouts=100),
    #                                game_count=20, process_count=5)

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
    #OTHELLO_TESTER.create_dataset("Othello_train_100-000", 100000, True)

    # latest enabled
    #OTHELLO_TESTER.create_predictor(OthelloStatePredictor(OTHELLO_STATE_SIZE, OTHELLO_ACTION_SIZE), name=OTHELLO_PROX_MODEL_15, train_file_name="Othello_train_1-000", batch_size=1000, epoch_count=4)
    #OTHELLO_TESTER.test_predictor(predictor_cls=OthelloStatePredictor,name=OTHELLO_PROX_MODEL_15, test_file_name="Othello_train_100-000")
    
    #HEX_TESTER.create_predictor(HexStatePredictor(HEX_STATE_SIZE, HEX_ACTION_SIZE), name=HEX_PROX_MODEL, train_file_name="hex_train_{0}".format(trainSize), epoch_count=250)
    #HEX_TESTER.test_predictor(predictor_cls=HexStatePredictor, name=HEX_PROX_MODEL, test_file_name="hex_train_1000")

    #HEX_TESTER.create_mixed_dataset("mixed_test_1", rollout_count=1)
    #HEX_TESTER.create_mixed_dataset("mixed_test_100", rollout_count=100)
    #HEX_TESTER.create_mixed_dataset("mixed_test2_1000", rollout_count=1000)
    #HEX_TESTER.create_transition_classifier(classifier=HexTransitionClassifier(HEX_STATE_SIZE), name="test_01", train_file_name="mixed_test_100", batch_size=10000, epoch_count=500)
    #HEX_TESTER.test_transition_classifier(HexTransitionClassifier, name="test_01", test_file_name="mixed_test2_1000")
    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL, test_file_name="mixed_test2_1000")
    #HEX_TESTER.test_rnd_predictor_2(HexRND, name=HEX_RND_MODEL_2, test_file_name="mixed_test2_1000")

    #HEX_TESTER.create_rnd_predictor(HexRND(HEX_STATE_SIZE), name=HEX_RND_MODEL_F, train_file_name="hex_train_1000", epoch_count=250, batch_size=10000)
    #HEX_TESTER.test_rnd_predictor(HexRND, name=HEX_RND_MODEL_F, test_file_name="mixed_test2_1000")

    ### FINAL TESTS FTW #####

    othello_015 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_15, 50000))
    othello_045 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_45, 50000))
    othello_075 = OthelloSimulator(predictor=OthelloPredictor(OTHELLO_BOARD_SIZE, OTHELLO_TEST_FOLDER + OTHELLO_PROX_MODEL_75, 50000))

    OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=OthelloSimulator(), rollouts=100),
                                    player2=create_mcts(simulator=othello_015, rollouts=100),
                                    game_count=50, process_count=6,
                                    save_file=SAVE_FILE, run_name="OTHELLO > mcts vs mcts 15")
    OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=OthelloSimulator(), rollouts=100),
                                    player2=create_mcts(simulator=othello_045, rollouts=100),
                                    game_count=50, process_count=6,
                                    save_file=SAVE_FILE, run_name="OTHELLO > mcts vs mcts 45")
    OTHELLO_TESTER.play_games_async(player1=create_mcts(simulator=OthelloSimulator(), rollouts=100),
                                    player2=create_mcts(simulator=othello_075, rollouts=100),
                                    game_count=50, process_count=6,
                                    save_file=SAVE_FILE, run_name="OTHELLO > mcts vs mcts 75")

    checkers_sim_80 = CheckersGameSimulatorWrapper(simulator=CheckersSimulatorPredictor(predictor=CheckersStatePredictor.from_file(file_name=CHECKERS_TEST_FOLDER+CHECKERS_PROX_MODEL_80)))
    checkers_sim_45 = CheckersGameSimulatorWrapper(simulator=CheckersSimulatorPredictor(predictor=CheckersStatePredictor.from_file(file_name=CHECKERS_TEST_FOLDER+CHECKERS_PROX_MODEL_45)))
    checkers_sim_15 = CheckersGameSimulatorWrapper(simulator=CheckersSimulatorPredictor(predictor=CheckersStatePredictor.from_file(file_name=CHECKERS_TEST_FOLDER+CHECKERS_PROX_MODEL_15)))

    CHECKERS_TESTER.play_games_async(player1=create_mcts(simulator=CheckersGameSimulatorWrapper(simulator=CheckersSimulator()), rollouts=100),
                                    player2=create_mcts(simulator=checkers_sim_80, rollouts=100),
                                    game_count=50, process_count=6,
                                    save_file=SAVE_FILE, run_name="CHECKERS > mcts vs mcts 75")

    CHECKERS_TESTER.play_games_async(player1=create_mcts(simulator=CheckersGameSimulatorWrapper(simulator=CheckersSimulator()), rollouts=100),
                                     player2=create_mcts(simulator=checkers_sim_45, rollouts=100),
                                     game_count=50, process_count=6,
                                     save_file=SAVE_FILE, run_name="CHECKERS > mcts vs mcts 45")

    CHECKERS_TESTER.play_games_async(player1=create_mcts(simulator=CheckersGameSimulatorWrapper(simulator=CheckersSimulator()), rollouts=100),
                                    player2=create_mcts(simulator=checkers_sim_15, rollouts=100),
                                    game_count=50, process_count=6,
                                    save_file=SAVE_FILE, run_name="CHECKERS > mcts 15 vs mcts 15")