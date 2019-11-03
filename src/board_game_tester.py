import multiprocessing as mp
import pickle
from tqdm import tqdm
from ml.state_predictor_model import _StatePredictor
from ml.rnd_model import _RandomNetworkDistillation
from ml.transition_classifier import _TransitionClassifier
from ml.data_manager import StateTransitionDataset, StateTransitionDatasetForNovelty, StateTransitionDatasetTEST
from dataset_creator import DatasetCreator, DatasetCreatorAsync
from boardGames.game_trial import DataCollectTrial, GameTrailBase, DataCollectWithSimCategory, DataCollectWithInvalidRolloutCount
from boardGames.game_simulator_base import GameSimulator

class BoardGameTester:
    def __init__(self, game_simulator:GameSimulator, board_size:int,
                 test_folder:str, aproximate_game_simulator:GameSimulator=None):
        self.game_simulator  = game_simulator
        self.board_size      = board_size
        self.test_folder     = test_folder
        self.aprox_simulator = aproximate_game_simulator

    def create_predictor(self, predictor_model:_StatePredictor, name:str, train_file_name:str,
                         batch_size=1000, epoch_count=1000):
        path = self.test_folder + train_file_name
        train_data = StateTransitionDataset(csv_path="{0}.csv".format(path),
                                            board_size=self.board_size*self.board_size)
        predictor_model.train_model(train_data, batch_size=batch_size, epoch_count=epoch_count)
        predictor_model.save(self.test_folder + name)

    def test_predictor(self, predictor_cls, name:str, test_file_name:str):
        path = self.test_folder + test_file_name
        predictor_model = predictor_cls.from_file(self.test_folder + name)
        test_data = StateTransitionDataset(csv_path="{0}.csv".format(path),
                                           board_size=self.board_size*self.board_size)
        predictor_model.test_module(test_data)

    def create_rnd_predictor(self, rnd_model:_RandomNetworkDistillation, name:str,
                             train_file_name:str, batch_size=1000, epoch_count=1000):
        path = self.test_folder + train_file_name
        train_data = StateTransitionDatasetForNovelty(csv_path="{0}.csv".format(path),
                                                      board_size=self.board_size*self.board_size)
        rnd_model.train_model(train_data, batch_size=batch_size, epoch_count=epoch_count)
        rnd_model.save(self.test_folder + name)

    def test_rnd_predictor(self, predictor_cls, name:str, test_file_name):
        path = self.test_folder + test_file_name
        novelty_model = predictor_cls.from_file(self.test_folder + name)
        test_data = StateTransitionDatasetTEST(csv_path="{0}.csv".format(path),
                                               board_size=self.board_size*self.board_size)
        novelty_model.test_module(test_data)

    def create_transition_classifier(self, classifier:_TransitionClassifier, name:str,
                             train_file_name:str, batch_size=1000, epoch_count=1000):
        path = self.test_folder + train_file_name
        train_data = StateTransitionDatasetTEST(csv_path="{0}.csv".format(path),
                                                board_size=self.board_size*self.board_size)
        classifier.train_model(train_data, batch_size=batch_size, epoch_count=epoch_count)
        classifier.save(self.test_folder + name)

    def test_transition_classifier(self, classifier_cls, name:str, test_file_name):
        path = self.test_folder + test_file_name
        novelty_model = classifier_cls.from_file(self.test_folder + name)
        test_data = StateTransitionDatasetTEST(csv_path="{0}.csv".format(path),
                                               board_size=self.board_size*self.board_size)
        novelty_model.test_module(test_data)

    def create_dataset(self, name:str, rollout_count:int=100, 
                       override:bool=False, process_count:int=-1):
        return self._create_dataset(self.game_simulator, name=name,
                                    rollout_count=rollout_count, override=override,
                                    process_count=process_count)

    def create_approx_dataset(self, name:str, rollout_count:int=100, 
                              override:bool=False, process_count:int=-1):
        if self.aprox_simulator is None:
            print("NO APROXIMATE SIMULATOR SET!")
            return None
        return self._create_dataset(self.aprox_simulator, name=name,
                                    rollout_count=rollout_count, override=override,
                                    process_count=process_count)

    def _create_dataset(self, simulator, name:str, rollout_count:int, 
                        override:bool, process_count:int):
        path = self.test_folder + name
        if rollout_count > 10:
            creator = DatasetCreatorAsync(DataCollectTrial(simulator), process_count=process_count)
        else:
            creator = DatasetCreator(DataCollectTrial(simulator))
        return creator.create_dataset(rollout_count, path, override)

    def create_mixed_dataset(self, name:str, rollout_count:int, process_count:int=-1):
        path = self.test_folder + name
        if rollout_count > 10:
            creator_real = DatasetCreatorAsync(DataCollectWithSimCategory(self.game_simulator), process_count=process_count)
            creator_fake = DatasetCreatorAsync(DataCollectWithSimCategory(self.aprox_simulator), process_count=process_count)
        else:
            creator_real = DatasetCreator(DataCollectWithSimCategory(self.game_simulator))
            creator_fake = DatasetCreator(DataCollectWithSimCategory(self.aprox_simulator))
        creator_real.create_dataset(rollout_count, path, override=False)
        creator_fake.create_dataset(rollout_count, path, override=False)
        

    def play_games(self, player1=None, player2=None, game_count:int=100):
        game_trail = GameTrailBase(self.game_simulator, player1, player2)
        pbar = tqdm(total=game_count)
        win_count = [0, 0]
        for winner in game_trail.do_rollouts(game_count):
            if winner == 1:
                win_count[0] += 1
            elif winner == -1:
                win_count[1] += 1
            games_played = win_count[0] + win_count[1]
            pbar.set_description("p1: {2} ({0:.1f}%) / p2: {3} ({1:.1f}%) ".format(
                (win_count[0] / max(games_played, 1)) * 100,
                (win_count[1] / max(games_played, 1)) * 100,
                win_count[0],
                win_count[1]
            ))
            pbar.update(1)
        pbar.close()

    def _play_game_process(self, result_queue, player1, player2, rollout_count):
        game_trail = DataCollectWithInvalidRolloutCount(pickle.loads(pickle.dumps(self.game_simulator)), player1, player2)
        for winner, invalid_rollouts in game_trail.do_rollouts(rollout_count):
            result_queue.put([winner, invalid_rollouts])

    def _main_process(self, result_queue, game_count:int):
        pbar = tqdm(total=game_count)
        win_count = [0, 0]
        total_invalid_rollouts = 0
        for curr_count in range(game_count):
            winner, invalid_rollouts = result_queue.get()
            total_invalid_rollouts += invalid_rollouts
            if winner == 1:
                win_count[0] += 1
            elif winner == -1:
                win_count[1] += 1
            pbar.set_description("p1: {2} ({0:.1f}%) / p2: {3} ({1:.1f}%) - invalidRollouts={4}".format(
                (win_count[0] / max(curr_count+1, 1)) * 100,
                (win_count[1] / max(curr_count+1, 1)) * 100,
                win_count[0],
                win_count[1],
                total_invalid_rollouts
            ))
            pbar.update(1)
        pbar.close()

    def play_games_async(self, player1=None, player2=None,
                         game_count:int=100, process_count:int=-1):
        if process_count <= 1:
            process_count = mp.cpu_count()
        workers      = process_count - 1
        manager      = mp.Manager()
        result_queue = manager.Queue()
        workers_load = (game_count / workers).__round__()
        total_load   = workers_load * workers
        processes    = [mp.Process(target=self._play_game_process,
                                   args=(result_queue, player1, player2, workers_load))
                        for _ in range(workers)]
        main_process = mp.Process(target=self._main_process, args=(result_queue, total_load))
        main_process.start()
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
        main_process.join()
        