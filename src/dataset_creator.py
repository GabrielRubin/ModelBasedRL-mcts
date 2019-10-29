import torch.multiprocessing as mp
#import ray
from tqdm import tqdm
from ml.data_manager import save_data_csv
from boardGames.game_trial import DataCollectTrial

MAX_ROLLOUTS_PER_RUN = 10000

class DatasetCreator:
    def __init__(self, data_creator:DataCollectTrial):
        self.data_creator = data_creator

    def create_dataset(self, rollouts:int, path:str, override:bool=False):
        print("--CREATING DATASET--")
        if override:
            save_data_csv(content=[], file_name=path, write_mode='w')
        pbar = tqdm(total=rollouts)
        for data, _ in self.data_creator.do_rollouts(rollouts, True):
            save_data_csv(content=data, file_name=path, write_mode='a')
            pbar.update(1)

class DatasetCreatorAsync(DatasetCreator):
    def __init__(self, data_creator:DataCollectTrial, process_count:int=-1):
        super().__init__(data_creator)
        self.process_count = process_count

    @staticmethod
    def _file_writer(result_queue, rollouts:int, path:str):
        total_data = []
        for _ in tqdm(range(rollouts)):
            total_data += result_queue.get()
        print("--WRITING FILE--")
        save_data_csv(content=total_data, file_name=path, write_mode='a')

    @staticmethod
    def _worker_process(result_queue, data_creator:DataCollectTrial, rollouts:int):
        for data, _ in data_creator.do_rollouts(rollouts, True):
            result_queue.put(data)

    def create_dataset(self, rollouts:int, path:str, override:bool=False):
        print("--CREATING DATASET ASYNC--")
        remaining_rollouts = rollouts
        proc_count = self.process_count
        if proc_count <= 1:
            proc_count = mp.cpu_count()
        workers      = proc_count - 1
        manager      = mp.Manager()
        result_queue = manager.Queue()
        if override:
            save_data_csv(content=[], file_name=path, write_mode='w')

        while(remaining_rollouts > 0):
            print("REMAINING_ROLLOUTS = {0}".format(remaining_rollouts))
            rollouts = min(remaining_rollouts, MAX_ROLLOUTS_PER_RUN)
            proc_count   = min(2, rollouts)
            workers      = proc_count - 1
            manager      = mp.Manager()
            result_queue = manager.Queue()
            workers_load = max(1, (rollouts / workers).__round__())
            total_load   = workers_load * workers
            processes    = [
                mp.Process(target=self._worker_process,
                        args=(result_queue, self.data_creator, workers_load))
                for _ in range(workers)
            ]
            writer_proc  = mp.Process(target=self._file_writer, args=(result_queue, total_load, path))
            writer_proc.start()
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join()
            writer_proc.join()
            remaining_rollouts -= MAX_ROLLOUTS_PER_RUN

'''
NOT WORKING
class DatasetCreatorAsyncRay(DatasetCreator):
    def __init__(self, dataCreator:DataCreator, processCount:int=-1):
        super().__init__(dataCreator)
        self.processCount = processCount

    def CreateDataset(self, rollouts:int, path:str, override:bool=False):
        print("--CREATING DATASET ASYNC RAY--")
        procCount = self.processCount
        if procCount <= 1:
            procCount = mp.cpu_count()
        ray.init(num_cpus=procCount)
        result_ids = [self._WorkerProcess.remote(self.dataCreator) for _ in range(rollouts)]
        pbar = tqdm(total=rollouts)
        while(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            self._FileWriter(ray.get(done_id[0]), path)
            pbar.update(1)
        pbar.close()
        ray.shutdown()

    @staticmethod
    def _FileWriter(data, path:str):
        SaveDataCSV(content=data, fileName=path, writeMode='a')

    @ray.remote
    def _WorkerProcess(self, dataCreator:DataCreator):
        return dataCreator.CreateData()
'''