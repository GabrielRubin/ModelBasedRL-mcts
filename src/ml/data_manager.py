import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset

class StateTransitionDataset(Dataset):
    @classmethod
    def from_list(cls, data_list, board_size:int):
        dataset = cls(None, board_size)
        dataset.data = pd.DataFrame(data_list)
        return dataset

    def __init__(self, csv_path:str, board_size:int):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
        self.board_size = board_size

    def __getitem__(self, index):
        return self.data.iloc[index].values #np.asarray(self.data.iloc[index])

    def __len__(self):
        return len(self.data.index)

class StateTransitionDatasetForNovelty(StateTransitionDataset):
    @classmethod
    def from_list(cls, data_list, board_size:int):
        dataset = cls(None, board_size)
        dataset.data = pd.DataFrame(data_list)
        return dataset

    def __getitem__(self, index):
        return np.append(self.data.iloc[index, :self.board_size*2].values \
                         - self.data.iloc[index, self.board_size*4:].values,
                         self.data.iloc[index, self.board_size*2:self.board_size*4])
        #return np.append(self.data.iloc[index, :self.board_size*2].values,
        # self.data.iloc[index, self.board_size*4:].values)

class StateTransitionDatasetTEST(StateTransitionDataset):
    @classmethod
    def from_list(cls, data_list, board_size:int):
        dataset = cls(None, board_size)
        dataset.data = pd.DataFrame(data_list)
        return dataset

    def __getitem__(self, index):
        part_1 =  np.append(self.data.iloc[index, :self.board_size*2].values \
                            - self.data.iloc[index, self.board_size*4:-1].values,
                            self.data.iloc[index, self.board_size*2:self.board_size*4])
        return np.append(part_1, self.data.iloc[index, -1])
        #return np.append(self.data.iloc[index, :self.board_size*2].values,
        # self.data.iloc[index, self.board_size*4:].values)
    
class StateTransitionDatasetTEST2(StateTransitionDataset):
    @classmethod
    def from_list(cls, data_list, board_size:int):
        dataset = cls(None, board_size)
        dataset.data = pd.DataFrame(data_list)
        return dataset

    def __getitem__(self, index):
        part_1 =  np.append(self.data.iloc[index, :self.board_size*2].values,
                            self.data.iloc[index, self.board_size*4:-1].values)
        return np.append(part_1, self.data.iloc[index, -1])
        #return np.append(self.data.iloc[index, :self.board_size*2].values,
        # self.data.iloc[index, self.board_size*4:].values)

class StateTransitionDatasetTEST2_slim(StateTransitionDataset):
    @classmethod
    def from_list(cls, data_list, board_size:int):
        dataset = cls(None, board_size)
        dataset.data = pd.DataFrame(data_list)
        return dataset

    def __getitem__(self, index):
        return np.append(self.data.iloc[index, :self.board_size*2].values,
                            self.data.iloc[index, self.board_size*4:].values)
        #return np.append(self.data.iloc[index, :self.board_size*2].values,
        # self.data.iloc[index, self.board_size*4:].values)

def save_data_csv(content, file_name:str, write_mode='a'):
    data_frame = pd.DataFrame(content)
    data_frame.to_csv('{0}.csv'.format(file_name), mode=write_mode, header=False, index=False, chunksize=100000)

def get_data(file_name:str, dtype:str='int'):
    return pd.read_csv('{0}.csv'.format(file_name), dtype=dtype)