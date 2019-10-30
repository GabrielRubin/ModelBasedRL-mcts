import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from abc import ABC, abstractmethod
from ml.early_stopping import EarlyStopping

class CustomModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self._create_module_parameters()
        self.optimizer      = self._create_optimizer()
        self.scheduler      = self._create_scheduler()
        self.early_stopping = self._create_early_stopping()
        self.epoch_count    = 0
        self.loss           = 0

    def train_model(self, dataset:Dataset, batch_size:int, epoch_count:int, 
                    *args, shuffle:bool=True, loss_history_len:int=10, **kargs):
        print("-TRAINING-")
        self.train()
        dataset_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
        pbar = tqdm(total=epoch_count)
        last_loss    = 0
        loss_history = []
        for epoch in range(epoch_count):
            self.loss = self._train_epoch(epoch, dataset_loader, args, kargs)
            if self.scheduler:
                self.scheduler.step(self.loss)
            if self.early_stopping:
                self.early_stopping.step(self.loss)
                if self.early_stopping.stop:
                    print("End of Training because of \
                           early stopping at epoch {}".format(epoch))
                    pbar.close()
                    break
            self.epoch_count += 1
            pbar.update(1)
            loss_history_value = 0
            if last_loss != 0:
                loss_history.append(self.loss - last_loss)
                loss_history_value = np.mean(loss_history)
            last_loss = self.loss
            if(len(loss_history) > loss_history_len):
                loss_history.pop(0)
            pbar.set_description("loss= {0:.4f} - delta= {1:.4f} ".format(
                self.loss, loss_history_value
            ))
        pbar.close()

    def test_module(self, dataset:Dataset, *args, batch_size:int=1000, **kargs):
        print("-TESTING-")
        self.eval()
        dataset_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
        self._test(dataset_loader, args, kargs)

    def _create_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def _create_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)

    def _create_early_stopping(self):
        return EarlyStopping('min', patience=5)

    def save(self, file_name:str):
        if self.scheduler is not None:
            scheduler_dict = self.scheduler.state_dict()
        else:
            scheduler_dict = None
        if self.early_stopping is not None:
            early_stopping_dict = self.early_stopping.state_dict()
        else:
            early_stopping_dict = None
        checkpoint = {'dict':           self.state_dict(),
                      'optimizer':      self.optimizer.state_dict(),
                      'scheduler':      scheduler_dict,
                      'early_stopping': early_stopping_dict,
                      'epoch_count':    self.epoch_count,
                      'loss':           self.loss }
        checkpoint.update(self._get_custom_checkpoint())
        torch.save(checkpoint, '{0}.pth'.format(file_name))

    @classmethod
    def from_file(cls, file_name:str):
        if not file_name.__contains__('.pth'):
            file_name = "{0}.pth".format(file_name)
        checkpoint = torch.load(file_name)
        instance   = cls._custom_from_file(checkpoint)
        instance.load_state_dict(checkpoint['dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer'])
        if instance.scheduler is not None:
            instance.scheduler.load_state_dict(checkpoint['scheduler'])
        if instance.early_stopping is not None:
            instance.early_stopping.load_state_dict(checkpoint['early_stopping'])
        instance.epoch_count = checkpoint['epoch_count']
        instance.loss        = checkpoint['loss']
        return instance
    
    @abstractmethod
    def _create_module_parameters(self):
        '''
        Create the module's parameters here
        '''

    @abstractmethod
    def _train_epoch(self, epoch:int, data_loader:DataLoader, *args, **kargs):
        '''
        Code that will run at each train epoch
        (epoch, DataLoader, *, **) -> loss (float)
        '''
    @abstractmethod
    def _test(self, data_loader:DataLoader, *args, **kargs):
        '''
        Default test call
        (using tqdm for progress bar is a must here)
        '''
    @abstractmethod
    def loss_function(self, *args, **kargs):
        '''
        Default loss call
        '''
    @property
    @abstractmethod
    def learning_rate(self):
        '''
        Return the module's learning rate (for the optimizer)
        '''
    @abstractmethod
    def _get_custom_checkpoint(self):
        '''
        Returns a dict with the custom checkpoint params (for loading)
        '''
    @classmethod
    @abstractmethod
    def _custom_from_file(cls, checkpoint):
        '''
        Returns an instance from a file
        '''