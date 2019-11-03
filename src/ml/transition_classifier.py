from typing import List
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml.ml_model_base import CustomModule
from ml.early_stopping import EarlyStopping

gpu = torch.device("cuda")
cpu = torch.device("cpu")

class _TransitionClassifier(CustomModule):
    def __init__(self, state_length):
        self.state_length = state_length * 2
        super().__init__()
        self._loss_f = nn.MSELoss(reduction='sum')

    def _create_module_parameters(self):
        self.pred_net   = self._get_network_structure()
        self.pred_net.to(gpu)

    def forward(self, state_diff):
        return self.pred_net(state_diff)

    @property
    def learning_rate(self):
        return 0.001

    def loss_function(self, pred_state, tgt_state):
        return self._loss_f(pred_state, tgt_state)

    def _create_early_stopping(self):
        return EarlyStopping('min', patience=15, threshold=0.001)
 
    def get_novelty(self, state_diff:List[int], device='cpu'):
        self.eval()
        if device == 'cpu':
            x = torch.tensor(state_diff, device=cpu, dtype=torch.float)
        else:
            x = torch.tensor(state_diff, device=gpu, dtype=torch.float)
        prediction = self.forward(x)
        return sum(mse)

    def _train_epoch(self, epoch:int, data_loader:DataLoader, *args, **kargs):
        total_loss        = 0
        for _, data in enumerate(data_loader):
            state_diff = data[:, :-1].to(device=gpu, dtype=torch.float)
            state_cat  = data[:, -1].to(device=gpu, dtype=torch.float)
            prediction = self.forward(state_diff)
            prediction  = prediction.view(-1)
            loss = self.loss_function(prediction, state_cat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss/max(len(data_loader.dataset), 1)

    def _test(self, data_loader:DataLoader, *args, **kargs):
        total   = len(data_loader.dataset)
        pbar    = tqdm(total=total)
        correct = 0
        wrong   = 0
        for _, data in enumerate(data_loader):
            state_data    = data[:, :-1].to(device=gpu, dtype=torch.float)
            category      = data[:, -1].to(device=gpu, dtype=torch.float)
            pred_category = self.forward(state_data).tolist()
            for i in range(len(category)):
                if category[i] == pred_category[i][0].__round__():
                    correct += 1
                else:
                    wrong += 1
            pbar.update(data_loader.batch_size)
            pbar.set_description(desc="ACC={0:.1f} - CORRECT={1}/{2}".format(
                (correct*100.0)/(correct+wrong), correct, correct+wrong
            ))
        pbar.close()

    def _get_custom_checkpoint(self):
        return {
            'state_length' : self.state_length,
        }

    @classmethod
    def _custom_from_file(cls, checkpoint):
        state_length  = checkpoint['state_length']
        state_length  = int((state_length * 0.5).__round__())
        return cls(state_length)

class HexTransitionClassifier(_TransitionClassifier):
    def _get_network_structure(self):
        return nn.Sequential(
            nn.Linear(self.state_length, self.state_length * 6),
            nn.LeakyReLU(),
            nn.Linear(self.state_length * 6, self.state_length),
            nn.LeakyReLU(),
            nn.Linear(self.state_length, 1)
        )