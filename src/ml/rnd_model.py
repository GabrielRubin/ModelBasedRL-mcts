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

class _RandomNetworkDistillation(CustomModule):
    def __init__(self, state_length):
        self.state_length = state_length * 2
        super().__init__()
        self._loss_f = nn.MSELoss(reduction='sum')

    def _create_module_parameters(self):
        self.pred_net   = self._get_network_structure()
        self.target_net = self._get_network_structure()
        self.pred_net.to(gpu)
        self.target_net.to(gpu)
        self._reset_network()
        for param in self.target_net.parameters():
            param.requires_grad = False

    def _reset_network(self):
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                #torch.nn.init.kaiming_normal_(m, nonlinearity='relu') other option (?)
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()
        self.pred_net.apply(init_weights)
        self.target_net.apply(init_weights)

    def forward(self, state_diff):
        pred_output   = self.pred_net.forward(state_diff)
        target_output = self.target_net.forward(state_diff)
        return pred_output, target_output

    @property
    def learning_rate(self):
        return 0.001

    def loss_function(self, pred_state, tgt_state):
        return self._loss_f(pred_state, tgt_state)

    def _create_early_stopping(self):
        return EarlyStopping('min', patience=1, threshold=0.001)
 
    def get_novelty(self, state_diff:List[int], device='cpu'):
        self.eval()
        if device == 'cpu':
            x = torch.tensor(state_diff, device=cpu, dtype=torch.float)
        else:
            x = torch.tensor(state_diff, device=gpu, dtype=torch.float)
        prediction, target = self.forward(x)
        prediction = np.array(prediction.tolist())
        target     = np.array(target.tolist())
        mse        = (np.square(prediction - target)).mean(axis=1)
        return sum(mse)

    def _train_epoch(self, epoch:int, data_loader:DataLoader, *args, **kargs):
        total_loss        = 0
        for _, data in enumerate(data_loader):
            state_diff = data.to(device=gpu, dtype=torch.float)
            prediction, target = self.forward(state_diff)
            loss = self.loss_function(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss/max(len(data_loader.dataset), 1)

    def _test(self, data_loader:DataLoader, *args, **kargs):
        total = len(data_loader.dataset)
        pbar  = tqdm(total=total)
        total = 0
        _max  = 0
        for i, data in enumerate(data_loader):
            state_diff = data.to(device=gpu, dtype=torch.float)
            novelty = self.get_novelty(state_diff, device='gpu')
            total  += novelty
            _max    = max(novelty, _max)
            pbar.update(data_loader.batch_size)
            pbar.set_description(desc="MEAN={0:.6f} - MAX={1:.6f}".format(
                total/(i+1), _max
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

class HexRND(_RandomNetworkDistillation):
    def _get_network_structure(self):
        return nn.Sequential(
            nn.Linear(self.state_length, self.state_length * 6),
            nn.LeakyReLU(),
            nn.Linear(self.state_length * 6, self.state_length * 6),
            nn.LeakyReLU(),
            nn.Linear(self.state_length * 6, self.state_length)
        )

class OthelloRND(_RandomNetworkDistillation):
    def _get_network_structure(self):
        return nn.Sequential(
            nn.Linear(self.state_length, self.state_length * 12),
            nn.LeakyReLU(),
            nn.Linear(self.state_length * 12, self.state_length * 12),
            nn.LeakyReLU(),
            nn.Linear(self.state_length * 12, self.state_length * 12),
            nn.LeakyReLU(),
            nn.Linear(self.state_length * 12, self.state_length)
        )