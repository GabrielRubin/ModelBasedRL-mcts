from typing import List
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml.ml_model_base import CustomModule
from ml.early_stopping import EarlyStopping
import torch.nn.functional as F

gpu = torch.device("cuda")
cpu = torch.device("cpu")

class _RandomNetworkDistillation(CustomModule):
    def __init__(self, state_length, mean_correct_novelty=0):
        self.state_length = state_length
        self.mean_correct_novelty = mean_correct_novelty
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
 
    def get_novelty(self, state_diff:List[int], device='cpu', reduction='sum', is_train=False):
        self.eval()
        self._loss_f.reduction = reduction
        if device == 'cpu':
            x = torch.tensor(state_diff, device=cpu, dtype=torch.float)
        else:
            x = torch.tensor(state_diff, device=gpu, dtype=torch.float)
        prediction, target = self.forward(x)
        mse = self._loss_f(prediction, target)
        if is_train:
            if self.mean_correct_novelty == 0:
                self.mean_correct_novelty = mse.item()/len(state_diff)
            else:
                self.mean_correct_novelty = (self.mean_correct_novelty+mse.item()/len(state_diff))*0.5
            return mse
        if reduction == 'mean' or reduction == 'sum':
            return mse.item()
        else:
            return mse.tolist()

    def is_transition_valid(self, state_diffs):
        novelty = self.get_novelty(state_diffs, device='gpu')
        distance_from_mean = (novelty - self.mean_correct_novelty) / self.mean_correct_novelty
        return distance_from_mean <= 0

    def are_transitions_valid(self, state_diff:List[int]):
        novelties = self.get_novelty(state_diff, reduction='none', device='gpu')
        distances_from_mean = [(np.mean(novelty) - self.mean_correct_novelty) / self.mean_correct_novelty for novelty in novelties]
        return [distance_from_mean <= 0 for distance_from_mean in distances_from_mean]

    def _train_epoch(self, epoch:int, data_loader:DataLoader, *args, **kargs):
        total_loss = 0
        for _, data in enumerate(data_loader):
            state_diff = data.to(device=gpu, dtype=torch.float)
            loss = self.get_novelty(state_diff, device='gpu', is_train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss/max(len(data_loader.dataset), 1)

    def _test(self, data_loader:DataLoader, *args, **kargs):
        total = len(data_loader.dataset)
        pbar  = tqdm(total=total)
        correct = 0
        wrong   = 0
        true_neg = 0
        false_pos = 0
        for i, data in enumerate(data_loader):
            state_data    = data[:, :-1].to(device=gpu, dtype=torch.float)
            category      = data[:, -1].to(device=gpu, dtype=torch.float)
            pred_category = self.get_novelty(state_data, device='gpu', reduction='none')
            for i in range(len(category)):
                distance_from_mean = pred_category[i][0] - self.mean_correct_novelty
                curCategory = category[i]
                curNovelty  = pred_category[i][0]

                if curCategory == 1 and distance_from_mean <= 0:
                    correct += 1
                elif curCategory == 0 and distance_from_mean > 0:
                    correct += 1
                    true_neg += 1
                else:
                    wrong += 1
                    if curCategory == 0:
                        false_pos += 1

                    #print("Category = {0} / Novelty = {1} / Distance = {2}".format(curCategory, curNovelty, distance_from_mean))
            pbar.update(data_loader.batch_size)
            pbar.set_description(desc="ACC={0:.1f} - CORRECT={1}/{2} - True Negative={3} - False Positive={4}".format(
                (correct*100.0)/(correct+wrong), correct, correct+wrong, true_neg, false_pos
            ))
        pbar.close()

    def _get_custom_checkpoint(self):
        return {
            'state_length' : self.state_length,
            'mean_correct_novelty' : self.mean_correct_novelty
        }

    @classmethod
    def _custom_from_file(cls, checkpoint):
        state_length  = checkpoint['state_length']
        state_length  = state_length
        mean_correct_novelty = checkpoint['mean_correct_novelty']
        return cls(state_length, mean_correct_novelty)

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
            nn.ReLU(),
            nn.Linear(self.state_length * 12, self.state_length * 12),
            nn.ReLU(),
            nn.Linear(self.state_length * 12, self.state_length * 12),
            nn.ReLU(),
            nn.Linear(self.state_length * 12, (self.state_length * 0.25).__round__())
        )

class CheckersRND(_RandomNetworkDistillation):
    def _get_network_structure(self):
        return nn.Sequential(
            nn.Linear(self.state_length, self.state_length * 12),
            nn.ReLU(),
            nn.Linear(self.state_length * 12, self.state_length * 12),
            nn.ReLU(),
            nn.Linear(self.state_length * 12, self.state_length * 12),
            nn.ReLU(),
            nn.Linear(self.state_length * 12, (self.state_length * 0.25).__round__())
        )