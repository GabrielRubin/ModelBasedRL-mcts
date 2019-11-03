from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml.ml_model_base import CustomModule
from ml.early_stopping import EarlyStopping
from ml.rnd_model import _RandomNetworkDistillation

gpu = torch.device("cuda")
cpu = torch.device("cpu")

class _StatePredictor(CustomModule):
    def __init__(self, state_length, action_length):
        self.state_length  = state_length
        self.action_length = action_length
        super().__init__()
        self._loss_f = nn.MSELoss(reduction='sum')

    @classmethod
    def compare_states(cls, state_a, state_b):
        if len(state_a) != len(state_b):
            return False
        for j in range(len(state_a)):
            if state_a[j] != state_b[j]:
                return False
        return True

    @property
    def learning_rate(self):
        return 0.001

    def forward(self, state_and_action):
        return state_and_action

    def loss_function(self, pred_state, next_state):
        return self._loss_f(pred_state, next_state)
 
    def get_next_state(self, state:List[int], action:List[int]):
        self.eval()
        x = torch.tensor(state + action, device=gpu, dtype=torch.float)
        result = self.forward(x).tolist()
        return [float(j).__round__() for j in result]

    def get_next_states(self, last_state, states, actions, rnd):
        self.eval()
        x = torch.tensor(np.append(states, actions, axis=1), device=gpu, dtype=torch.float)
        results = self.forward(x).tolist()
        results = [[float(j).__round__() for j in result] for result in results]
        transitions = [np.append(np.array(last_state) \
                        - np.array(results[i]),
                        np.array(actions[i]))
                       for i in range(len(actions))]
        valid_transitions = rnd.are_transitions_valid(transitions)
        return [results[i] for i in range(len(results)) if valid_transitions[i]]
    
    def _train_epoch(self, epoch:int, data_loader:DataLoader, *args, **kargs):
        state_data_length = self.state_length + self.action_length
        total_loss        = 0
        for _, data in enumerate(data_loader):
            state_action = data[:, :state_data_length].to(device=gpu, dtype=torch.float)
            next_state   = data[:, state_data_length:].to(device=gpu, dtype=torch.float)
            pred_state   = self.forward(state_action)
            loss         = self.loss_function(pred_state, next_state)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss/max(len(data_loader.dataset), 1)

    def _test(self, data_loader:DataLoader, *args, **kargs):
        state_data_length = self.state_length + self.action_length
        total   = len(data_loader.dataset)
        pbar    = tqdm(total=total)
        correct = 0
        wrong   = 0
        for _, data in enumerate(data_loader):
            state_action = data[:, :state_data_length].to(device=gpu, dtype=torch.float)
            next_state   = data[:, state_data_length:].to(device=gpu, dtype=torch.float)
            pred_state   = self.forward(state_action)
            for i in range(len(pred_state)):
                result = [float(j).__round__() for j in pred_state[i]]
                actual = [float(j).__round__() for j in next_state[i]]
                is_equal = self.compare_states(result, actual)
                if is_equal:
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
            'action_length': self.action_length
        }

    @classmethod
    def _custom_from_file(cls, checkpoint):
        state_length  = checkpoint['state_length']
        action_length = checkpoint['action_length']
        return cls(state_length, action_length)

class HexStatePredictor(_StatePredictor):
    def _create_module_parameters(self):
        state_action_length = self.state_length + self.action_length
        self.sequential = nn.Sequential(
            nn.Linear(state_action_length, state_action_length),
            nn.ReLU(),
            nn.Linear(state_action_length, self.state_length)
        )
        self.sequential.to(gpu)

    def forward(self, state_and_action):
        return self.sequential(state_and_action)

class OthelloStatePredictor(_StatePredictor):
    def _create_module_parameters(self):
        self.num_channels = 512 #512
        self.conv1 = nn.Conv2d(4, self.num_channels, 3, stride=1, padding=1).to(gpu)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1).to(gpu)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1).to(gpu)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1).to(gpu)

        self.bn1 = nn.BatchNorm2d(self.num_channels).to(gpu)
        self.bn2 = nn.BatchNorm2d(self.num_channels).to(gpu)
        self.bn3 = nn.BatchNorm2d(self.num_channels).to(gpu)
        self.bn4 = nn.BatchNorm2d(self.num_channels).to(gpu)

        self.fc1 = nn.Linear(self.num_channels*(6-4)*(6-4), 1024).to(gpu)
        self.fc_bn1 = nn.BatchNorm1d(1024).to(gpu)

        self.fc2 = nn.Linear(1024, 512).to(gpu)
        self.fc_bn2 = nn.BatchNorm1d(512).to(gpu)

        self.fc3 = nn.Linear(512, self.action_length).to(gpu)

    def _create_early_stopping(self):
        return EarlyStopping('min', patience=25)

    #TEMP
    @classmethod
    def compare_states(cls, state_a, state_b):
        if len(state_a) != len(state_b):
            return False
        for j in range(len(state_a)):
            if min(abs(state_a[j]), 1) != abs(state_b[j]):
                return False
        return True

    def forward(self, s, training=True):
        board_x = 6
        board_y = 6
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 4, board_x, board_y)                          # batch_size x 1 x board_x x board_y
        s = F.leaky_relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.leaky_relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.leaky_relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.leaky_relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, 512*(board_x-4)*(board_y-4))

        s = F.dropout(F.leaky_relu(self.fc_bn1(self.fc1(s))), p=0.3, training=training)  # batch_size x 1024
        s = F.dropout(F.leaky_relu(self.fc_bn2(self.fc2(s))), p=0.3, training=training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        #v = self.fc4(s)                                                                          # batch_size x 1

        return pi