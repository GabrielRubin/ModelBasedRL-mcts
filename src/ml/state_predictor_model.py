from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .ml_model_base import CustomModule
from torch.utils.data import DataLoader

gpu = torch.device("cuda")
cpu = torch.device("cpu")

class _StatePredictor(CustomModule):
    def __init__(self, state_length, action_length):
        self.state_length  = state_length
        self.action_length = action_length
        super().__init__()
        self._loss_f = nn.MSELoss(reduction='sum')

    @property
    def learning_rate(self):
        return 0.001

    def forward(self, state_and_action):
        return state_and_action

    def loss_function(self, pred_state, next_state):
        return self._loss_f(pred_state, next_state)
 
    def get_next_state(self, state:List[int], action:List[int]):
        self.eval()
        x = torch.tensor(state + action, device=cpu, dtype=torch.float)
        result = self.forward(x).tolist()
        return [float(j).__round__() for j in result]
    
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
                isEquals = True
                for j in range(len(actual)):
                    if actual[j] != result[j]:
                        isEquals = False
                        break
                if isEquals:
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
        state_action_length = self.state_length + self.action_length
        self.sequential = nn.Sequential(
            nn.Linear(state_action_length, state_action_length * 6),
            nn.LeakyReLU(),
            nn.Linear(state_action_length * 6, state_action_length * 6),
            nn.LeakyReLU(),
            nn.Linear(state_action_length * 6, state_action_length * 6),
            nn.LeakyReLU(),
            nn.Linear(state_action_length * 6, self.state_length)
        )
        self.sequential.to(gpu)

    def forward(self, state_and_action):
        return self.sequential(state_and_action)