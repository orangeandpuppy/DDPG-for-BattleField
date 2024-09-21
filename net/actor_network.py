import torch
import torch.nn as nn
from config import MAX_FIGHTER
from utils import weight_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, alpha, n_state, n_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state//MAX_FIGHTER, 100)
        self.ln1 = nn.LayerNorm(100)
        self.fc2 = nn.Linear(100, 60)
        self.ln2 = nn.LayerNorm(60)
        self.action = nn.Linear(60, n_action//MAX_FIGHTER)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        # state: [batch_size, n_state/MAX_FIGHTER] 每个智能体观测到的状态
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        # action: [batch_size, n_action/MAX_FIGHTER] 每个智能体的动作
        action = torch.tanh(self.action(x))

        move = action[..., 0:5]
        move = torch.softmax(move, dim=-1)
        attack = action[..., 5:10]
        attack = torch.softmax(attack, dim=-1)
        fuels = action[..., 10]
        fuels = torch.sigmoid(fuels)
        fuels = fuels.unsqueeze(-1)
        missiles = action[..., 11]
        missiles = torch.sigmoid(missiles)
        missiles = missiles.unsqueeze(-1)

        action = torch.cat([move, attack, fuels, missiles], dim=-1)
        # action: [batch_size, 12]
        return action

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))