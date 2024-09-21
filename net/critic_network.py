import torch
import torch.nn as nn
from utils import weight_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CriticNetwork(nn.Module):
    def __init__(self, beta, n_state, n_action):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 100)
        self.ln1 = nn.LayerNorm(100)
        self.fc2 = nn.Linear(100, 60)
        self.ln2 = nn.LayerNorm(60)
        self.fc3 = nn.Linear(n_action, 60)
        self.q = nn.Linear(60, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta, weight_decay=0.001)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):
        x_s = state.view(state.shape[0], -1)
        x_s = torch.relu(self.ln1(self.fc1(x_s)))
        x_s = torch.relu(self.ln2(self.fc2(x_s)))

        x_a = action.view(action.shape[0], -1)
        x_a = self.fc3(x_a)

        q = torch.relu(x_s + x_a)
        q = self.q(q)
        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))