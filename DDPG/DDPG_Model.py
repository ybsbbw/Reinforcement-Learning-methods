import torch
import torch.nn as nn
import torch.nn.functional as F

#import numpy as np

class Actor(nn.Module):
    def __init__(self, state_typesize, action_typesize, seed=0.):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_typesize, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_typesize)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))*2

class Critic(nn.Module):
    def __init__(self, state_typesize, action_typesize, seed=0.):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_typesize, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256 + action_typesize, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
