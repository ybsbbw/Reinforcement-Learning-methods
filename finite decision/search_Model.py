import torch
import torch.nn as nn
import torch.nn.functional as F

#import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Actor(nn.Module):
    def __init__(self, state_typesize, action_typesize, seed=0):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_typesize, affine=False)
        self.fc1 = nn.Linear(state_typesize, 256)
        # self.bn1 = nn.BatchNorm1d(state_typesize, affine=False)
        self.fc2 = nn.Linear(256, 256)
        # self.bn1 = nn.BatchNorm1d(state_typesize, affine=False)
        self.fc3 = nn.Linear(256, 256)
        # self.bn1 = nn.BatchNorm1d(state_typesize, affine=False)
        self.fc4 = nn.Linear(256, action_typesize)
        # self.input = torch.randn(256, state_typesize).to(device)

    def forward(self, state):
        # aaa = self.test(self.input)
        # state = self.bn1(state)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class Critic(nn.Module):
    def __init__(self, state_typesize, action_typesize, seed=0):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # self.test = nn.BatchNorm1d(state_typesize, 512)
        self.bn1 = nn.BatchNorm1d(state_typesize, affine=False)
        self.fc1 = nn.Linear(state_typesize, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256 + action_typesize, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, action):
        # x = self.bn1(state)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.cat((x, action), dim=1)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
