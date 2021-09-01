import torch
import torch.nn as nn
import torch.nn.functional as F

#import numpy as np

class Advantage_Actor_Critic(nn.Module):
    def __init__(self, state_typesize, action_typesize, seed=0.):
        super(Advantage_Actor_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_typesize, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_typesize)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x), self.fc5(x)
#
# class Critic(nn.Module):
#     def __init__(self, state_typesize, action_typesize, seed=0.):
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_typesize, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256 + action_typesize, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, 1)
#
#     def forward(self, state, action):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = torch.cat((x, action), dim=1)
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         return self.fc5(x)
# class A2C_nn(nn.Module):
#     '''
#     Advantage actor-critic neural net
#     '''
#
#     def __init__(self, input_shape, n_actions):
#         super(A2C_nn, self).__init__()
#
#         self.lp = nn.Sequential(
#             nn.Linear(input_shape[0], 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU())
#         self.policy = nn.Linear(32, n_actions)
#         self.value = nn.Linear(32, 1)
#
#     def forward(self, x):
#         l = self.lp(x.float())
#         # return the actor and the critic
#         return self.policy(l), self.value(l)