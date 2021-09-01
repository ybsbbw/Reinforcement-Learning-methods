#!/usr/bin/python
# -- coding:utf-8 --
'''
        ====================================================================================================
         .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
        | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
        | |     ______   | || |      __      | || |    _______   | || |     _____    | || |      __      | |
        | |   .' ___  |  | || |     /  \     | || |   /  ___  |  | || |    |_   _|   | || |     /  \     | |
        | |  / .'   \_|  | || |    / /\ \    | || |  |  (__ \_|  | || |      | |     | || |    / /\ \    | |
        | |  | |         | || |   / ____ \   | || |   '.___`-.   | || |      | |     | || |   / ____ \   | |
        | |  \ `.___.'\  | || | _/ /    \ \_ | || |  |`\____) |  | || |     _| |_    | || | _/ /    \ \_ | |
        | |   `._____.'  | || ||____|  |____|| || |  |_______.'  | || |    |_____|   | || ||____|  |____|| |
        | |              | || |              | || |              | || |              | || |              | |
        | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
         '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 


                 .----------------.    .----------------.    .----------------.    .----------------.               
                | .--------------. |  | .--------------. |  | .--------------. |  | .--------------. |              
                | |     _____    | |  | |    _______   | |  | |    _______   | |  | |     ______   | |              
                | |    |_   _|   | |  | |   /  ___  |  | |  | |   /  ___  |  | |  | |   .' ___  |  | |              
                | |      | |     | |  | |  |  (__ \_|  | |  | |  |  (__ \_|  | |  | |  / .'   \_|  | |              
                | |      | |     | |  | |   '.___`-.   | |  | |   '.___`-.   | |  | |  | |         | |              
                | |     _| |_    | |  | |  |`\____) |  | |  | |  |`\____) |  | |  | |  \ `.___.'\  | |              
                | |    |_____|   | |  | |  |_______.'  | |  | |  |_______.'  | |  | |   `._____.'  | |              
                | |              | |  | |              | |  | |              | |  | |              | |              
                | '--------------' |  | '--------------' |  | '--------------' |  | '--------------' |              
                 '----------------'    '----------------'    '----------------'    '----------------'      
        ====================================================================================================  
          Current Task :  | User : Wensheng Zhang
        ====================================================================================================  
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from main_TZ import EachEpoch
import matplotlib.pyplot as plt

class Label():
    def __init__(self, state_dim, action_dim, lr, gamma, seed, device):
        super(Label, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr = lr
        self.device =device
        self.gamma = gamma
        self.action_max = np.array([415, 360, 360])
        self.action_min = np.array([0, 0, 0])

        self.actor_eval = Actor(state_dim, action_dim, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.lr)

        self.noise = OUNoise(action_dim, seed)

    def train(self, state, action, score, times):
        input_state = torch.from_numpy(state).detach().float().unsqueeze(0).to(self.device)
        input_action = torch.from_numpy(action).detach().float().unsqueeze(0).to(self.device)

        self.ee = EachEpoch(False)

        x = []
        score_list = []

        for i in range(times):
            prediction = self.actor_eval(input_state)

            loss = F.mse_loss(prediction, input_action)  # eval->target
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            output_action = prediction.cpu().data.numpy().squeeze()  # + self.noise.sample()

            ratio_list = np.split((np.tanh(output_action) + 1) / 2, int(self.action_dim / 3))
            send_action = self.action_min + (self.action_max - self.action_min) * ratio_list

            current_score = self.ee.run(send_action)

            if current_score > score:
                input_action = prediction.detach()
                score = current_score
                if current_score >= 2000:
                    self.actor_eval.save_network(episode, score, time)

            print('\rtrain_episode:{} Score:{} \naction:{}\n'.format(i, current_score, send_action), end=" ")
            if current_score >= -1000:
                print("=========>  SUCCESS")
            else:
                print(" ")

            self.ee.reset()

            x.append(i + 1)
            score_list.append(current_score)

            plt.ion()
            plt.clf()

            plt.plot(x, score_list, label='score', color="r")
            plt.legend(loc='best')

            plt.show()
            plt.pause(0.1)
            plt.ioff()

class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0, runtype=1):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(256, 256)
        self.action = nn.Linear(256, action_size)
        self.std = nn.Linear(256, action_size)

        """initialization"""
        layer_init(self.fc1)
        # layer_init(self.fc2)
        # layer_init(self.fc3)
        layer_init(self.fc4)
        layer_init(self.action)
        layer_init(self.std)

        self.runtype = runtype

        self.a_std = torch.zeros(1, 1) + 0.5
        self.min_val = 1e-7

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        if self.runtype == 1:
            action = self.action(x)
            # a_std = torch.sin(self.std(x)) + 1 + self.min_val
            a_std = torch.tanh(self.std(x)) + 1 + self.min_val
            # a_std = F.softplus(self.std(x)) + self.min_val
            # a_std = self.a_std
            return action
        else:
            action_prob = F.softmax(self.action(x), dim=-1) + 1e-7
            return action_prob

    def save_network(self, episode, score, time):
        if os.path.exists("./regression_networks/" + time) is False:
            os.mkdir("./regression_networks/" + time)
        print("Saving episode{} regression_networks...".format(episode))
        torch.save(self.state_dict(), "./regression_networks" + time + "/" + "episode_{}, score{}".format(episode, score) + ".pth")

    def load_network(self, path):
        if os.path.exists("./regression_networks/load") is False:
            print("The folder does not exist!")
        elif os.listdir("./regression_networks/load") is None:
            print("Could not find the param file in folder!")
        else:
            print("Loading regression_networks...")
            get_dir = os.listdir("./regression_networks/load")
            self.load_state_dict(torch.load("./regression_networks/load/" + get_dir[0]))


def layer_init(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

