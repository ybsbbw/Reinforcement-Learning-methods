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
import torch
import random
import numpy as np
from math import pow as Mpow
from collections import namedtuple, deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Memory():
    def __init__(self, horizon_step, gamma, lamda, device):
        """
        self.state_list = deque(maxlen=horizon_step)
        self.action_list = deque(maxlen=horizon_step)
        self.reward_list = deque(maxlen=horizon_step)
        self.done_list = deque(maxlen=horizon_step)
        self.next_state_list = deque(maxlen=horizon_step)
        self.value_list = deque(maxlen=horizon_step)       此处value_list应当比其它list多一位，因为计算delta要用到Vi+1，避免报错
        """
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.done_list = []
        self.next_state_list = []
        self.log_prob_list = []
        self.value_list = []

        self.horizon_step = horizon_step
        self.gamma = gamma
        self.lamda = lamda

        self.k = 0

        self.device = device

    def reset(self):
        """
        self.state_list = deque(maxlen=horizon_step)
        self.action_list = deque(maxlen=horizon_step)
        self.reward_list = deque(maxlen=horizon_step)
        self.done_list = deque(maxlen=horizon_step)
        self.next_state_list = deque(maxlen=horizon_step)
        self.value_list = deque(maxlen=horizon_step)
        """
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.done_list = []
        self.next_state_list = []
        self.log_prob_list = []
        self.value_list = []

        self.k = 0

    def add(self, state, action, reward, done, next_state, log_prob, value):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.next_state_list.append(next_state)
        self.log_prob_list.append(log_prob)
        self.value_list.append(value)

    def compute(self, delta_type=1, infinite=1, norm=1):
        self.Q_list = []
        self.delta_list = []
        self.GAE_list = []
        Q = self.value_list[-1]
        GAE = 0

        for i in reversed(range(len(self.state_list))):
            Q = self.reward_list[i] + (1 - self.done_list[i]) * self.gamma * Q
            self.Q_list.append(Q)

            """ 
            delta设计两种处理思路：第一类，delta_type == 1, delta_n = r_n + gamma*V_n+1 - V_n
                                   第二类，delta_type != 1, delta_n = Q_n - V_n
            """
            if delta_type == 1:
                delta = self.reward_list[i] + (1 - self.done_list[i]) * self.gamma * self.value_list[i+1] - self.value_list[i]
            else:
                delta = Q - self.value_list[i]
            self.delta_list.append(delta)

            """
            GAE系数设计两种处理思路：第一类，infinite == 1, 按照轨迹序列无限长近似结果
                                             GAE_t = delta_n + gamma*lamda*GAE_r+1

                                     第二类，infinite != 1, 按照轨迹序列有限长推导结果，k为当前step离trajectory最后一步step的距离
                                             GAE_t = delta_n + gamma*lamda*GAE_r+1*[(1-lamda^k)/(1-lamda^(k+1))]
            """
            if infinite == 1:
                GAE = delta + self.gamma * self.lamda * GAE
            else:
                GAE = delta + self.gamma * self.lamda * ((1-Mpow(self.lamda, self.k))/(1-Mpow(self.lamda, self.k+1))) * GAE

            self.GAE_list.append(GAE)

            self.k += 1

        if norm == 1:
            self.GAE_list = (self.GAE_list - np.mean(self.GAE_list)) / (np.std(self.GAE_list) + 1e-8)

        """to device"""
        self.state_list = torch.from_numpy(np.vstack(self.state_list)).float().to(self.device)
        self.action_list = torch.from_numpy(np.vstack(self.action_list)).float().to(self.device)
        self.reward_list = torch.from_numpy(np.vstack(self.reward_list)).float().to(self.device)
        self.done_list = torch.from_numpy(np.vstack(self.done_list)).float().to(self.device)
        self.next_state_list = torch.from_numpy(np.vstack(self.next_state_list)).float().to(self.device)
        self.log_prob_list = torch.from_numpy(np.vstack(self.log_prob_list)).float().to(self.device)
        self.value_list = torch.from_numpy(np.vstack(self.value_list)).float().to(self.device)

        self.Q_list = torch.from_numpy(np.vstack(list(reversed(self.Q_list)))).float().to(self.device)
        self.delta_list = torch.from_numpy(np.vstack(list(reversed(self.delta_list)))).float().to(self.device)
        self.GAE_list = torch.from_numpy(np.vstack(list(reversed(self.GAE_list)))).float().to(self.device)
        """"""

        return self.Q_list, self.delta_list, self.GAE_list

    def sample(self, batch_size):
        index_list = []
        for index in BatchSampler(SubsetRandomSampler(range(len(self.state_list))), batch_size, False):
            index_list.append(index)
        return index_list

class Buffer():
    def __init__(self, buffer_size, batch_size,  device, seed=0):
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

        self.store = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        store_add = self.experience(state, action, reward, next_state, done)
        self.store.append(store_add)

    def sample(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        experience = random.sample(self.store, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)
        # action_log_pros = torch.from_numpy(np.vstack([e.action_log_pro for e in experience if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.store)