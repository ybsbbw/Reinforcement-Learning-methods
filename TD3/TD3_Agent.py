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
import copy
import torch
import random
import numpy as np
from TD3_Buffer import Memory, Buffer
from TD3_Model import Actor, Value, Q_net
from copy import deepcopy
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Agent():
    def __init__(self, args, device):
        """action_type == 1 action continous
           action_type != 1 action discrete"""
        self.action_type = args.action_type
        self.action_mode = args.action_mode

        self.learn_type = args.learn_type

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.actor_state_dim = args.actor_state_dim
        self.critic_state_dim = args.critic_state_dim
        self.action_dim = args.action_dim

        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size

        self.horizon_steps = int(args.horizon_steps)
        self.update_steps = int(args.update_steps)

        self.lamda = args.lamda
        self.gamma = args.gamma
        self.lr = args.lr

        self.clip = args.clip
        self.max_grad_norm = args.max_grad_norm
        self.entropy_factor = args.entropy_factor

        self.delta_type = args.delta_type
        self.infinite = args.infinite
        self.norm = args.norm

        self.tau = args.tau

        self.seed = random.seed(args.seed)

        self.actor_eval = Actor(args).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=args.lr_anneal_step,
                                                               gamma=args.lr_anneal, last_epoch=-1)
        self.actor_target = Actor(args).to(self.device)

        self.Q_eval_01 = Q_net(args).to(self.device)
        self.Q_01_optimizer = optim.Adam(self.Q_eval_01.parameters(), lr=self.lr)
        self.Q_01_scheduler = torch.optim.lr_scheduler.StepLR(self.Q_01_optimizer, step_size=args.lr_anneal_step,
                                                                gamma=args.lr_anneal, last_epoch=-1)
        self.Q_target_01 = Q_net(args).to(self.device)

        self.Q_eval_02 = Q_net(args).to(self.device)
        self.Q_02_optimizer = optim.Adam(self.Q_eval_02.parameters(), lr=self.lr)
        self.Q_02_scheduler = torch.optim.lr_scheduler.StepLR(self.Q_02_optimizer, step_size=args.lr_anneal_step,
                                                                gamma=args.lr_anneal, last_epoch=-1)
        self.Q_target_02 = Q_net(args).to(self.device)

        # self.memory = Memory(self.horizon_steps, self.gamma, self.lamda)
        self.memory = Buffer(self.buffer_size, self.batch_size, self.device, self.seed)

        self.start_time = ""

        self.training_step = 0

        self.min_val = 1e-6

        self.target_entropy = -self.action_dim

        self.alpha_auto_entropy = args.alpha_auto_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha_scheduler = torch.optim.lr_scheduler.StepLR(self.alpha_optimizer, step_size=args.lr_anneal_step,
                                                                gamma=args.lr_anneal, last_epoch=-1)

        self.max_action = args.max_action

        # self.writer = SummaryWriter(args.tensorboard_savepath)

    def act(self, state, addNoise=True):
        #eps暂时没有使用
        state = torch.from_numpy(state).detach().float().unsqueeze(0).to(self.device)

        action = self.actor_eval(state)

        if addNoise == True:
            action = action + torch.rand_like(action).clamp(-self.clip, self.clip)

        action = action.detach().cpu().numpy()

        return np.clip(action.squeeze(0), -self.max_action, self.max_action)

    def step(self, eps, state, next_state, action, reward, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            return self.learn(eps, experiences)
        else:
            return 0, 0

    def learn(self, eps, experiences):
        """
        Clipped Double-Q Learning.使用两个Q函数进行学习，并在更新参数时使用其中最小的一个来避免value的过高估计。
        “Delayed” Policy Updates.对Q_Target网络以及policy网络都进行延时更新，避免更新过程中的累积误差。
        Target Policy Smoothing.对target action增加噪音，实现对Q函数进行平滑操作，减少policy的误差。
        """
        states, actions, rewards, next_states, dones = experiences

        noise = torch.rand_like(actions).clamp(-self.clip, self.clip)
        next_action = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        Q_target_1= self.Q_target_01(next_states, next_action).detach()
        Q_target_2 = self.Q_target_02(next_states, next_action).detach()
        target_Q = rewards + (1 - dones) * self.gamma * torch.min(Q_target_1, Q_target_2)

        q_eval_1 = self.Q_eval_01(states, actions)
        q_eval_2 = self.Q_eval_02(states, actions)

        q_01_loss = F.mse_loss(q_eval_1, target_Q)
        q_02_loss = F.mse_loss(q_eval_2, target_Q)

        self.Q_01_optimizer.zero_grad()
        q_01_loss.backward()
        self.Q_01_optimizer.step()

        self.Q_02_optimizer.zero_grad()
        q_02_loss.backward()
        self.Q_02_optimizer.step()

        if eps % self.update_steps == 0:
            new_q_eval_1 = self.Q_eval_01(states, self.actor_eval(states))
            new_q_eval_2 = self.Q_eval_02(states, self.actor_eval(states))
            actor_loss = -torch.min(new_q_eval_1, new_q_eval_2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.Q_eval_01, self.Q_target_01, self.tau)
            self.soft_update(self.Q_eval_02, self.Q_target_02, self.tau)
            self.soft_update(self.actor_eval, self.actor_target, self.tau)

            self.actor_scheduler.step()
            self.Q_01_scheduler.step()
            self.Q_02_scheduler.step()


    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



