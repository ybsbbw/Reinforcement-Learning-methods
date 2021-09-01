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
from SAC_Buffer import Memory, Buffer
from SAC_Model import Actor, Value, Q_net
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

        self.value_eval = Value(args).to(self.device)
        self.value_eval_optimizer = optim.Adam(self.value_eval.parameters(), lr=self.lr)
        self.value_eval_scheduler = torch.optim.lr_scheduler.StepLR(self.value_eval_optimizer, step_size=args.lr_anneal_step,
                                                                gamma=args.lr_anneal, last_epoch=-1)
        self.value_target = Value(args).to(self.device)

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

        self.min_val = 1e-7

        self.target_entropy = -self.action_dim

        self.alpha_auto_entropy = args.alpha_auto_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha_scheduler = torch.optim.lr_scheduler.StepLR(self.alpha_optimizer, step_size=args.lr_anneal_step,
                                                                gamma=args.lr_anneal, last_epoch=-1)
        # self.writer = SummaryWriter(args.tensorboard_savepath)

    def rp_action(self, a_mean, a_std):
        """再参数化
        reparameterisation trick: 两种处理方式均可
        dist = Normal(a_mean, a_std)
        normal = Normal(0, 1)
        z = normal.sample()
        return_action = torch.tanh(mean + std * z.to(self.device))
        action_log_prob = dist.log_prob(mean + std * z.to(self.device)) - torch.log(1 - return_action.pow(2) + self.min_val)"""
        dist = Normal(a_mean, a_std)
        action = dist.rsample()  # rsample means it is sampled using reparameterisation trick
        return_action = torch.tanh(action)
        action_log_prob = dist.log_prob(action) - torch.log(1 - return_action.pow(2) + self.min_val)

        return return_action, action_log_prob

    def act(self, state, type=0):
        state = torch.from_numpy(state).detach().float().unsqueeze(0).to(self.device)

        if self.action_type == 1:
            if type == 0:
                a_mean, log_std = self.actor_eval(state)
                a_std = log_std.exp()
                action, action_log_prob = self.rp_action(a_mean, a_std)

                return action.detach().cpu().numpy().squeeze(0), action_log_prob.detach().cpu().numpy().squeeze(0)

            else:
                a_mean, log_std = self.actor_eval(state)
                a_std = log_std.exp()
                dist = Normal(a_mean, a_std)
                action = dist.sample()
                return_action = torch.tanh(action)
                action_log_prob = dist.log_prob(action) - torch.log(1 - return_action.pow(2) + self.min_val)

                return return_action.detach().cpu().numpy().squeeze(0), action_log_prob.detach().cpu().numpy().squeeze(0)

        else:
            p = self.actor_eval(state)
            if self.action_mode == 0:
                a = Categorical(p).sample()
            else:
                a = torch.argmax(p, dim=-1)
            action = a.detach().cpu().numpy().squeeze(0)
            action_log_prob = torch.log(p + self.min_val).detach().cpu().numpy().squeeze(0)

            return action, action_log_prob

    def step(self, eps, state, next_state, action, reward, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size and (eps+1) % self.update_steps == 0:
            experiences = self.memory.sample()
            if self.learn_type == 1:
                self.alpha = 1
                self.learn_v1(experiences)
            else:
                self.alpha = self.log_alpha.exp()
                self.learn_v2(experiences)

    def get_alpha(self, log_prob):
        if self.alpha_auto_entropy == 1:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn_v1(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        if self.action_type == 1:
            a_mean, log_std = self.actor_eval(states)
            a_std = log_std.exp()
            new_action, action_log_prob = self.rp_action(a_mean, a_std)

            self.get_alpha(action_log_prob)

            q_eval = self.Q_eval_01(states, actions)
            value_eval = self.value_eval(states)
            target_value = self.value_target(next_states)
            next_q_eval = rewards + (1 - dones) * self.gamma * target_value
            Q_loss = F.mse_loss(q_eval, next_q_eval.detach())

            self.Q_01_optimizer.zero_grad()
            Q_loss.backward()
            self.Q_01_optimizer.step()

            new_q_eval = self.Q_eval_01(states, new_action)
            next_value = new_q_eval - self.alpha * action_log_prob
            value_loss = F.mse_loss(value_eval, next_value.detach())

            self.value_eval_optimizer.zero_grad()
            value_loss.backward()
            self.value_eval_optimizer.step()

            policy_loss = (self.alpha * action_log_prob - new_q_eval).mean()  # policy_loss = - next_value.mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        else:
            p = self.actor_eval(states)
            if self.action_mode == 0:
                a = Categorical(p).sample()
            else:
                a = torch.argmax(p, dim=-1)

            new_action = a.unsqueeze(1).float()
            action_log_prob = torch.log(p + self.min_val)  # .gather(1, new_action.long())

            self.get_alpha(action_log_prob)

            q_eval = self.Q_eval_01(states).gather(1, actions.long())
            value_eval = self.value_eval(states)
            target_value = self.value_target(next_states)
            next_q_eval = rewards + (1 - dones) * self.gamma * target_value
            Q_loss = F.mse_loss(q_eval, next_q_eval.detach())

            self.Q_01_optimizer.zero_grad()
            Q_loss.backward()
            self.Q_01_optimizer.step()

            new_q_eval = self.Q_eval_01(states)
            next_value = torch.sum(p * (new_q_eval - self.alpha * action_log_prob), dim=1).unsqueeze(1)
            value_loss = F.mse_loss(value_eval, next_value.detach())

            self.value_eval_optimizer.zero_grad()
            value_loss.backward()
            self.value_eval_optimizer.step()

            policy_loss = (p * (self.alpha * action_log_prob - new_q_eval)).sum(dim=1).mean()  # policy_loss = - next_value.mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        self.soft_update(self.value_eval, self.value_target, self.tau)

        self.actor_scheduler.step()
        self.value_eval_scheduler.step()
        self.Q_01_scheduler.step()

    def learn_v2(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        if self.action_type == 1:
            a_mean, log_std = self.actor_eval(states)
            a_std = log_std.exp()
            new_action, action_log_prob = self.rp_action(a_mean, a_std)

            next_a_mean, next_log_std = self.actor_eval(next_states)
            next_a_std = next_log_std.exp()
            next_new_action, next_action_log_prob = self.rp_action(next_a_mean, next_a_std)

            q_eval_1 = self.Q_eval_01(states, actions)
            q_eval_2 = self.Q_eval_02(states, actions)

            next_q_eval_1 = self.Q_target_01(next_states, next_new_action)
            next_q_eval_2 = self.Q_target_02(next_states, next_new_action)

            self.get_alpha(action_log_prob)

            q_eval_min = torch.min(next_q_eval_1, next_q_eval_2) - self.alpha * next_action_log_prob
            target_q = rewards + (1 - dones) * self.gamma * q_eval_min  # 如果 done==1, only reward
            q_01_loss = F.mse_loss(q_eval_1, target_q.detach())  # 加 detach: no gradients for the variable
            q_02_loss = F.mse_loss(q_eval_2, target_q.detach())

            self.Q_01_optimizer.zero_grad()
            q_01_loss.backward()
            self.Q_01_optimizer.step()

            self.Q_02_optimizer.zero_grad()
            q_02_loss.backward()
            self.Q_02_optimizer.step()

            next_q_target_min = torch.min(self.Q_eval_01(states, new_action), self.Q_eval_02(states, new_action))
            policy_loss = (self.alpha * action_log_prob - next_q_target_min).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        else:
            p = self.actor_eval(states)
            if self.action_mode == 0:
                a = Categorical(p).sample()
            else:
                a = torch.argmax(p, dim=-1)

            new_action = a.unsqueeze(1).float()
            action_log_prob = torch.log(p + self.min_val)

            next_p = self.actor_eval(next_states)
            if self.action_mode == 0:
                next_a = Categorical(p).sample()
            else:
                next_a = torch.argmax(p, dim=-1)

            next_new_action = next_a.unsqueeze(1).float()
            next_action_log_prob = torch.log(next_p + self.min_val)

            q_eval_1 = self.Q_eval_01(states).gather(1, actions.long())
            q_eval_2 = self.Q_eval_02(states).gather(1, actions.long())

            next_q_eval_1 = self.Q_target_01(next_states)
            next_q_eval_2 = self.Q_target_02(next_states)

            self.get_alpha(action_log_prob)

            q_eval_min = p * (torch.min(next_q_eval_1, next_q_eval_2) - self.alpha * next_action_log_prob)
            target_q = rewards + (1 - dones) * self.gamma * q_eval_min.sum(dim=1).unsqueeze(1)  # 如果 done==1, only reward
            q_01_loss = F.mse_loss(q_eval_1, target_q.detach())  # 加detach: no gradients for the variable
            q_02_loss = F.mse_loss(q_eval_2, target_q.detach())

            self.Q_01_optimizer.zero_grad()
            q_01_loss.backward()
            self.Q_01_optimizer.step()

            self.Q_02_optimizer.zero_grad()
            q_02_loss.backward()
            self.Q_02_optimizer.step()

            next_q_target_min = torch.min(self.Q_eval_01(states), self.Q_eval_02(states))
            policy_loss = (p * (self.alpha * action_log_prob - next_q_target_min)).sum(dim=1).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        self.soft_update(self.Q_eval_01, self.Q_target_01, self.tau)
        self.soft_update(self.Q_eval_02, self.Q_target_02, self.tau)

        self.actor_scheduler.step()
        self.Q_01_scheduler.step()
        self.Q_02_scheduler.step()