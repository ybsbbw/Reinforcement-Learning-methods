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
          Current Task : PPO_ZWS | User : Wensheng Zhang
        ====================================================================================================
'''
import torch
import numpy as np
import random
from PPO_Buffer import Buffer
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from PPO_Model import Actor, Critic
from tensorboardX import SummaryWriter

class Agent():
    def __init__(self, env, device, buffer_size, batch_size, horizon_steps, update_steps=10,
                 lamda=0.95, gamma=0.99, lr=0.0001, tau=1e-3, runtype=1, clip=0.2, max_grad_norm=0.5, seed=0):

        """runtype == 1 action continous
           runtype != 1 action discrete"""
        self.runtype = runtype

        self.device = device

        self.state_dim = env.state_space
        self.action_dim = env.action_space

        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size

        self.horizon_steps = int(horizon_steps)
        self.update_steps = int(update_steps)

        self.lamda = lamda
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.seed = random.seed(seed)

        self.actor_eval = Actor(self.state_dim, self.action_dim, seed, runtype).to(self.device)
        # self.actor_target = Actor(self.state_typesize, self.action_typesize, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.lr)

        self.critic_eval = Critic(self.state_dim, seed, runtype).to(self.device)
        # self.critic_target = Critic(self.state_typesize, self.action_typesize, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.memory = Buffer(self.horizon_steps, self.gamma, self.lamda, self.device)

        self.training_step = 0
        self.writer = SummaryWriter('../PPO_continuous_ZWS/exp')

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        if self.runtype == 1:
            a_mean, a_std = self.actor_eval(state)
            dist = Normal(a_mean, a_std.to(self.device))
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            return action.item(), action_log_prob.item()
        else:
            action_prob = self.actor_eval(state)
            action = Categorical(action_prob).sample()
            return action.item(), action_prob[action.item()].item()

    def step(self, t, state, action, reward, done, next_state, action_pro):
        inout_state = torch.from_numpy(state).float().to(self.device)
        value = self.critic_eval(inout_state)
        if done:
            terminal = 1
        else:
            terminal = 0

        self.memory.add(state, action, reward, terminal, next_state, action_pro, value.item())

        if (t+1) % self.horizon_steps == 0 or terminal == 1:
            next_state = torch.from_numpy(next_state).float().to(self.device)
            self.memory.value_list.append(self.critic_eval(next_state).item())
            self.memory.compute(delta_type=1, infinite=1, norm=1)
            self.learn()
            self.memory.reset()

    def learn(self):
        for update_index in range(self.update_steps):
            sample_list = self.memory.sample(self.batch_size)
            for i in sample_list:
                states = self.memory.state_list[i]
                actions = self.memory.action_list[i]
                rewards = self.memory.reward_list[i]
                dones = self.memory.done_list[i]
                next_states = self.memory.next_state_list[i]
                old_log_probs = self.memory.log_prob_list[i]
                values = self.memory.value_list[i]

                Qs = self.memory.Q_list[i]
                deltas = self.memory.delta_list[i]
                GAEs = self.memory.GAE_list[i]

                if self.runtype == 1:
                    a_mean, a_std = self.actor_eval(states)
                    dist = Normal(a_mean, a_std.to(self.device))
                    action_log_probs = dist.log_prob(actions)
                    ratio = torch.exp(action_log_probs - old_log_probs)
                else:
                    action_probs = self.actor_eval(states).gather(1, actions.long())
                    ratio = action_probs/old_log_probs

                surr1 = ratio * GAEs
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * GAEs

                action_loss = -torch.min(surr1, surr2).mean()
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_eval.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(Qs, self.critic_eval(states))
                self.writer.add_scalar('loss/value_loss', critic_loss, global_step=self.training_step)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_eval.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

        return action_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy()

# class Replaybuffer():
#     def __init__(self, buffer_size, batch_size, device, seed=0.):
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#
#         self.device = device
#         self.seed = random.seed(seed)
#         # self.store = []
#         self.store = deque(maxlen=self.buffer_size)
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "value", "action_pro"])
#
#         self.result = deque(maxlen=self.buffer_size)
#
#     def add(self, state, action, reward, next_state, done, value, action_pro):
#         store_add = self.experience(state, action, reward, next_state, done, value, action_pro)
#         self.store.append(store_add)
#
#     def set_to_pool(self, state, action, reward, next_state, done, value, action_pro):
#         sample_add = self.experience(state, action, reward, next_state, done, value, action_pro)
#         self.result.append(sample_add)
#
#     def sample(self):
#         # experience = random.sample(self.result, self.batch_size)
#         experience = self.result
#         # random.shuffle(experience)
#
#         states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(self.device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long().to(self.device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(self.device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)
#         values = torch.from_numpy(np.vstack([e.value for e in experience if e is not None])).float().to(self.device)
#         action_pros = torch.from_numpy(np.vstack([e.action_pro for e in experience if e is not None])).float().to(self.device)
#         # selected_actions = torch.from_numpy(np.vstack([e.selected_action for e in experience if e is not None])).long().to(self.device)
#
#         return (states, actions, rewards, next_states, dones, values, action_pros)
#
#     def reset(self):
#         # self.store = []
#         self.store.clear()
#         self.result.clear()
#         # self.store = deque(maxlen=self.buffer_size)
#
#     def __len__(self):
#         return len(self.store)