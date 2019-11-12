import torch
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
from DDPG_Model import Actor
from DDPG_Model import Critic

class Agent():
    def __init__(self, state_typesize, action_typesize, device, batch_size, buffer_size,
                 learning_rate=0.005, gamma= 0.99, tau=1e-3, seed=0.):

        self.state_typesize = state_typesize
        self.action_typesize = action_typesize
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = learning_rate
        self.gamma = gamma
        # self.gamma = 1
        self.tau = tau

        self.seed = random.seed(seed)

        self.actor_eval = Actor(state_typesize, action_typesize, seed).to(device)
        self.actor_target = Actor(state_typesize, action_typesize, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.lr)

        self.critic_eval = Critic(state_typesize, action_typesize, seed).to(device)
        self.critic_target = Critic(state_typesize, action_typesize, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.memory = Replaybuffer(self.batch_size, self.buffer_size, self.device)
        # self.noise = OUNoise(action_size, random_seed)


    def act(self, state):
        #eps暂时没有使用
        state = torch.from_numpy(state).float().to(self.device)


        action = self.actor_eval(state).cpu().data.numpy()
        # action = np.random.choice(np.arange(self.action_typesize))

        return np.clip(action, -2, 2)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            return self.learn(experiences)
        else:
            return 0, 0

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions) * (1-dones)

        Q_targets = rewards + self.gamma * next_Q_targets
        Q_expected = self.critic_eval(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        lossC = critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        actions_eval = self.actor_eval(states)
        # print("actions==", actions, "\n", "actions_eval==",actions_eval,"\n")
        # actor_loss = -self.critic_eval(states, next_actions).mean()
        actor_loss = -self.critic_eval(states, actions_eval).mean()
        lossA = actor_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # next_actions = self.actor_target(next_states)
        # next_Q_targets = self.critic_target(next_states, next_actions) * (1-dones)
        #
        # Q_targets = rewards + self.gamma * next_Q_targets
        # Q_expected = self.critic_eval(states, actions)
        #
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # lossC = critic_loss
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        #
        #
        # actions_eval = self.actor_eval(states)
        # # print("actions==", actions, "\n", "actions_eval==",actions_eval,"\n")
        # actor_loss = -self.critic_eval(states, actions_eval).mean()
        # lossA = actor_loss
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        self.soft_update(self.critic_eval, self.critic_target, self.tau)
        self.soft_update(self.actor_eval, self.actor_target, self.tau)

        return lossC, lossA

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Replaybuffer():
    def __init__(self, batch_size, buffer_size, device, seed=0.):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.seed = random.seed(seed)

        self.store = deque(maxlen=self.buffer_size )
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        store_add = self.experience(state, action, reward, next_state, done)
        self.store.append(store_add)

    def sample(self):
        experience = random.sample(self.store, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.store)