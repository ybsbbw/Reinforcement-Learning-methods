import torch
import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
from ddpg.DDPG_Model import Actor
from ddpg.DDPG_Model import Critic

class Agent():
    def __init__(self, state_typesize, action_typesize, device, buffer_size, batch_size,
                 learning_rate=1e-4, gamma= 0.99, tau=1e-3, seed=0.):

        self.state_typesize = state_typesize
        self.action_typesize = action_typesize
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = learning_rate
        self.gamma = gamma
        # self.gamma = 1
        self.tau = tau

        self.action_max = np.array([415, 360, 360])
        self.action_min = np.array([0, 0, 0])

        self.seed = random.seed(seed)

        self.actor_eval = Actor(state_typesize, action_typesize, seed).to(device)
        self.actor_target = Actor(state_typesize, action_typesize, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.lr)

        self.critic_eval = Critic(state_typesize, action_typesize, seed).to(device)
        self.critic_target = Critic(state_typesize, action_typesize, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr * 5)

        # self.memory = Replaybuffer(self.batch_size, self.buffer_size, self.device)
        self.noise = OUNoise(action_typesize, seed)

    def reset(self):
        self.noise.reset()

    def act(self, state, addNoise=False):
        #eps暂时没有使用
        state = torch.from_numpy(state).detach().float().unsqueeze(0).to(self.device)

        self.actor_eval.eval()
        with torch.no_grad():
            action = self.actor_eval(state).cpu().data.numpy().squeeze()
        self.actor_eval.train()
        # action = np.random.choice(np.arange(self.action_typesize))
        # result = self.action_min + (self.action_max - self.action_min) * np.random.rand(len(self.action_max))

        if addNoise == True:
            action += self.noise.sample()

        return action

    def step(self, state, action, score):

        state = torch.tensor([state]).float().to(self.device)
        action = torch.tensor([action]).float().to(self.device)
        score = torch.tensor([score]).float().to(self.device)

        self.learn(state, action, score)

    def learn(self, state, action, score):

        Q_targets = score
        Q_expected = self.critic_eval(state, action)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        lossC = critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_eval = self.actor_eval(state)
        actor_loss = -self.critic_eval(state, actions_eval).mean()
        lossA = actor_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_eval, self.critic_target, self.tau)
        self.soft_update(self.actor_eval, self.actor_target, self.tau)

        return lossC, lossA

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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

class Replaybuffer():
    def __init__(self, batch_size, buffer_size, device, seed=0.):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.seed = random.seed(seed)

        self.store = deque(maxlen=self.buffer_size)
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
