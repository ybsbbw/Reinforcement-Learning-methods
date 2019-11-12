import torch
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F
from DQN_Model import QNetwork

UPDATE_EVERY = 4
dir = "C:\\Users\\a2\Desktop\\projects\\MoutainCar-v0\\DQN_ZWS"

class Agent():
    def __init__(self, state_size, action_size,  device, batch_size, buffer_size, learning_rate=0., gamma=0., tau=0., seed=0.):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate
        self.device = device
        self.Batch_size = batch_size
        self.Buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau

        self.t_step = 0

        self.Qnet_eval = QNetwork(self.state_size, self.action_size).to(device)
        self.Qnet_target = QNetwork(self.state_size, self.action_size).to(device)

        self.optimizer = torch.optim.Adam(self.Qnet_eval.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(self.action_size, self.Batch_size, self.Buffer_size, self.device)

    def act(self, states, eps=0.):
        states = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
        Q_actions = self.Qnet_eval(states)   #Tensor shape==[1,3]
        selected_actions = np.argmax(Q_actions.cpu().data.numpy())

        # self.Qnet_eval.eval()
        # with torch.no_grad():
        #     action = self.Qnet_eval(states).cpu().data.numpy()
        # self.Qnet_eval.train()


        if random.random() >= eps:
            return selected_actions
        else:
            return np.random.choice(np.arange(self.action_size))


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        self.t_step = self.t_step % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.Batch_size:
                store = self.memory.sample()
                return self.learn(store, self.gamma)

    def learn(self, store, gamma):
        states, actions, rewards, next_states, dones = store

        # 按照action作为索引依次在[Batch_size，3]的Qnet_eval(states)上取值，结果为
        Q_eval = self.Qnet_eval(states).gather(1, actions)    #Tesnor  shape==[Batch_size，1]

        Q_target = self.Qnet_target(next_states).max(1)[0].unsqueeze(1)*(1-dones)
        #max(dim=1)将最大值的结果按列（dim=1）进行合并，[0]对应值，[1]对应索引
        Q_new_target = rewards + gamma*Q_target

        loss = F.mse_loss(Q_eval, Q_new_target)
        rloss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for eval_param, target_param in zip(self.Qnet_eval.parameters(), self.Qnet_target.parameters()):
            target_param.data.copy_((1-self.tau)*target_param.data+self.tau*eval_param.data)
        return rloss

class ReplayBuffer():
    def __init__(self, action_size, batch_size, buffer_size, device, seed=0):
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.seed = random.seed(seed)

        self.store = deque(maxlen=self.buffer_size,)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        ep = self.experience(state, action, reward, next_state, done)
        self.store.append(ep)

    def sample(self):
        experience = random.sample(self.store, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)
#        print('\rstates: {}, actions: {}, rewards: {}, next_states: {}, dones: {}'.format(states, actions, rewards, next_states, dones), end=' ')

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.store)

