import torch
import numpy as np
import random
from collections import namedtuple, deque
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
from A2C_Model import Advantage_Actor_Critic
# from A2C_Model import A2C_nn

class Agent():
    def __init__(self, state_typesize, action_typesize, device, batch_size, buffer_size, lamda=20,
                 learning_rate=0.005, gamma=0.99, tau=1e-3, seed=0.):

        self.state_typesize = state_typesize
        self.action_typesize = action_typesize
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.n_step = lamda
        self.steps = 0

        self.CLIP_GRAD = 0.1

        self.seed = random.seed(seed)

        self.A2C_eval = Advantage_Actor_Critic(state_typesize, action_typesize, seed).to(device)
        # self.A2C_target = Advantage_Actor_Critic(state_typesize, action_typesize, seed).to(device)
        self.A2C_optimizer = optim.Adam(self.A2C_eval.parameters(), lr=self.lr)

        self.ENTROPY_BETA = 0.01
        # self.actor_eval = A2C_nn(state_typesize, action_typesize, seed)[0].to(device)
        # self.critic_eval = A2C_nn(state_typesize, action_typesize, seed)[1].to(device)

        self.memory = Replaybuffer(self.batch_size, self.buffer_size, self.device)
        self.memory_record = []
        self.trajectory = Replaybuffer(self.batch_size, self.buffer_size, self.device)
        self.trace_record = []

        # self.noise = OUNoise(action_size, random_seed)
        self.current_value = 0

    def act(self, state):
        #eps暂时没有使用
        state = torch.from_numpy(state).float().to(self.device)

        actions_eval = self.A2C_eval(state)[0]
        P_actions = F.softmax(actions_eval)

        selected_actions = int(np.random.choice(np.arange(self.action_typesize), p=P_actions.cpu().data.numpy(), size=1))
        # action = np.random.choice(np.arange(self.action_typesize))

        # if random.random() >= eps:
        return selected_actions
        # else:
        #     return np.random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.trace_record.append(self.trajectory.experience(state, action, reward, next_state, done))
        state = torch.from_numpy(state).float().to(self.device)

        self.steps += 1
        if self.steps % self.n_step == 0:
            next_state = torch.from_numpy(next_state).float().to(self.device)
            self.current_value = self.A2C_eval(next_state)[1].cpu().detach().numpy()
            self.state_value()
            self.trace_record = []
        #     # if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
        #     #     return self.learn(experiences)
            loss = self.learn(experiences)
        #
            self.steps = 0

            return loss
        return 0

    def state_value(self):
        self.memory.reset()
        for i in reversed(range(len(self.trace_record))):
            if self.trace_record[i].done:
                self.current_value = 0
            self.current_value = self.trace_record[i].reward + self.current_value * self.gamma
            self.memory.add(self.trace_record[i].state, self.trace_record[i].action, self.current_value, self.trace_record[i].next_state, self.trace_record[i].done)

        return self.memory

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        #states, actions, rewards, next_states, actions_eval, vl_eval =1,2,3,4,[5,6],7
        actions_eval = self.A2C_eval(states)[0]
        vl_eval = self.A2C_eval(states)[1]

        P_log_softmax = F.log_softmax(actions_eval, dim=1)

        action_log_softmax = P_log_softmax.gather(1, actions)

        adv = rewards - vl_eval.detach()

        pg_loss = - torch.mean(action_log_softmax * adv)

        vl_loss = F.mse_loss(vl_eval, rewards)

        P_actions = F.softmax(actions_eval, dim=1)

        entropy_loss = - self.ENTROPY_BETA * torch.mean(torch.sum(P_actions * P_log_softmax, dim=1))

        loss = pg_loss + vl_loss + entropy_loss

        self.A2C_optimizer.zero_grad()

        loss.backward()

        clip_grad_norm_(self.A2C_eval.parameters(), self.CLIP_GRAD)
        self.A2C_optimizer.step()

        # self.soft_update(self.A2C_eval, self.A2C_target, self.tau)
        # self.soft_update(self.actor_eval, self.actor_target, self.tau)

        # return lossC, lossA
        return loss

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Replaybuffer():
    def __init__(self, batch_size, buffer_size, device, seed=0.):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.seed = random.seed(seed)
        self.store = []
        # self.store = deque(maxlen=self.buffer_size )
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        store_add = self.experience(state, action, reward, next_state, done)
        self.store.append(store_add)

    def sample(self):
        # experience = random.sample(self.store, self.batch_size)
        experience = self.store

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)
        # selected_actions = torch.from_numpy(np.vstack([e.selected_action for e in experience if e is not None])).long().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def reset(self):
        self.store = []

    def __len__(self):
        return len(self.store)