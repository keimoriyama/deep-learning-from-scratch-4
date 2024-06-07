import torch.nn as nn
import torch
import numpy as np
from torch import optim
import copy
import gym
from replay_buffer import RelplayBuffer


class QNet(nn.Module):
    def __init__(self, action_size) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr = 0.00005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2
        self.replay_buffer = RelplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)

        self.optimizer = optim.Adam(self.qnet.parameters(), self.lr)
        self.mse = nn.MSELoss()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        else:
            state = torch.tensor(state).unsqueeze(0)
            qs = self.qnet(state)
            action = qs.argmax().detach().numpy()
            return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        # print(state)
        qs = self.qnet(torch.tensor(state))
        q = qs[np.arange(self.batch_size), action]
        next_qs = self.qnet_target(torch.tensor(next_state))
        next_q = next_qs.max(dim=1).values.detach().numpy()

        target = reward + (1 - done) * self.gamma * next_q
        target = torch.from_numpy(target.astype(np.float32)).clone()
        loss = self.mse(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


episodes = 10000
sync_interval = 20
env = gym.make("CartPole-v1")
agent = DQNAgent()
reward_history = []
for episode in range(episodes):
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()
    print(f"episode {episode} reward {total_reward}")
    reward_history.append(total_reward)
