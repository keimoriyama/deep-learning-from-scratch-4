import torch.nn as nn
import torch
import numpy as np
from torch import optim
import copy
import gym
from replay_buffer import RelplayBuffer


class PolicyNet(nn.Module):
    def __init__(self, action_size) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.softmax(self.l2(x))
        return x


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        return self.l2(x)


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), self.lr)
        self.optimizer_v = optim.Adam(self.v.parameters(), self.lr)
        self.mse = nn.MSELoss()

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0))
        probs = probs[0]
        action = torch.multinomial(probs, 1)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = torch.tensor(state).unsqueeze(0)
        next_state = torch.tensor(next_state).unsqueeze(0)

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        v = self.v(state)
        loss_v = self.mse(v, target)
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta

        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_pi.backward(retain_graph=True)
        loss_v.backward()
        self.optimizer_pi.step()
        self.optimizer_v.step()


env = gym.make("CartPole-v1")
state = env.reset()
state = state[0]
agent = Agent()
episodes = 2000
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action, prob = agent.get_action(state)
        action = action.detach().numpy()[0]
        next_state, reward, done, info, _ = env.step(action)
        agent.update(state, prob, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"episode {episode} reward {total_reward}")
