import torch.nn as nn
import torch
import numpy as np
from torch import optim
import copy
import gym
from replay_buffer import RelplayBuffer


class Policy(nn.Module):
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


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), self.lr)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0))
        probs = probs[0]
        # action = np.random.choice(len(probs), p=probs)
        action = torch.multinomial(probs, 1)
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.optimizer.zero_grad()
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        for reward, prob in self.memory:
            loss += -torch.log(prob) * G

        loss.backward()
        self.optimizer.step()
        self.memory = []


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
        agent.add(reward, prob)
        state = next_state
        total_reward += reward

    agent.update()
    print(f"episode {episode} reward {total_reward}")
