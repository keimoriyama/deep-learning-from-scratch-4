import numpy as np
import matplotlib.pyplot as plt
from bandit import Agent


class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


epsilon = 0.1
steps = 10000
bandit = NonStatBandit()
agent = Agent(epsilon)
total_reward = 0
rates = []
for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward
    rates.append(total_reward / (step + 1))

alphaAgent = AlphaAgent(epsilon, alpha=0.8)
alphaRates = []
alpha_total_reward = 0
for step in range(steps):
    action = alphaAgent.get_action()
    reward = bandit.play(action)
    alphaAgent.update(action, reward)
    alpha_total_reward += reward
    alphaRates.append(alpha_total_reward / (step + 1))


plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(rates, label="sample average")
plt.plot(alphaRates, label="alpha const update")
plt.legend()
plt.show()
