import numpy as np

np.random.seed(0)
rewards = []
Q = 0

for n in range(1, 11):
    reward = np.random.rand()
    # rewards.append(reward)
    # Q = sum(rewards) / n
    # print(Q)
    Q = Q + (reward - Q) / n
    print(Q)
