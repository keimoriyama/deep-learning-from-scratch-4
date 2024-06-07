from collections import deque
import random
import numpy as np
import gym


class RelplayBuffer:
    def __init__(self, buffer_size, batch_size) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.stack([x[4] for x in data])
        return state, action, reward, next_state, done


env = gym.make("CartPole-v1")
replay_buffer = RelplayBuffer(buffer_size=10000, batch_size=32)
