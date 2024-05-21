from gridworld import GridWorld
import numpy as np

env = GridWorld()
V = {}
for state in env.states():
    V[state] = np.random.randn()

env.render_v(V)
