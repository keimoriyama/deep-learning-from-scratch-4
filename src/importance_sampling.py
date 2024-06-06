import numpy as np

b = np.array([1 / 3, 1 / 3, 1 / 3])
x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])
e = np.sum(x * pi)
print(f"E_pi[x]: {e}")

n = 100
samples = []
for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)

mean = np.mean(samples)
std = np.std(samples)

print(f"MC: {mean} (var: {std})")
