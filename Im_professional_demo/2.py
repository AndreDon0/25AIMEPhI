import numpy as np

class NormalDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def compile(self, x):
        return (1 / (2 * np.pi * np.linalg.det(self.sigma))) * \
               np.exp((-1 / 2) * (x - self.mu).T @ np.linalg.inv(self.sigma) @ (x - self.mu))
    

P1 = NormalDistribution([2, 3], [[2, 0], [0, 1]])
P2 = NormalDistribution([1, 0], [[3, -1], [-1, 4]])

print(np.linalg.eig([[19/22, 2/22], [2/22, 28/22]]))