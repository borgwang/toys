import numpy as np

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    noises = []
    for i in range(10000):
        noises.append(ou.noise())

    import matplotlib.pyplot as plt
    plt.plot(noises)
    plt.show()
