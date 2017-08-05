# Gaussian Process
# refer: https://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf

import numpy as np
import matplotlib.pyplot as plt


train_data = np.array([[1, 1], [3, 2], [4, 5]])
test_data = np.linspace(0, 5, 1000)

len_train = len(train_data)
len_test = len(test_data)


def cov(x):
    """
    Calculate covariance matrix given a 1-D vector
    """
    def kernal(x1, x2):
        l = 1.0
        nu = 1.27
        return nu**2 * np.exp(-(x1 - x2)**2 / (2 * l**2))

    matrix = []
    for x1 in x:
        row = []
        for x2 in x:
            row.append(kernal(x1, x2))
        matrix.append(row)
    return np.array(matrix)


x = np.concatenate((train_data[:, 0], test_data))
matrix = cov(x)

k00 = matrix[:len_train, :len_train]  # 3x3
k01 = matrix[:len_train, len_train:]  # 3x1000
k10 = matrix[len_train:, :len_train]  # 1000x3
k11 = matrix[len_train:, len_train:]  # 1000x1000

mean = np.dot(k10, np.linalg.inv(k00)).dot(train_data[:, 1])  # 1000,

cov = k11 - np.dot(k10, np.linalg.inv(k00)).dot(k01)    # 1000x1000

# sample from a multivariate normal distribution
sample_fxs = np.random.multivariate_normal(mean, cov, size=100)

for f in sample_fxs:
    plt.plot(test_data, f, alpha=0.6)
plt.show()
