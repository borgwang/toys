import numpy as np


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def softmax(y):
    y = y - y.max(1, keepdims=True)
    return np.exp(y) / np.exp(y).sum(1, keepdims=True)


def is_numerical(val):
    return isinstance(val, int) or isinstance(val, float)
