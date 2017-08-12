from __future__ import division

import numpy as np


class KFlod(object):

    def __init__(self, X, Y, k=10):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        assert len(X) == len(Y) and len(X) > 0, 'Invalid data!'
        self._total_size = len(X)
        self.__num_k = k
        assert self.__num_k <= self._total_size, 'Invalid K!'
        self.current_k = 0
        self._val_size = int(self._total_size / self.__num_k)
        self._train_size = self._total_size - self._val_size

    def next_segementation(self):
        val_start_idx = self.current_k * self._val_size
        val_end_idx = val_start_idx + self._val_size
        self.current_k = (self.current_k + 1) % self.__num_k
        val_x = self.X[val_start_idx: val_end_idx, :]
        val_y = self.Y[val_start_idx: val_end_idx, :]
        train_x = np.vstack(
            (self.X[:val_start_idx, :], self.X[val_end_idx:, :]))
        train_y = np.vstack(
            (self.Y[:val_start_idx, :], self.Y[val_end_idx:, :]))

        return {
            'train': {
                'inputs': train_x, 'labels': train_y, 'size': self._train_size},
            'val': {
                'inputs': val_x, 'labels': val_y, 'size': self._val_size}}
