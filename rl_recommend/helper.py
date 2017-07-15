from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class Helper(object):
    user_history = []
    @classmethod
    def gen_state(cls, favor):
        favor = np.copy(favor)
        favor -= np.mean(favor)
        if not (favor == 0).all():
            favor /= np.std(favor)
        return favor

    @classmethod
    def gen_reward(cls, rec, room, tips):
        return tips if rec == room else 0.0

    @classmethod
    def gen_exp(cls, state, rec, room, tips, next_state):
        reward = cls.gen_reward(rec, room, tips)
        done = False
        return {'state': state, 'action': rec, 'reward': reward,
                'next_state': next_state, 'done': done}

    @staticmethod
    def log(favor, visit_stat):
        num_rooms = len(favor)
        fig1 = plt.figure('user_favor')
        plt.plot(range(num_rooms), favor)

        fig2 = plt.figure('visit_stat')
        plt.plot(range(num_rooms), visit_stat['times'])
        plt.plot(range(num_rooms), visit_stat['tips'])

        plt.show()
