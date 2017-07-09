from __future__ import division

import numpy as np
from agent import Agent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class User(object):
    def __init__(self, popularity, favor, earning_ability):
        # enter_prob is determined by room popularity and user favor.
        # tip_prob is determined by the earning_ability of a romm and user favor.
        # The weights of two these factors suppose to be different for different users.
        self.enter_prob = popularity * 0.3 + favor * 0.7
        self.tip_prob = earning_ability * 0.3 + favor * 0.7
        self.stay_prob = np.array(favor)

        self.history = []

        self.recommender = Agent()
        self.running_tip = None
        # self.avg_reward = np.mean(-(np.log(1 - np.asarray(enter_prob)) + np.log(1 - np.array(stay_prob))))

    def run(self):
        for step in range(500000):
            rec, room = self._start()
            tips, stay_time = 0, 0
            while True:
                stay_time += 1
                tips += self._tip(room)
                if not self._is_stay(room):
                    state, reward, next_state = self._leave_room(rec, room, tips, stay_time)
                    self.recommender.update(state, room, reward, next_state, False)
                    break

            if step % 1000 == 0 and step:
                self._log(step)

    def _start(self):
        rec = self.recommender.random_act(self._gen_state(self.history))
        if np.random.random() < self.enter_prob[rec]:
            pass
        room = np.random.choice(range(5), p=self.enter_prob)
        return rec, room

    def _tip(self, room):
        return 1 if np.random.random() < self.tip_prob[room] else 0

    def _is_stay(self, room):
        return True if np.random.random() < self.stay_prob[room] else False

    def _leave_room(self, rec, room, tips, stay_time):
        state = self._gen_state(self.history)
        self.history.append([room, tips, stay_time])
        next_state = self._gen_state(self.history)
        reward = self._cal_reward(rec)
        return state, reward, next_state

    def _cal_reward(self, rec):
        reward = -(np.log(1-self.enter_prob[rec]) + np.log(1-self.stay_prob[rec]))
        return reward

    def _log(self, step):
        history = np.asarray(self.history)
        rooms = history[-1000:, 0]
        tips = history[-1000:, 1]
        times = history[-1000:, 2]
        if self.running_tip:
            self.running_tip = self.running_tip * 0.99 + np.mean(tips) * 0.01
        else:
            self.running_tip = np.mean(tips)
        print 'step: %d avg_tip: %.4f staytime: %.2d ' % \
                (step, self.running_tip, np.mean(times))

    @staticmethod
    def _gen_state(history):
        state = np.array(history[-5:])
        if len(state) == 0:
            return np.zeros((5,3))
        elif len(state) < 5:
            return np.vstack((np.zeros([5-len(state), 3]), state))
        else:
            return state

    @staticmethod
    def softmax(dis):
        return np.exp(dis - np.max(dis)) / np.sum(np.exp(dis - np.max(dis)))

# simple user model
popularity = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
favor = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
earning_ability = np.array([0.5, 0.2, 0.1, 0.1, 0.1])   # the earning ability of each room

u = User(popularity, favor, earning_ability)
u.run()
