from __future__ import division

import numpy as np
from rl_agent import RLAgent


class User(object):
    def __init__(self, popularity, favor, earning_ability):
        self._init_user(popularity, favor, earning_ability)
        self.recommender = RLAgent()
        self.history = []
        self.rec_history = []
        self.running_tip = None
        self.counter = self.popularity * 500

    def _init_user(self, popularity, favor, earning_ability):
        '''
        - enter_prob: affected by [room popularity] and [user favor].
        - tip_prob: affected by [the earning_ability of a romm] and [user favor].
        - stay_prob: determined by user favor.

        The weights of these factors suppose to be different for different users.
        '''
        self.popularity = popularity
        self.favor = favor

        self.enter_prob = 0.5 * popularity + 0.5 * favor
        self.tip_prob = 0.5 * earning_ability + 0.5 * favor
        self.stay_prob = favor

        print 'enter_prob: ', self.enter_prob
        print 'tip_prob: ', self.tip_prob
        print 'stay_prob: ', self.stay_prob
        print

    def run(self):
        for step in range(500000):
            tips, stay_time = 0, 0
            rec, room = self._start()
            while self._is_stay(room):
                stay_time += 1
                tips += self._tip(room)

            state, next_state = self._leave(room, tips, stay_time)
            reward = self._gen_reward(rec, room)
            self.recommender.update(state, rec, reward, next_state, False)

            if step and step % 1000 == 0:
                self._log(step)

    def _start(self):
        rec = self.recommender.act(self._gen_state())
        self.rec_history.append(rec)
        room = np.random.choice(range(5), p=self.enter_prob)
        if room == rec:
            self._update_user(rec)

        return rec, room

    def _is_stay(self, room):
        return True if np.random.random() < self.stay_prob[room] else False

    def _tip(self, room):
        return 1 if np.random.random() < self.tip_prob[room] else 0

    def _leave(self, room, tips, stay_time):
        state = self._gen_state()
        self.history.append([room, tips, stay_time])
        next_state = self._gen_state()
        return state, next_state

    def _gen_reward(self, rec, room):
        if rec == room:
            # return -(0.3 * np.log(1 - self.popularity[rec]) + 0.7 * np.log(1 - self.favor[rec]))
            return -np.log(1 - self.favor[rec])
        else:
            return 0

    def _log(self, step):
        history = np.asarray(self.history)
        rooms = history[-1000:, 0]
        tips = history[-1000:, 1]
        times = history[-1000:, 2]
        if self.running_tip:
            self.running_tip = 0.99 * self.running_tip + 0.01 * np.mean(tips)
        else:
            self.running_tip = np.mean(tips)
        print 'step%d avg_tip: %.4f stay_time: %.2f ' % \
                (step, self.running_tip, np.mean(times))

        if step % 5000 == 0:
            print '-----'
            print 'enter_prob: ', self.enter_prob
            print 'recommend stats: ',
            for i in range(5):
                print '%.1f%% ' % (100 * self.rec_history[-5000:].count(i) / 5000),
            print
            print '-----'

    def _gen_state(self):
        state = np.array(self.history[-5:])
        if len(state) == 0:
            return np.zeros((5, 3))
        elif len(state) < 5:
            return np.vstack((np.zeros([5 - len(state), 3]), state))
        else:
            return state

    def _update_user(self, rec):
        '''
        Model how recommendation affect users' subsequence behaviors
        recommendation hits -> affect popularity -> affect enter_prob

        Note: hitting recommendations should not change [user favor].
        '''
        self.counter[rec] += 1
        self.popularity = self.counter / np.sum(self.counter)
        self.enter_prob = 0.5 * self.popularity + 0.5 * self.favor


if __name__ == '__main__':

    popularity = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
    favor = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
    earning_ability = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    # enter_prob:  [ 0.25   0.25   0.25   0.175  0.075]
    # tip_prob:  [ 0.45  0.2   0.1   0.15  0.1 ]
    # stay_prob:  [ 0.4  0.2  0.1  0.2  0.1]

    User(popularity, favor, earning_ability).run()
