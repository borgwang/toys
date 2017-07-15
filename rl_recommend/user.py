from __future__ import absolute_import
from __future__ import division

import scipy.stats as stats
import numpy as np

from helper import Helper
from agent import QAgent


class User(object):
    def __init__(self, uid, num_rooms, popularity, earning_ability, recommender):
        self.num_rooms = num_rooms
        self.__favor = np.zeros(num_rooms)
        self.__visit_stat = {'times': np.zeros(num_rooms),
                             'tips': np.zeros(num_rooms)}
        self.__visit_prob = self.cal_visit_prob(popularity)
        self.__recommender = recommender

        self.popularity = popularity
        self.earning_ability = earning_ability

    @staticmethod
    def cal_visit_prob(pop):
        # The probability of visiting an unvisited room is affected by the
        # room's popularities.
        visit_prob = (pop - np.min(pop)) / (np.max(pop) - np.min(pop))
        visit_prob = visit_prob * 0.7 + 0.1
        return visit_prob

    def run(self):
        while True:
            state = Helper.gen_state(self.__favor)
            rec = self.__recommender.act(state)
            if np.random.random() < self.__visit_prob[rec]:
                room = rec
            else:
                if any(self.__favor):
                    enter_prob = self.__favor / np.sum(self.__favor)
                    non_zero_idx = np.where(enter_prob != 0)[0]
                    non_zero_probs = enter_prob[non_zero_idx]
                    room = np.random.choice(non_zero_idx, p=non_zero_probs)
                else:
                    room = None
            if room is not None:
                tips = self._do_one_visit(room)
                next_state = Helper.gen_state(self.__favor)
                experience = Helper.gen_exp(state, rec, room, tips, next_state)
                self.__recommender.update(experience)

            if all(self.__favor):
                print('All items have been recommended!')
                print('Total steps: %d' % self.__recommender.global_steps)
                Helper.log(self.__favor, self.__visit_stat)
                break

    def _do_one_visit(self, room):
        """
        After visiting, users favor may either increase or decrease.
        """
        self.__visit_stat['times'][room] += 1
        favor = np.random.uniform(-0.5, 0.5)
        tips = 0.0
        if favor > 0.3:
            tips = self.earning_ability[room] * favor
            self.__visit_stat['tips'][room] += tips

        self.__favor[room] += 0.2 * favor
        if self.__favor[room] < 0.0:
            # assign a small positive num to distinguish from unvisited rooms
            self.__favor[room] = 1e-4
        return tips


num_rooms = 100
lower, upper = 0, 5
mu, sigma = 2, 0.5
popularity = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
									loc=mu, scale=sigma, size=num_rooms)
earning_ability = popularity + np.random.normal(scale=0.5, size=num_rooms)


recommender = QAgent(num_rooms)
u = User(1, num_rooms, popularity, earning_ability, recommender)
u.run()
