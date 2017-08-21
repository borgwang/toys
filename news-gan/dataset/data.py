# author: wangguibo [wangguibo@interns.chuangxin.com]
# date: 2017-08-19
#
# filename: data.py
# description: Data Class. Process and analyse data.

from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


class Data(object):
    def __init__(self):
        self.X_dim = 50
        self.z_dim = 10
        self.load_data()

    def get_batch(self, batch_size=1):
        idx = np.random.choice(range(self.data_size), size=batch_size)
        return self.data[idx]

    def load_data(self):
        try:
            self.history_dict = pickle.load(open('./dataset/history.p', 'rb'))
            self.item_dict = OrderedDict(
                pickle.load(open('./dataset/item_dict.p', 'rb')))
        except Exception:
            raise

        data = []
        for h in self.history_dict:
            data.append(self.item_dict[h]['vector'])

        self.data = np.array(data)
        self.data_size = len(self.data)
        self.mean_vector = np.mean(self.data, 0)
        print('%d data loaded' % self.data_size)

    def analyse(self, vec):
        # ------ top k items compare ------
        k = 10
        real_topk = np.argsort(self.mean_vector)[-k:]
        gen_topk = np.argsort(vec, 1)[:, -k:]
        hit_list = []
        for items in gen_topk:
            hit_list.append(len([i for i in items if i in real_topk]))

        hit_rate = np.mean(hit_list) / k
        similarity = self.get_dist(vec, self.mean_vector)

        for i in range(10):
            d = vec[np.random.choice(range(len(vec)))]
            plt.plot(range(50), d, 'g-', label=str(i))
        plt.plot(range(50), self.mean_vector, 'b', label='ground true')
        plt.legend()
        plt.show()
        return hit_rate, similarity

    @staticmethod
    def get_dist(v1, v2):
        # return cosine similarity
        # kl_divergence = (
        #     -np.sum(v1 * np.log(v2)) + np.sum(v1 * np.log(v1))
        #     + np.sum(-v2 * np.log(v1)) + np.sum(v1 * np.log(v1))) * 0.5
        cosine_similarity = np.mean(
            np.sum(v1 * v2, 1) / ((np.sum(v1**2, 1) * np.sum(v2**2)) ** 0.5))

        return cosine_similarity


# ------------
# TEST
# ------------
# if __name__ == '__main__':
#     data = Data()
#     a, b = data.get_batch(16)
