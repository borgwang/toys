import numpy as np
from agent import Agent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class User(object):
    def __init__(self, uid, enter_prob, stay_prob, tip_prob):
        self.num_host = 5
        self.uid = uid
        self.enter_prob = enter_prob
        self.stay_prob = stay_prob
        self.tip_prob = tip_prob
        self.history = []
        self.recommender = Agent()

        softmax_enter_prob = self.softmax(self.enter_prob)
        self.coefficient = softmax_enter_prob / np.mean(softmax_enter_prob)
        self.running_tip = None
        self.avg_reward = np.mean(-(np.log(1 - np.asarray(enter_prob)) + np.log(1 - np.array(stay_prob))))

        self.global_step = 0

    def run(self):
        total_time = 0
        for i in range(100000):
            rec, room = self.__start()
            tip = 0
            room_time = 0
            while True:
                total_time += 1
                room_time += 1
                tip += self.__tip(room)
                if not self.__is_stay(room):
                    state, reward, next_state = self.__leave_room(rec, room, tip, room_time)
                    self.recommender.update(state, room, reward, next_state, False)
                    self.global_step += 1
                    break

            if self.global_step % 1000 == 0 and self.global_step:
                print self.global_step,
                self.__stat()

    def __start(self):
        rec = self.recommender.act(self.__gen_state(self.history))
        # update enter prob
        if np.random.random() < self.enter_prob[rec]:
            self.enter_prob -= 0.002
            self.enter_prob[rec] += 0.01
        # print self.enter_prob

        room = np.random.choice(np.arange(5), p=self.enter_prob)
        return rec, room

    def __tip(self, room):
        if np.random.random() < (self.tip_prob * self.coefficient[room]):
            return np.random.randint(1, 4)
        else:
            return 0

    def __is_stay(self, room):
        noise = (np.random.random(5)-0.5) * 0.1
        stay_prob = self.stay_prob + noise
        if np.random.random() < stay_prob[room]:
            return True
        else:
            return False

    def __leave_room(self, rec, room, tip, room_time):
        self.rec_history.append(rec)
        state = self.__gen_state(self.history)
        self.history.append([room, tip, room_time])
        next_state = self.__gen_state(self.history)
        reward = self.__cal_reward(rec)
        return state, reward, next_state

    def __random_recommend(self):
        return np.random.choice(np.arange(5), size=3, replace=False)

    def __cal_reward(self, rec):
        reward = -(np.log(1-self.enter_prob[rec]) + np.log(1-self.stay_prob[rec]))
        return reward - self.avg_reward

    def __stat(self):
        history = np.array(self.history)
        rooms = history[-1000:, 0]
        tips = history[-1000:, 1]
        times = history[-1000:, 2]
        if self.running_tip:
            self.running_tip = self.running_tip * 0.99 + np.mean(tips) * 0.01
        else:
            self.running_tip = np.mean(tips)
        print 'avg_tip: ', self.running_tip, ' staytime: ', np.mean(times)

    @staticmethod
    def __gen_state(history):
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


u = User(1, enter_prob=[0.1,0.5,0.2,0.1,0.1], stay_prob=[0.3, 0.8, 0.5, 0.7, 0.4], tip_prob=0.1)
u.run()
