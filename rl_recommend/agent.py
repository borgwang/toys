from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)


class RLAgent(object):
    def __init__(self, num_rooms):
        self.state_dim = num_rooms
        self.action_dim = num_rooms
        self._global_steps = 0

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def construct_model(self):
        raise NotImplementedError()

    def act(self, state):
        raise NotImplementedError()

    def random_act(self, state):
        return np.random.randint(self.action_dim)

    def update_model(self):
        raise NotImplementedError()

    @property
    def global_steps(self):
        return self._global_steps


class QAgent(RLAgent):
    def __init__(self, num_rooms):
        super(QAgent, self).__init__(num_rooms)
        self.epsilon = 0.5
        self.final_epsilon = 0.01
        self.delta_epsilon = (self.epsilon - self.final_epsilon) / 5e5
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 16
        self.target_network_update_interval = 5000
        self.learning_rate = 0.003
        self.decay = 0.9
        self.construct_model()
        self.init_session()

    def q_network(self, state_input):
        hidden_units = 256
        h1 = tf.contrib.slim.fully_connected(
            state_input, hidden_units, activation_fn=tf.nn.relu, scope='h1')
        output_q = tf.contrib.slim.fully_connected(
            h1, self.action_dim, activation_fn=None, scope='output')
        return output_q

    def construct_model(self):
        with tf.name_scope('model_input'):
            self.state_input = tf.placeholder(
                shape=[None, self.state_dim], dtype=tf.float32)

        with tf.variable_scope('q_network'):
            self.output_q = self.q_network(self.state_input)

        with tf.name_scope('train'):
            self.action_input = tf.placeholder(
                tf.float32, [None, self.action_dim])
            action_q = tf.reduce_sum(tf.multiply(
                self.output_q, self.action_input), 1)
            self.target_q = tf.placeholder(tf.float32, [None])
            self.loss = tf.losses.mean_squared_error(action_q, self.target_q)
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate, decay=self.decay)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.variable_scope('target_q_network'):
            self.target_output_q = self.q_network(self.state_input)

        q_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
        target_q_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q_network')

        with tf.name_scope('update_target_network'):
            update_list = []
            for source, target in zip(q_params, target_q_params):
                update_op = target.assign(source)
                update_list.append(update_op)
            self.update_target_network = tf.group(*update_list)

    def act(self, state):
        self._global_steps += 1
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = np.ravel(state).reshape(1,-1)
            output_q = self.sess.run(
                self.output_q,
                {self.state_input: state})[0]
            return np.argmax(output_q)

    def update(self, experience):
        onehot_action = np.zeros(self.action_dim)
        onehot_action[experience['action']] = 1
        state = np.ravel(experience['state'])
        next_state = np.ravel(experience['next_state'])
        reward = experience['reward']
        done = experience['done']

        self.replay_buffer.append(
            [state, onehot_action, reward, next_state, done])
        if len(self.replay_buffer) > self.batch_size:
            self._update_model()

    def _update_model(self):
        if self._global_steps % self.target_network_update_interval == 0:
            self.sess.run(self.update_target_network)

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        b_s, b_a, b_r, b_s2, b_d = np.asarray(minibatch).T.tolist()
        s2_all_q = self.sess.run(self.target_output_q, {self.state_input: b_s2})
        # s2_max_q = np.max(s2_all_q, 1)

        # Double Q-learning (https://arxiv.org/pdf/1509.06461.pdf)
        s2_action = np.argmax(self.sess.run(
            self.output_q, {self.state_input: b_s2}), 1)
        s2_max_q = s2_all_q[np.arange(self.batch_size), s2_action]

        b_target_q = b_r + 0.99 * s2_max_q

        feed_dict = {self.state_input: b_s,
                     self.action_input: b_a,
                     self.target_q: b_target_q}

        self.sess.run(self.train_op, feed_dict)

    def decay_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.delta_epsilon


class A3CAgent(RLAgent):
    def __init__(self, num_rooms):
        super(A3CAgent, self).__init__(num_rooms)
