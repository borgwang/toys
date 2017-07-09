import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import deque
import random


class RLAgent(object):
    def __init__(self):
        self.state_dim = 15
        self.action_dim = 5
        self.epsilon = 0.5
        self.final_epsilon = 0.01
        self.delta_epsilon = (self.epsilon - self.final_epsilon) / 5e5
        self.construct_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 16
        self.global_step = 0
        self.target_network_update_interval = 1000

    def q_nework(self, state_input):
        hidden_units = 256
        h1 = slim.fully_connected(inputs=state_input, num_outputs=hidden_units,
                                  activation_fn=tf.nn.relu, scope='h1')
        output_q = slim.fully_connected(inputs=h1, num_outputs=self.action_dim,
                                      activation_fn=None, scope='output')
        return output_q

    def construct_model(self):
        with tf.name_scope('model_input'):
            self.state_input = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)

        with tf.variable_scope('q_network'):
            self.output_q = self.q_nework(self.state_input)

        with tf.name_scope('train'):
            self.action_input = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32)
            action_q = tf.reduce_sum(tf.multiply(self.output_q, self.action_input), 1)
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(action_q, self.target_q)
            self.train = tf.train.RMSPropOptimizer(learning_rate=0.003, decay=0.9).minimize(self.loss)

        with tf.variable_scope('target_q_network'):
            self.target_output_q = self.q_nework(self.state_input)

        q_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
        target_q_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q_network')

        with tf.name_scope('update_target_network'):
            update_list = []
            for source, target in zip(q_params, target_q_params):
                update_op = target.assign(source)
                update_list.append(update_op)
            self.update_target_network = tf.group(*update_list)

    def act(self, state):
        self.global_step += 1
        # epsilon greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = np.ravel(state).reshape(1,-1)
            output_q = self.sess.run(self.output_q, {self.state_input: state})[0]
            return np.argmax(output_q)

    def random_act(self, state):
        return np.random.randint(self.action_dim)

    def update(self, state, action, reward, next_state, done):
        onehot_action = np.zeros(self.action_dim)
        onehot_action[action] = 1
        state = np.ravel(state)
        next_state = np.ravel(next_state)

        self.replay_buffer.append([state, onehot_action, reward, next_state, done])
        if len(self.replay_buffer) > self.batch_size:
            self._update_model()

    def _update_model(self):
        if self.global_step % self.target_network_update_interval == 0:
            self.sess.run(self.update_target_network)

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        b_s, b_a, b_r, b_ns, b_d = np.asarray(minibatch).T.tolist()
        next_state_all_q = self.sess.run(self.target_output_q, {self.state_input: b_ns})
        # next_state_max_q = np.max(next_state_all_q, 1)

        # Double Q-learning (https://arxiv.org/pdf/1509.06461.pdf)
        next_state_action = np.argmax(self.sess.run(self.output_q, {self.state_input: b_ns}), 1)
        next_state_max_q = next_state_all_q[np.arange(self.batch_size), next_state_action]

        b_target_q = b_r + 0.99 * next_state_max_q

        self.sess.run(self.train, {
            self.state_input: b_s,
            self.action_input: b_a,
            self.target_q: b_target_q})

    def decay_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.delta_epsilon
