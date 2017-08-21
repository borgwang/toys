# author: wangguibo [wangguibo@interns.chuangxin.com]
# date: 2017-08-17
#
# filename: cgan.py
# description: CGAN Class.

from __future__ import absolute_import
from __future__ import division

from utils import sample_z
import tensorflow as tf


class CGAN(object):

    def __init__(self, generator, discriminator, data, args):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.args = args

        self._construct_model()

    def _construct_model(self):
        self.z_dim = self.data.z_dim    # noisy input
        self.X_dim = self.data.X_dim    # ground true

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])

        # nets
        self.G_sample = self.generator(self.z)
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse=True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_real, labels=tf.ones_like(self.D_real))) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # optimize
        self.train_D = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(
            self.D_loss, var_list=self.discriminator.vars)
        self.train_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(
            self.G_loss, var_list=self.generator.vars)

    def _init_job(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session()
        if self.args.eval:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(self.args.save_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('Initialize a new model')
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        self._init_job()
        for i in range(self.args.train_iters + 1):
            # update D
            X_b = self.data.get_batch(self.args.batch_size)
            z_b = sample_z(self.args.batch_size, self.z_dim)
            self.sess.run(self.train_D, feed_dict={self.X: X_b, self.z: z_b})
            # update G
            self.sess.run(self.train_G, feed_dict={self.z: z_b})
            # log
            if i % 500 == 0:
                d_loss = self.sess.run(self.D_loss, {
                    self.X: X_b,
                    self.z: z_b})
                g_loss = self.sess.run(self.G_loss, {
                    self.z: z_b})
                print('steps%d d_loss: %.4f, g_loss: %.4f' %
                      (i, d_loss, g_loss))

            # save model
            if i and i % 5000 == 0:
                self.saver.save(self.sess, self.args.save_path + str(i))

    def eval(self):
        self._init_job()
        eval_batch = 1000
        z_b = sample_z(eval_batch, self.z_dim)
        samples = self.sess.run(self.G_sample, feed_dict={self.z: z_b})
        print(self.data.analyse(samples))
