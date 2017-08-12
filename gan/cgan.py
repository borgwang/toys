from __future__ import absolute_import

from nets import G_mlp, D_mlp
from dataset.data import MNIST
from utils import *

import tensorflow as tf
import numpy as np
import argparse
import os


class CGAN(object):

    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self._construct_model()

    def _construct_model(self):
        self.z_dim = self.data.z_dim    # noisy input
        self.y_dim = self.data.y_dim    # condition
        self.X_dim = self.data.X_dim    # ground true

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.y_dim])

        # nets
        self.G_sample = self.generator(tf.concat((self.z, self.y), 1))
        self.D_real = self.discriminator(tf.concat((self.X, self.y), 1))
        self.D_fake = self.discriminator(
            tf.concat((self.G_sample, self.y), 1), reuse=True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_real, labels=tf.ones_like(self.D_real))) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # optimize
        self.train_D = tf.train.AdamOptimizer().minimize(
            self.D_loss, var_list=self.discriminator.vars)
        self.train_G = tf.train.AdamOptimizer().minimize(
            self.G_loss, var_list=self.generator.vars)

    def _init_job(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, args):
        self._init_job()
        fig_count = 0
        for i in range(args.train_iters):
            # update D
            X_b, y_b = self.data.get_batch(args.batch_size)
            self.sess.run(self.train_D, feed_dict={
                self.X: X_b,
                self.y: y_b,
                self.z: sample_z(args.batch_size, self.z_dim)})
            # update G
            self.sess.run(self.train_G, feed_dict={
                self.y: y_b,
                self.z: sample_z(args.batch_size, self.z_dim)})

            # log
            if i % 100 == 0:
                d_loss = self.sess.run(self.D_loss, feed_dict={
                    self.X: X_b,
                    self.y: y_b,
                    self.z: sample_z(args.batch_size, self.z_dim)})
                g_loss = self.sess.run(self.G_loss, feed_dict={
                    self.y: y_b,
                    self.z: sample_z(args.batch_size, self.z_dim)})

                print('steps%d d_loss: %.4f, g_loss: %.4f' %
                      (i, d_loss, g_loss))

            # evaluate
            if i % 1000 == 0:
                y_s = sample_y(16, self.y_dim, fig_count % 10)
                samples = self.sess.run(self.G_sample, feed_dict={
                    self.y: y_s,
                    self.z: sample_z(16, self.z_dim)})

                save_image(samples, fig_count, args.eval_path)
                fig_count += 1

            # save model
            # if i % 2000 == 0:
            #     self.saver.save(self.sess, args.save_path + str(i) + '_cgan')


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', default='./eval/mnist_cgan_mlp/')
    parser.add_argument('--save_path', default='./models/')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--train_iters', default=100000)

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    if not os.path.exists(args.eval_path):
        os.makedirs(args.eval_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    data = MNIST()

    cgan = CGAN(G_mlp(), D_mlp(), data)
    cgan.train(args)
