import tensorflow as tf


def leaky_relu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


class G_mlp(object):

    def __init__(self):
        self.X_dim = 50
        self.name = 'G_mlp'
        self.hidden_units = 128

    def __call__(self, z):
        with tf.variable_scope(self.name) as sp:
            h1 = tf.contrib.layers.fully_connected(
                z, self.hidden_units, activation_fn=leaky_relu,
                weights_initializer=tf.random_normal_initializer(0, 0.02))

            out = tf.contrib.layers.fully_connected(
                h1, self.X_dim, activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(0, 0.02))

            return out

    @property
    def vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class D_mlp(object):

    def __init__(self):
        self.name = 'D_mlp'
        self.hidden_units = 128

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            h1 = tf.contrib.layers.fully_connected(
                x, self.hidden_units, activation_fn=leaky_relu,
                weights_initializer=tf.random_normal_initializer(0, 0.02))

            out = tf.contrib.layers.fully_connected(
                h1, 1, activation_fn=None,
                weights_initializer=tf.random_normal_initializer(0, 0.02))

            return out

    @property
    def vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
