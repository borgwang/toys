from __future__ import absolute_import

from tensorflow.examples.tutorials.mnist import input_data


class MNIST(object):

    def __init__(self):
        self.X_dim = 784
        self.z_dim = 100
        self.y_dim = 10
        self.data_path = './dataset/mnist'
        self.data = input_data.read_data_sets(self.data_path, one_hot=True)

    def get_batch(self, batch_size):
        return self.data.train.next_batch(batch_size)


# TEST
# if __name__ == '__main__':
#     mnist = MNIST()
#     print mnist.get_batch(3)
