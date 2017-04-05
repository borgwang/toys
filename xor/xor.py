import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

hidden_dim = 10
input_data = tf.placeholder(tf.float32, [None, 2], name='input')
label = tf.placeholder(tf.float32, [None, 1], name='ground_true')
w_init = tf.contrib.layers.variance_scaling_initializer(
        factor=2.0, mode='FAN_IN', uniform=True)
b_init = tf.zeros_initializer()
w1 = tf.get_variable('w1', [2, hidden_dim], initializer=w_init)
b1 = tf.get_variable('b1', [hidden_dim], initializer=b_init)
h1 = tf.nn.relu(tf.matmul(input_data, w1) + b1)

w2 = tf.get_variable('w2', [hidden_dim, 1], initializer=w_init)
b2 = tf.get_variable('b2', [1], initializer=b_init)

predict = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

loss = tf.losses.mean_squared_error(label, predict)
train_op = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train!
running_loss = -1
for step in range(10000):
    idx = np.random.randint(len(X))
    l, _ = sess.run([loss, train_op],{
        input_data: X[idx][np.newaxis,:],
        label: y[idx][np.newaxis,:]
    })
    if running_loss < 0:
        running_loss = l
    else:
        running_loss = 0.99 * running_loss + 0.01 * l
    if step % 1000 == 0:
        print 'Step: %d, runnint loss: %.6f' % (step, running_loss)

X_test = []
r = np.arange(0,1,0.01)
for x in r:
    for y in r:
        X_test.append([x,y])
X_test = np.array(X_test)
y_test = sess.run(predict, {input_data: X_test})

ax=subplot(111,projection='3d')
ax.scatter(X_test[:,0],X_test[:,1],y_test.ravel(),c='b',s=5,edgecolors='face')

x,y,z = [0,1,0,1],[0,1,1,0],[0,0,1,1]

ax.scatter(x[:2],y[:2],z[:2],c='r')
ax.scatter(x[2:],y[2:],z[2:],c='g')

plt.show()
