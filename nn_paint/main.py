from PIL import Image
import numpy as np
import tensorflow as tf
img = np.asarray(Image.open('./res/origin.jpg'), dtype='float32')
img /= 255.0
# construct training data
h, w = img.shape[0], img.shape[1]
X, y = [], []
for i in range(h):
    for j in range(w):
        X.append([(i-h/2.0)/h, (j-w/2.0)/w])
        y.append(img[i][j])
X = np.array(X); y = np.array(y)
data_len = len(X)

# construct network
hidden_dim = 30
input_data = tf.placeholder(tf.float32, [None, 2], name='input')
label = tf.placeholder(tf.float32, [None, 3], name='ground_true')
w_init = tf.contrib.layers.variance_scaling_initializer(
        factor=2.0, mode='FAN_IN', uniform=True)
b_init = tf.zeros_initializer()
w1 = tf.get_variable('w1', [2, hidden_dim], initializer=w_init)
b1 = tf.get_variable('b1', [hidden_dim], initializer=b_init)
h1 = tf.nn.relu(tf.matmul(input_data, w1) + b1)

w2 = tf.get_variable('w2', [hidden_dim, hidden_dim], initializer=w_init)
b2 = tf.get_variable('b2', [hidden_dim], initializer=b_init)
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = tf.get_variable('w3', [hidden_dim, hidden_dim], initializer=w_init)
b3 = tf.get_variable('b3', [hidden_dim], initializer=b_init)
h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

w4 = tf.get_variable('w4', [hidden_dim, hidden_dim], initializer=w_init)
b4 = tf.get_variable('b4', [hidden_dim], initializer=b_init)
h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)

w5 = tf.get_variable('w5', [hidden_dim, hidden_dim], initializer=w_init)
b5 = tf.get_variable('b5', [hidden_dim], initializer=b_init)
h5 = tf.nn.relu(tf.matmul(h4, w5) + b5)

w6 = tf.get_variable('w6', [hidden_dim, hidden_dim], initializer=w_init)
b6 = tf.get_variable('b6', [hidden_dim], initializer=b_init)
h6 = tf.nn.relu(tf.matmul(h5, w6) + b6)

w7 = tf.get_variable('w7', [hidden_dim, 3], initializer=w_init)
b7 = tf.get_variable('b7', [3], initializer=b_init)
predict = tf.nn.sigmoid(tf.matmul(h6, w7) + b7)


loss = tf.losses.mean_squared_error(label, predict)
train_op = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train!
batch = 8
running_loss = -1
for step in range(10000000):
    idx = [np.random.randint(data_len) for _ in range(batch)]
    l, _ = sess.run([loss, train_op],{
        input_data: X[idx],
        label: y[idx]
    })
    if running_loss < 0:
        running_loss = l
    else:
        running_loss = 0.999 * running_loss + 0.001 * l
    if step % 10000 == 0:
        print 'Step: %d, runnint loss: %.6f' % (step, running_loss)
        paint = sess.run(predict, {input_data: X}).reshape(h, w, -1)
        paint = (paint * 255.0).astype('uint8')
        Image.fromarray(paint).save('./res/paint.jpg')
