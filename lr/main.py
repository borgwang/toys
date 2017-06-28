from __future__ import division
import csv
import numpy as np
import random
import tensorflow as tf

# data prepare

X, y = [], []

with open('./data/credit_card_dataset.csv') as f:
    f_csv = csv.reader(f)
    header = next(f_csv)
    for row in f_csv:
        X.append([float(f) for f in row[1:-1]])
        y.append(int(row[-1]))

part_train = 0.6
part_test = 0.2
part_val = 0.2
dim_feature = len(header) - 2
total_num_data = len(X)

num_train = int(part_train * total_num_data)
num_val = int(part_val * total_num_data)
num_test = int(part_test * total_num_data)
X = np.asarray(X)
y = np.vstack(y)

train_X = X[:num_train]  # (16000, 23)
train_y = y[:num_train]

val_X = X[num_train:num_train+num_val]
val_y = y[num_train:num_train+num_val]

test_X = X[num_train+num_val:]
test_y = y[num_train+num_val:]


input_X = tf.placeholder(tf.float32, shape=[None, dim_feature])
labels = tf.placeholder(tf.float32, shape=[None, 1])

# w1 = tf.Variable(tf.truncated_normal([dim_feature, dim_h1],stddev=1.0 / np.sqrt(dim_feature)),name="w1")
w1 = tf.get_variable(name='w1', shape=[dim_feature, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(0.01 * tf.ones(shape=[1]), dtype=tf.float32)

prob = tf.nn.sigmoid(tf.matmul(input_X, w1) + b1)
pred = tf.round(prob)

# metrics
# accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
tp = tf.count_nonzero(pred * labels)
tn = tf.count_nonzero((1-pred) * (1-labels))
fp = tf.count_nonzero(pred * (1-labels))
fn = tf.count_nonzero((1-pred) * labels)
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)

# optimize
loss = tf.losses.sigmoid_cross_entropy(labels, prob)
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

num_iters = 10000
batch = 3000
# running_loss = None

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1, num_iters+1):
    idx = np.random.choice(range(num_train), size=batch)
    t_X = train_X[idx]
    t_y = train_y[idx]
    l, _ = sess.run([loss, train_op], {input_X: t_X, labels: t_y})
    # if running_loss:
    #     running_loss = running_loss * 0.99 + l * 0.01
    # else:
    #     running_loss = l

    # if i % 500 == 0:
    #     print 'iter: %d running_loss: %.4f' % (i, running_loss)

    # eval
    if i % 1000 == 0:
        eval_loss, eval_acc = sess.run(
            [loss, accuracy], {input_X: val_X, labels: val_y})
        etp,etn,efp,efn = sess.run([tp,tn,fp,fn], {input_X: val_X, labels: val_y})
        print '---- Eval at step %d ----' % i
        print 'Loss: %.4f, accuracy: %.4f' % (eval_loss, eval_acc)
        print '[tp,tn,fp,fn]',etp,etn,efp,efn
