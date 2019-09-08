import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

NUM_ITER = 200


def get_loss(x, y, seed=0):
    def gaussian_hills(x, y, x_mean, y_mean, x_sig, y_sig):
        normalizing = 1 / (2 * np.pi * x_sig * y_sig)
        x_exp = (-1 * (x - x_mean) ** 2) / (2 * x_sig ** 2)
        y_exp = (-1 * (y - y_mean) ** 2) / (2 * y_sig ** 2)
        return normalizing * tf.exp(x_exp + y_exp)
    loss = 0
    n_hills = 10
    np.random.seed(seed)
    weights = np.random.uniform(-0.2, 0.05, n_hills)
    x_means = np.random.uniform(-1., 1., n_hills)
    y_means = np.random.uniform(-1., 1., n_hills)
    x_sigmas = np.random.uniform(0.2, 0.8, n_hills)
    y_sigmas = np.random.uniform(0.2, 0.8, n_hills)
    for i in range(n_hills):
        loss += weights[i] * gaussian_hills(x, y, x_means[i], y_means[i], x_sigmas[i], y_sigmas[i])
    return loss


def build_graph(seed, start_x=0.0, start_y=0.0):
    tf.reset_default_graph()
    x_var = tf.get_variable(name="x", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(start_x))
    y_var = tf.get_variable(name="y", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(start_y))
    loss = get_loss(x_var, y_var, seed)
    adam_opt = tf.train.AdamOptimizer(1e-1).minimize(
        loss, var_list=[x_var, y_var], name="Adam")
    rmsprop_opt = tf.train.RMSPropOptimizer(1e-1).minimize(
        loss, var_list=[x_var, y_var], name="RMSProp")
    momentum_opt = tf.train.MomentumOptimizer(1e-1, momentum=0.9).minimize(
        loss, var_list=[x_var, y_var], name="Momentum")
    nag_opt = tf.train.MomentumOptimizer(1e-1, momentum=0.9, use_nesterov=True).minimize(
        loss, var_list=[x_var, y_var], name="NAG")
    adagrad_opt = tf.train.AdagradOptimizer(1e-1).minimize(
        loss, var_list=[x_var, y_var], name="Adagrad")
    adadelta_opt = tf.train.AdadeltaOptimizer(1e-1).minimize(
        loss, var_list=[x_var, y_var], name="Adadelta")
    sgd_opt = tf.train.GradientDescentOptimizer(1e-1).minimize(
        loss, var_list=[x_var, y_var], name="GradientDecent")

    colors = "bgrcmykw"
    optimizers = list()
    for i, opt in enumerate([adam_opt, rmsprop_opt, momentum_opt, adagrad_opt, adadelta_opt, sgd_opt, nag_opt]):
        optimizers.append({"opt": opt, "name": opt.name, "lr": 0.1, "trajectory": list(), "color": colors[i]})
    return x_var, y_var, loss, optimizers


def plot_surface(seed):
    s = 50
    x_lb, x_ub = -1.5, 1.5
    y_lb, y_ub = -1.5, 1.5

    x = np.arange(x_lb, x_ub, (x_ub - x_lb) / s)
    y = np.arange(y_lb, y_ub, (y_ub - y_lb) / s)
    x, y = np.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)

    tf.reset_default_graph()
    with tf.Session() as sess:
        loss = sess.run(get_loss(x, y, seed))
    x = x.reshape((s, s))
    y = y.reshape((s, s))
    loss = loss.reshape((s, s))

    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    ax._axis3don = False
    ax.plot_surface(x, y, loss, rstride=1, cstride=1, cmap=plt.get_cmap("coolwarm"), alpha=0.6, zorder=0)
    r = loss.max() - loss.min()
    ax.set_zlim(loss.min() - 0.8 * r, loss.max() + 0.8 * r)
    ax.set_xlim(x_lb, x_ub)
    ax.set_ylim(y_lb, y_ub)
    ax.view_init(elev=45, azim=45)
    return fig, ax


def get_peak(seed):
    s = 50
    x_lb, x_ub = -1.5, 1.5
    y_lb, y_ub = -1.5, 1.5

    x = np.arange(x_lb, x_ub, (x_ub - x_lb) / s)
    y = np.arange(y_lb, y_ub, (y_ub - y_lb) / s)
    x, y = np.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)

    tf.reset_default_graph()
    with tf.Session() as sess:
        loss = sess.run(get_loss(x, y, seed))
    idx = np.argmax(loss)
    return x[idx], y[idx]


def plot_trajectories(fig, ax, optimizers):
    def update_fig(i):
        global scatters
        for j, opt in enumerate(optimizers):
            if scatters[j] is not None:
                scatters[j].remove()
            scat = ax.scatter(*zip(opt["trajectory"][i + 1]), zorder=10, linewidth=4, color=opt["color"], alpha=1.0)
            scatters[j] = scat
            ln = ax.plot(*zip(opt["trajectory"][i], opt["trajectory"][i + 1]), color=opt["color"], zorder=10,
                         label=opt["name"], linewidth=2, alpha=0.8)
        return ln,

    def init_fig():
        for opt in optimizers:
            ln = ax.scatter(*opt["trajectory"][0], color=opt["color"], zorder=10,
                            label="%s (%.2f)" % (opt["name"], opt["lr"]))
        ax.legend(prop={'size': 15})
        return ()

    global scatters
    scatters = [None for _ in range(len(optimizers))]
    anim = FuncAnimation(fig, update_fig, frames=NUM_ITER - 1, interval=60, init_func=init_fig, repeat=True)
    return anim


def run(seeds):
    for seed in seeds:
        start_x, start_y = get_peak(seed)
        x_var, y_var, loss_var, optimizers = build_graph(seed, start_x, start_y)
        with tf.Session() as sess:
            for opt in optimizers:
                sess.run(tf.global_variables_initializer())
                opt["trajectory"].append(sess.run([x_var, y_var, loss_var]))
                for _ in range(NUM_ITER):
                    sess.run(opt["opt"])
                    opt["trajectory"].append(sess.run([x_var, y_var, loss_var]))

        # plot loss curves
        plt.figure(figsize=(8, 6))
        for opt in optimizers:
            plt.plot(np.asarray(opt["trajectory"])[:, 2], label=opt["name"])
        plt.legend(prop={"size": 12})
        plt.savefig("./results/loss-curve-%s.png" % seed)
        plt.close()

        # plot trajectories
        fig, ax = plot_surface(seed)
        anim = plot_trajectories(fig, ax, optimizers)
        anim.save('./results/trajectory-%s.mp4' % seed, dpi=60, writer='imagemagick')
        plt.close()

        print("seed %s done" % seed)


if __name__ == "__main__":
    seeds = np.power(2, np.arange(16))
    run(seeds)
