import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sample_z(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])


def sample_y(m, n, idx):
    # return a [mxn] matrix whose idx_th column is one, others are zero.
    y = np.zeros((m, n))
    y[:, idx] = 1.0
    return y


def save_image(samples, fig_count, eval_path):
    sample_size = int(samples.shape[1] ** 0.5)
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(sample_size, sample_size), cmap='Greys_r')

    plt.savefig('{}/{}_{}.png'.format(
        eval_path, str(fig_count).zfill(3),
        str(fig_count % 10)), bbox_inches='tight')
    plt.close(fig)


# TEST
if __name__ == '__main__':
    print sample_y(5, 5, 2)
