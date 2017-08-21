from __future__ import absolute_import

from nets import G_mlp, D_mlp
from dataset.data import Data
from model import CGAN
import argparse
import os


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', default='./eval_results/')
    parser.add_argument('--save_path', default='./models/')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--train_iters', default=10000)

    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = args_parse()
    if not os.path.exists(args.eval_path):
        os.makedirs(args.eval_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    data = Data()

    generator = G_mlp()
    discriminator = D_mlp()
    cgan = CGAN(generator, discriminator, data, args)
    if args.eval:
        cgan.eval()
    else:
        cgan.train()
