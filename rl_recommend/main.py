from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import argparse
import time
import itertools

from user import User
from agent import QAgent


def main(args):
    recommender = QAgent(args.num_rooms)
    global_steps_counter = itertools.count()
    num_users = args.num_users
    users = []

    # create users
    for i in range(num_users):
        u = User(i, args.num_rooms)
        users.append(u)



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', default=10, type=int,
        help='Numbers of users to simulate.')
    parser.add_argument('--num_rooms', default=1000, type=int,
        help='Numbers of rooms in the system.')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
