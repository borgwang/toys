from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gflags

from base.registry import Register
from concurrent.futures import ProcessPoolExecutor


gflags.DEFINE_string('evaluate_worker_nums', 4, 'The nums of evaluate worker')

FLAGS = gflags.FLAGS

class EvaluateException(Exception):
    pass


class EvaluatorManager(object):
    _manager = None
    _mutex = threading.RLock()

    def __init__(self):
        self.__worker_pool = ProcessPoolExecutor(max_workers=FLAGS.evaluate_worker_nums)

    @staticmethod
    def get_manager():
        if EvaluatorManager._manager is None:
            with EvaluatorManager._mutex:
                if EvaluatorManager._manager is None:
                    EvaluatorManager._manager = EvaluatorManager()
        return EvaluatorManager._manager
