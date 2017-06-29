from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import timeit
import multiprocessing
import os
import signal


class RuntimeWatchException(Exception):
    pass


class TimingTask(object):
    def __init__(self, name):
        self.__name = name
        self.__start_time = timeit.default_timer()

    @property
    def name(self):
        return self.__name

    @property
    def start_time(self):
        return self.__start_time

    @property
    def elapsed_time(self):
        return timeit.default_timer() - self.__start_time


class RuntimeWatcher(object):
    def __init__(self):
        self.__tasks_dict = OrderedDict()

    def start_task(self, name):
        self.__tasks_dict[name] = TimingTask(name)

    def elapsed_time(self, name):
        if name not in self.__tasks_dict:
            raise RuntimeWatchException('Timing task %s not exist' % name)
        return self.__tasks_dict[name].elapsed_time

    def stop_task(self, name):
        if name not in self.__tasks_dict:
            raise RuntimeWatchException('Timing task %s not exist' % name)
        task = self.__tasks_dict.pop(name)
        return task.elapsed_time


class TimeoutException(Exception):
    pass


class LimitTimeCall(object):
    def __init__(self, timeout=-1):
        self.__timeout = timeout if timeout > 0 else None

    def __call__(self, func, *args, **kwargs):
        sub_process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        sub_process.start()
        sub_process.join(self.__timeout)
        if not sub_process.is_alive():
            return
        os.kill(sub_process.ident, signal.SIGALRM)
        sub_process.join()
        raise TimeoutException('func %s call timeout.' % func.__name__)
