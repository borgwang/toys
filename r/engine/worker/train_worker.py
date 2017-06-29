from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gflags

from .train_helper import TrainHelper
from base.runtime_watcher import LimitTimeCall
from engine.worker.base_worker import BaseWorker
from engine.job_queue.redis_job_queue_client import RedisJobQueueClient
from common_proto.train.train_job_pb2 import TrainJob
from base.runtime_watcher import LimitTimeCall

FLAGS = gflags.FLAGS

class TrainWorker(BaseWorker):
    def __init__(self, worker_name):
        super(TrainWorker, self).__init__(worker_name)
        self._job_queue_name = FLAGS.train_job_queue
        self._job_cls = TrainJob
        self._job_queue_client = RedisJobQueueClient(FLAGS.redis_url, self._job_queue_name, self._job_cls)
        self.__info_log_template = 'status=%s\tcost=%.4f'

    def _do_job_internal(self, job_id, job):
        self._runtime_watcher.start_task('do_train_job')
        train_helper = TrainHelper(self._logger)

        timeout = (job.learning_schema.max_train_time if job.learning_schema.max_train_time > 0 else FLAGS.max_train_time)
        limit_time_caller = LimitTimeCall(timeout)

        try:
            limit_time_caller(func=train_helper.do_train_job, job_id=job_id, job=job)
        finally:
            train_cost = self._runtime_watcher.stop_task('do_train_job')
            self._logger.info(self.__info_log_template % ('train_job_finish', train_cost))



    def registry(self):
        self._running = True
