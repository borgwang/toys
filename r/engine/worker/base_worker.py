from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gflags

from base.logger.logger_helper import LoggerHelper
from base.runtime_watcher import RuntimeWatcher

gflags.DEFINE_string('redis_url', 'redis://root:chuangxingongchang@10.18.125.15:8420', 'the path of redis queue')

FLAGS = gflags.FLAGS


class BaseWorker(object):
    def __init__(self, worker_name):
        self._worker_name = worker_name
        self._runtime_watcher = RuntimeWatcher()
        self._running = False
        self._job_queue_name = None
        self._job_cls = None
        self._job_queue_client = BaseJobQueueClient(FLAGS.redis_url, self._job_queue_name, self._job_cls)
        self._status_register = None
        self._logger = LoggerHelper.init_logger(self._worker_name)
        self.__log_info_template = 'job_id=%s\tcost=%.4f'

    def registry(self):
        raise NotImplementedError('Must use specify worker')

    def do_work_loop(self):
        while self._running:
            job_id, job = self._job_queue_client.fetch_job()
            self._runtime_watcher.start_task('do_job')
            log_handler = None
            try:
                log_handler = self._init_job_log_handler(job_id)
                job_result = self._do_job_internal(job_id, job)
                self._finish_job(job_id, job_result)
                cost = self._runtime_watcher.stop_task('do_job')
                self._logger.info(self.__log_info_template % (job_id, cost))
            except Exception as e:
                cost = self._runtime_watcher.stop_task('do_job')
                err_log = self.__log_info_template % (job_id, cost)
                err_log += '\terr=%s' % str(e)
                self._logger.error(err_log)
            finally:
                self._remove_job_log_handler(log_handler)

    def _init_job_log_handler(self, job_id):
        log_dir = '/'.join([FLAGS.base_dir, job_id, 'log'])
        log_file_name = '.'.join([self._worker_name, 'log'])
        log_handler = LoggerHelper.create_file_handler(log_dir, log_file_name)
        LoggerHelper.set_handler(self._logger, log_handler)
        return log_handler

    def _remove_job_log_handler(self, handler):
        if not handler:
            return
        LoggerHelper.remove_handler(handler)

    def _init_job_log_handler(self, job_id):
        pass

    def _do_job_internal(self, job_id, job):
        pass

    def _finish_job(self, job_id, job_result):
        pass
