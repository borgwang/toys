from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseJobQueueClient(object):
    
    def __init__(self, queue_path, queue_name, job_cls):
        self._queue_path = queue_path
        self._client = None
        self._queue_name = queue_name
        self._job_cls = job_cls

    def fetch_job(self):
        raise NotImplementedError('Must use specify job queue client.')

    def push_job_id(self, job_id):
        raise NotImplementedError('Must use specify job queue client.')

    def add_job(self, job_id, job_pb):
        raise NotImplementedError('Must use specify job queue client.')

    def queue_size(self):
        raise NotImplementedError('Must use specify job queue client.')

    def empty_queue(self):
        raise NotImplementedError('Must use specify job queue client.')

    def del_job_by_id(self, job_id):
        raise NotImplementedError('Must use specify job queue client.')
