from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_job_queue_client import BaseJobQueueClient
from base.redis_client import RedisClientPool
from base.registry import RegisterDecorator


@RegisterDecorator('queue_client', 'RedisQueueClient')
class RedisJobQueueClient(BaseJobQueueClient):
    def __init__(self, redis_url, queue_name, job_cls):
        super(RedisQueueClient, self).__init__(redis_url, queue_name, job_cls)
        self._client = RedisClientPool.redis_client(redis_url)

    def fetch_job(self, block=True):
        if block:
            queue_name, job_id = self._client.br_pop(self._queue_name)
        else:
            queue_name, job_id = self._client.r_pop(self._queue_name)

        if not job_id:
            raise ValueError('Fetch empty job id')

        job_str = self._client.get(job_id)
        if not job_str:
            raise ValueError('job %s str is not exist' % job_id)

        job_pd = self._client.get(job_id)
        job_pb.ParseFromString(job_str)
        return job_id.decode('utf-8'), job_pb

    def push_job_id(self, job_id):
        pass

    def add_job(self, job_id, job_pb):
        pass

    def queue_size(self):
        pass

    def empty_queue(self):
        pass

    def del_job_by_id(self, job_id):
        pass
