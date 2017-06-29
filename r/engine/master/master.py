from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gflags

from engine.job_queue import *
from base.registry import Register
from common_proto.base.base_pb2 import JobType
from common_proto.train.train_job_pb2 import TrainJob


gflags.DEFINE_string('redie_url', 'redis://root:chuangxingongchang@10.18.125.15:8410', 'The path of redis queue.')
gflags.DEFINE_string('queue_client', 'RedisQueueClient', 'The queue clien class name')
gflags.DEFINE_string('job_dir_struct', 'input_data,model,output_data,log', 'The dir struct of a job')

FLAGS = gflags.FLAGS

class Master(object):
    def __init__(self):
        queue_client_cls = Register.get_obj_by_name('queue_client', FLAGS.queue_client)
        self.__job_queue_mapping = {
            JobType.TYPE_TRAIN_JOB: queue_client_cls(FLAGS.redis_url, FLAGS.train_job_queue, TrainJob)}
        self.__file_connector = Register.get_obj_by_name('file_connector', FLAGS.file_connector)

    def init_job_context(self, job_type, job):
        pass

    def push_job_id(self, job_type, job_id):
        pass

    def check_job_type(self, job_type):
        if job_type not in self.__job_queue_mapping:
            raise KeyError('Unknow job type: %s' % job_type)
