from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')
import time

from concurrent.futures import ThreadPoolExecutor
import gflags
import grpc

from base.logger.logger_helper import LoggerHelper
from common_proto.service import service_pb2
from engine.master.master import Master


gflags.DEFINE_string('base_dir', '/home/abbo/workspace/toys/r/abbo/jobs', 'The base dir of the job data and result.')
gflags.DEFINE_string('file_connector', 'LocalFileConnector', 'The type of fs connector')
gflags.DEFINE_string('hdfs_file_io', 'WebHDFSFileIO', 'The hdfs file io type.')
gflags.DEFINE_string('log_dir', '/home/abbo/workspace/toys/r/abbo/log', 'The base dir of log.')
gflags.DEFINE_string('logger_name', 'master', 'The name of master logger.')
gflags.DEFINE_string('master_path', '127.0.0.1:50053', 'The ip:port of service')
gflags.DEFINE_string('master_thread_nums', 2, 'The thread nums of master.')
gflags.DEFINE_string('train_job_queue', 'train:job_queue', 'The queue of train job.')

FLAGS = gflags.FLAGS

SECONDS_PER_DAY = 60 * 60 * 24

class MLServer(service_pb2.MLServiceServicer):
    def __init__(self):
        self.__master = Master()
        self.__logger = self.__init_logger()
        self.__log_info_template = 'path=%s\tjob_id=%s\tcost=%.4f'

    @staticmethod
    def __init_logger():
        logger = LoggerHelper.init_logger(FLAGS.logger_name)
        log_file_name = '.'.join([FLAGS.logger_name, 'log'])
        handler = LoggerHelper.create_file_handler(FLAGS.log_dir, log_file_name)
        LoggerHelper.set_handler(logger, handler)
        return logger

    def start(self):
        server = grpc.server(ThreadPoolExecutor(max_workers=FLAGS.master_thread_nums))
        service_pb2.add_MLServiceServicer_to_server(self, server)
        server.add_insecure_port(FLAGS.master_path)
        server.start()
        try:
            while True:
                time.sleep(SECONDS_PER_DAY)
        except KeyboardInterrupt:
            server.stop(None)

    def init_train_job(self, train_job, context):
        pass

    def start_train(self, train_job_id, context):
        pass


def main(argv=None):
    FLAGS(argv)
    ml_server = MLServer()
    ml_server.start()

if __name__ == '__main__':
    main(sys.argv)
