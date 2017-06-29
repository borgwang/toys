from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')

import gflags
from engine.worker.train_worker import TrainWorker

gflags.DEFINE_string('base_dir', '/home/abbo/workspace/toys/r/abbo/jobs', 'The base dir of the job data and result.')
gflags.DEFINE_string('file_connector', 'LocalFileConnector', 'type of fs connector')
gflags.DEFINE_string('hdfs_file_io', 'WebHDFSFileIO', 'the hdfs file io type')
gflags.DEFINE_string('train_job_queue', 'train:job_queue_abbo', 'the queue of train job')
gflags.DEFINE_string('worker_name', 'train-worker-1', 'the name of worker')

FLAGS = gflags.FLAGS

def main(argv=None):
    FLAGS(argv)
    train_worker = TrainWorker(FLAGS.worker_name)
    train_worker.registry()
    train_worker.do_work_loop()


if __name__ == '__main__':
    main(sys.argv)
