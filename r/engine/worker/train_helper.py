from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gflags
import signal

from base.registry import Register
from base.runtime_watcher import RuntimeWatcher
from base.runtime_watcher import TimeoutException
from evaluate.evaluator_manager import EvaluatorManager
from data_processor.dataset.data_set import DataSets
from common_proto.base.base_pb2 import ReaderType


gflags.DEFINE_string('max_train_time', 60 * 60 * 5, 'max number of seconds to train')

FLAGS = gflags.FLAGS


class TrainHelper(object):
    def __init__(self, logger):
        self.__logger = logger
        self.__info_log_template = 'status=%s\tcost=%.4f'
        self.__runtime_watcher = RuntimeWatcher()

    @staticmethod
    def raise_timeout(self, job_id, job):
        raise TimeoutException('train timeout.')

    def do_train_job(self, job_id, job):
        signal.signal(signal.SIGALRM, self.raise_timeout)

        self.__runtime_watcher = RuntimeWatcher()
        evaluator_manager = EvaluatorManager.get_manager()

        self.__runtime_watcher.start_task('load_dataset')
        datasets = self.__load_datasets(job_id, job.input_schema)
        init_train_cost = self.__runtime_watcher.stop_task('load_dataset')
        self.__logger.info(self.__info_log_template % ('load_dataset', init_train_cost))

        self.__runtime_watcher.start_task('prepare_train')
        trainer, evaluate_result_file = self.__prepare_to_train(job_id, job, datasets)
        init_train_cost = self.__runtime_watcher.stop_task('prepare_train')

        self.__runtime_watcher.start_task('do_train_loop')
        learning_schema = job.learning_schema
        output_schema = job.output_schema
        try:
            self.__do_train_loop()
        except StopIteration as stop:
            pass
        except Exception as e:
            train_cost = self.__runtime_watcher.elapsed_time('do_train_loop')
            err_log = self.__info_log_template % ('train_fail', train_cost)
            err_log += '\terr=%s' % str(e)
            self.__logg.error(error_log)
        finally:
            evaluate_result_file.close()

    def __load_datasets(self, job_id, input_schema):
        data_file_path = '/'.join([FLAGS.base_dir, job_id, 'input_data', input_schema.file_name])
        raw_data_frame = self.__read_raw_data_frame(input_schema.reader_type, data_file_path)
        pre_process_schema_list = input_schema.feature_schema.pre_process_schema
        processed_data_frame = self.__preprocess_raw_data_frame(raw_data_frame, pre_process_schema_list)
        datasets = DataSets(processed_data_frame, input_schema)
        return datasets

    def __prepare_to_train(self, job_id, job, datasets):
        pass

    def __do_train_loop():
        pass

    def __read_raw_data_frame(self, reader_type, data_file_path):
        read_cls = self.create_reader_cls(read_type)

    def __preprocess_raw_data_frame(self, raw_data_frame, pre_process_schema_list):
        pass

    @classmethod
    def create_reader_cls(cls, read_type):
        return Register.get_obj_by_name('reader', ReaderType.Type.name(read_type))
