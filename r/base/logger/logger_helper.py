from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import logging.handlers

from .log_handler import FileStreamHandler

class LoggerHelper(object):
    @staticmethod
    def init_logger(logger_name, level=logging.INFO):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        return logger

    @staticmethod
    def create_file_handler(log_base_dir, log_file_name, level=logging.INFO):
        log_file_path = '/'.join([log_base_dir, log_file_name])
        log_handler = FileStreamHandler(log_file_path, level=level)
        formatter = logging.Formatter(
            '%(name)-12s %(asctime)s %(levelname)-8s '
            '%(filename)s (%(lineno)d)\t####\t'
            '%(message)s', '%a, %d %b %Y %H:%M:%S', )
        log_handler.setFormatter(formatter)

        return log_handler

    @staticmethod
    def set_handler(logger, handler):
        logger.addHandler(handler)

    @staticmethod
    def remove_handler(logger, handler):
        handler.close()
        logger.removeHandler(handler)
