from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import gflags

from base.registry import Register

FLAGS = gflags.FLAGS

class FileStreamHandler(logging.FileHandler):
    def __init__(self, file_path, mode='a', level=logging.NOTSET):
        self.__file_path = file_path
        self.__mode = mode
        self.__file_connector = Register.get_obj_by_name('file_connector', FLAGS.file_connector)

        super(FileStreamHandler, self).__init__(file_connector, mode=mode)

        self.baseFilename = self.__file_path
        self.setLevel(level)

    def _open(self):
        return self.__file_connector.open(self.__file_path, self.__mode)
