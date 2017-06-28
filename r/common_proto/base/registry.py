from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

_LOCATION_TAG = 'location'
_OBJ_TAG = 'obj'
_registry_dict = dict()


class Registry(object):
    def __init__(self, registry_name):
        self.__name = registry_name
        self.__registry = dict()

    def register(self, obj_name, obj):
        if obj_name in self.__registry:


    def list_name(self):
        pass

    def get_obj_by_name(self, obj_name):
        pass


class Register(object):
    @classmethod
    def register(cls, registry_name, obj_name, obj):
        pass

    @classmethod
    def get_obj_by_name(cls, registry_name, obj_name):
        pass


def RegisterDecorator(object):
    def __init__(self, registry_name, obj_name):
        if registry_name not in _registry_dict:
            _registry_dict[registry_name] = Registry(registry_name)
        self.__registry = _registry_dict[registry_name]
        self.__obj_name = obj_name

    def __call__(self, obj):
        self.__registry.register(self.__obj_name, obj)
        return obj
