from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base import AutoJson


__dataset_factory = {
    'json': AutoJson
}


def get_names():
    return list(__dataset_factory.keys())


def init_dataset(name, **kwargs):
    if name not in list(__dataset_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name,
                                                                                        list(__dataset_factory.keys())))
    return __dataset_factory[name](**kwargs)
