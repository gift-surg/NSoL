##
# \file ParameterStudy.py
# \brief      Abstract class to define paths to predefined paths to write and
#             read for parameter studies
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import itertools
from abc import ABCMeta, abstractmethod

import pythonhelper.PythonHelper as ph
from numericalsolver.definitions import FILENAME_EXTENSION


class ParameterStudy(object):
    __metaclass__ = ABCMeta

    def __init__(self, directory, name):
        self._directory = directory
        self._name = name

    def _get_path_to_file_parameters(self, specifier="_parameters"):
        return os.path.join(self._directory,
                            self._name + specifier + "." + FILENAME_EXTENSION)

    def _get_path_to_file_measures(self, measure, specifier="_measure_"):
        return os.path.join(self._directory,
                            self._name + specifier + measure +
                            "." + FILENAME_EXTENSION)

    def _get_path_to_file_computational_time(self,
                                             specifier="_computational_time"):
        return os.path.join(self._directory,
                            self._name + specifier + "." + FILENAME_EXTENSION)
