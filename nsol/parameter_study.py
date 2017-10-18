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

import pysitk.python_helper as ph
from nsol.definitions import FILENAME_EXTENSION


##
# Abstract class holding the information where to store paths to read and write
# for parameter studies
# \date       2017-08-05 18:44:56+0100
#
class ParameterStudy(object):
    __metaclass__ = ABCMeta

    def __init__(self, directory, name):
        self._directory = directory
        self._name = name

    def get_parameter_study_name(self):
        return self._name

    ##
    # Gets the path to file where the parameter configurations are stored.
    # \date       2017-08-05 18:45:57+0100
    #
    # \param      self       The object
    # \param      specifier  Specifier of parameter file (for easier reading
    #                        later)
    #
    # \return     The path to file parameters.
    #
    def _get_path_to_file_parameters(self, specifier="_parameters"):
        return os.path.join(self._directory,
                            self._name + specifier + "." + FILENAME_EXTENSION)

    ##
    # Gets the path to file for each computed measure.
    # \date       2017-08-05 18:46:31+0100
    #
    # \param      self       The object
    # \param      measure    Measure ID as string
    # \param      specifier  Specifier of measure files (for easier reading
    #                        later)
    #
    # \return     The path to file measures.
    #
    def _get_path_to_file_measures(self, measure, specifier="_measure_"):
        return os.path.join(self._directory,
                            self._name + specifier + measure +
                            "." + FILENAME_EXTENSION)

    ##
    # Gets the path to file where computational times are stored
    # \date       2017-08-05 18:48:13+0100
    #
    # \param      self       The object
    # \param      specifier  Specifier of computational time file (for easier
    #                        reading later)
    #
    # \return     The path to file holding computational time.
    #
    def _get_path_to_file_computational_time(self,
                                             specifier="_computational_time"):
        return os.path.join(self._directory,
                            self._name + specifier + "." + FILENAME_EXTENSION)

    ##
    # Gets the path to file where reconstructions after n iterations are stored
    # \date       2017-08-05 18:48:13+0100
    #
    # \param      self       The object
    # \param      specifier  Specifier of reconstruction file (for easier
    #                        reading later)
    #
    # \return     The path to file holding reconstructions.
    #
    def _get_path_to_file_reconstructions(self,
                                          specifier="_reconstructions"):
        return os.path.join(self._directory,
                            self._name + specifier + ".npz")
