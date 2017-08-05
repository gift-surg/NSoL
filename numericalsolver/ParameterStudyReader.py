##
# \file ParameterStudyReader.py
# \brief Read parameter studies obtained by ParameterStudy object
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import numpy as np
import re
import natsort

import pythonhelper.PythonHelper as ph

from numericalsolver.ParameterStudyBase import ParameterStudyBase
from numericalsolver.definitions import REGEX_FILENAMES
from numericalsolver.definitions import FILENAME_EXTENSION


class ParameterStudyReader(ParameterStudyBase):

    def __init__(self, directory, name=None):
        self._directory = directory
        self._name = name

    def set_name(self, name):
        self._name = name

    def read_study(self):
        if self._name is None:
            raise ValueError("Set name of study first.")

        if not ph.directory_exists(self._directory):
            raise ValueError("Directory '%s' does not exist" %
                             (self._directory))

        self._measures = self._get_measure_names()

        if len(self._measures) == 0:
            raise RuntimeError("No measures to study '%s' found in '%s'"
                               % (self._name, self._directory))

        path_to_file_parameters = self._get_path_to_file_parameters()
        lines = ph.read_file_line_by_line(path_to_file_parameters)

        # Omit first line
        lines = lines[1:]

        for i in range(len(lines)):
            # Remove comments and newline at the end
            lines[i] = re.sub("## ", "", lines[i])
            lines[i] = re.sub("\n", "", lines[i])

        self._parameters_dic = self._get_parameters_dic(lines)
        self._line_to_parameter_labels_dic = \
            self._get_line_to_parameter_labels_dic(lines[1:])

    def get_measures(self):
        return self._measures

    def get_parameters(self):
        return self._parameters_dic

    def get_line_to_parameter_labels(self):
        return self._line_to_parameter_labels_dic

    def get_results(self, measure):
        return np.loadtxt(self._get_path_to_file_measures(measure), skiprows=2)

    def _get_measure_names(self):
        pattern = self._name + \
            "_measure_(" + REGEX_FILENAMES + ")[.]" + FILENAME_EXTENSION
        p = re.compile(pattern)

        measures = [p.match(f).group(1)
                    for f in os.listdir(self._directory) if p.match(f)]

        return measures

    def _get_parameters_dic(self, lines):

        # Link parameter name with parameter value
        # e.g. dict['alpha'] = 0.1,
        #      dict["data_loss_scale"] = 2,
        #      dict["data_loss"] = "linear"
        parameters_dic = {}

        # split at tabs to get filenames and forget about first line
        parameters = lines[0].split("\t")
        lines = lines[1:]

        for i, parameter in enumerate(parameters):

            tmp = [line.split("\t")[i] for line in lines]
            tmp = list(set(tmp))

            tmp = natsort.natsorted(tmp, key=lambda y: y.lower())

            try:
                tmp = [float(t) for t in tmp]
                # tmp = np.array(tmp)

            except:
                pass

            parameters_dic[parameter] = tmp

        return parameters_dic

    def _get_line_to_parameter_labels_dic(self, lines, separator=", ", full=True):

        # Link parameter values to line in text files
        # e.g. dict['0.1,2,linear'] = 2, i.e. in line number 2
        line_to_parameters_dic = {}
        for i, line in enumerate(lines):
            if full:
                label = [k + "=" + v for (k, v) in
                         zip(self._parameters_dic.keys(), line.split("\t"))]
                label = (separator).join(label)
            else:
                label = (separator).join(line.split("\t"))
            line_to_parameters_dic[i] = label

        return line_to_parameters_dic
