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

import pysitk.python_helper as ph

from nsol.parameter_study import ParameterStudy
from nsol.definitions import REGEX_FILENAMES
from nsol.definitions import FILENAME_EXTENSION


class ReaderParameterStudy(ParameterStudy):

    def __init__(self, directory, name):
        self._directory = directory
        self._name = name

        # List of lines of read parameter file
        self._lines_params = None

    ##
    # Reads the specified parameter study.
    # \date       2017-08-05 19:54:59+0100
    #
    # \param      self  The object
    #
    def read_study(self):

        if not ph.directory_exists(self._directory):
            raise ValueError("Directory '%s' does not exist" %
                             (self._directory))

        # Get all measures of study
        self._measures = self._get_measure_names()

        if len(self._measures) == 0:
            raise RuntimeError("No measures to study '%s' found in '%s'"
                               % (self._name, self._directory))

        # Store the lines of the parameter file which hold the
        # information on the used parameters for further processing
        self._lines_params = self._read_parameter_file_lines()

        # Rebuild dictionary which was used for the parameter study as input
        self._parameters_dic = self._get_parameters()

        parameters = self._get_parameters()
        for k in parameters.keys():
            if len(parameters[k]) == 0:
                raise RuntimeError(
                    "Directory '%s' does not contain "
                    "suitable parameter study info" % self._directory)

    ##
    # Gets the reconstructions.
    # \date       2017-09-12 14:09:51+0100
    #
    # \param      self  The object
    #
    # \return     All reconstructions and associated information as dictionary.
    #
    def get_reconstructions(self):
        if not ph.file_exists(self._get_path_to_file_reconstructions()):
            raise IOError("File '%s' not available" %
                          self._get_path_to_file_reconstructions())
        return np.load(self._get_path_to_file_reconstructions())

    ##
    # Gets the names of all evaluated measures belonging to selected study.
    # \date       2017-08-05 19:53:46+0100
    #
    # \param      self  The object
    #
    # \return     Names of all evaluated measures as list of strings.
    #
    def get_measures(self):
        self._check_that_study_was_read()
        return self._measures

    ##
    # Gets the file header.
    # \date       2019-03-20 15:36:41+0000
    #
    # \param      self  The object
    #
    # \return     The file header as string.
    #
    def get_file_header(self):
        self._check_that_study_was_read()

        # Fetch information on used parameters from file. List of strings
        # holds information of a single line in file
        path_to_file_parameters = self._get_path_to_file_parameters()
        lines = ph.read_file_line_by_line(path_to_file_parameters)

        return lines[0]

    ##
    # Gets the results for selected measure.
    #
    # Return results for selected measure as numpy data array for all parameter
    # runs (rows) and all iterations (columns).
    # \date       2017-08-05 19:32:12+0100
    #
    # \param      self     The object
    # \param      measure  ID of measure whose results shall be returned,
    #                      string
    #
    # \return     Results of measure as 2D numpy data array
    #
    def get_results(self, measure):
        return np.loadtxt(self._get_path_to_file_measures(measure), skiprows=2)

    ##
    # Gets the dictionary of the parameters which were used for the specified
    # parameter study as input.
    #
    # E.g. parameters = {
    #   "alpha": [0.02, 0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42, 0.47],
    #   "data_loss": ["linear", "arctan"],
    #   "data_loss_scale": [1., 1.1],
    # }
    # \date       2017-08-05 19:44:30+0100
    #
    # \param      self  The object
    #
    # \return     The parameters as dictionary.
    #
    def get_parameters(self):
        self._check_that_study_was_read()
        return self._parameters_dic

    ##
    # Gets the dictionary linking the parameters with the line in the result
    # array.
    #
    # E.g. dict = {
    #   ('0.02', 'linear', '1.'): 0,
    #   ('0.02', 'linear', '1.1'): 1,
    #   ('0.02', 'arctan', '1.'): 2,
    #   ...
    # }
    # \date       2017-08-05 21:04:17+0100
    #
    # \param      self  The object
    #
    # \return     The parameters to line.
    #
    def get_parameters_to_line(self):
        self._check_that_study_was_read()

        # Build dictionary which links parameter set to line of result array
        return self._get_parameters_to_line()

    ##
    # Gets the lines to given parameters where only key of dictionary is
    # varying.
    #
    # E.g. parameters = {
    #   'alpha': [0.02, 0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42, 0.47],
    #   'data_loss': 'arctan',
    #   'data_loss_scale': 1.,
    # }
    # The returned lines refer to the respective entry of the array obtained by
    # get_results()
    # \date       2017-08-05 22:08:20+0100
    #
    # \param      self        The object
    # \param      parameters  dictionary where only one parameter is varying
    #
    # \return     The lines to associated parameters as 1D numpy array.
    #
    def get_lines_to_parameters(self, parameters):
        self._check_that_study_was_read()
        return self._get_lines_to_parameters(parameters)

    ##
    # Gets dictionary which links each line/row data array results with
    # readable parameter configuration.
    #
    # E.g. dict = {
    #   0: 'alpha=0.02, rho=0.1',
    #   1: 'alpha=0.02, rho=0.5',
    #   2: 'alpha=0.02, rho=1.0',
    # }
    # \date       2017-08-05 19:45:28+0100
    #
    # \param      self       The object
    # \param      separator  Separator between parameter information, string
    # \param      compact    Turn on/off compact format, bool
    #
    # \return     Dictionary to link line/row of array with parameters.
    #
    def get_line_to_parameter_labels(self, separator=", ", compact=False):
        # Link line of numpy array results to readable label
        return self._get_line_to_parameter_labels(separator=separator,
                                                  compact=compact)

    def _read_parameter_file_lines(self):

        # Fetch information on used parameters from file. List of strings
        # holds information of a single line in file
        path_to_file_parameters = self._get_path_to_file_parameters()
        lines = ph.read_file_line_by_line(path_to_file_parameters)

        # Skip first line which contains header information only
        lines = lines[1:]

        # Remove comments and newline at the end of each line. Only relevant
        # parameter information remains
        for i in range(len(lines)):
            lines[i] = re.sub("## ", "", lines[i])
            lines[i] = re.sub("\n", "", lines[i])

        # Lines with information on the varying parameters of the study
        return lines

    def _get_measure_names(self):
        pattern = self._name + \
            "_measure_(" + REGEX_FILENAMES + ")[.]" + FILENAME_EXTENSION
        p = re.compile(pattern)

        measures = [p.match(f).group(1)
                    for f in os.listdir(self._directory) if p.match(f)]

        return measures

    def _get_parameters(self):

        lines = list(self._lines_params)

        # Link parameter name with parameter value
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
                tmp.sort()

            except:
                pass

            parameters_dic[parameter] = tmp

        return parameters_dic

    def _get_parameters_to_line(self):

        # Skip parameter information header
        lines = self._lines_params[1:]

        parameters_to_line_dic = {}
        for i, line in enumerate(lines):
            parameters = [v for v in line.split("\t")]
            parameters_to_line_dic[tuple(parameters)] = i

        return parameters_to_line_dic

    def _get_lines_to_parameters(self, parameters):

        if parameters.keys() != self._parameters_dic.keys():
            raise ValueError("Provided dictionary keys must match. "
                             "Required keys for this study are " +
                             str(self._parameters_dic.keys()))

        # Find single varying key
        varying_key = None
        for key in parameters.keys():
            if type(parameters[key]) in (tuple, list, np.ndarray):
                if len(parameters[key]) == 1:
                    raise ValueError("Single entry in key '%s' must not "
                                     "be a list" % (key))

                elif len(parameters[key]) > 1:
                    if varying_key is None:
                        varying_key = key
                        rows = len(parameters[key])
                    else:
                        raise ValueError("Provided dictionary can only vary "
                                         "in a single key")

        parameters_to_line_dic = self.get_parameters_to_line()
        lines = np.zeros(rows, dtype=int)
        for i in range(rows):
            key = []
            for k in parameters.keys():
                if k == varying_key:
                    key.append(str(parameters[k][i]))
                else:
                    key.append(str(parameters[k]))

            line = parameters_to_line_dic[tuple(key)]
            lines[i] = line

        # lines = sorted(lines)

        return lines

    def _get_line_to_parameter_labels(self, separator, compact):

        # Skip parameter information header
        lines = self._lines_params[1:]

        # Link parameter values to line in text files
        line_to_parameters_dic = {}
        for i, line in enumerate(lines):
            if compact:
                # e.g. dict[2] = '0.1,2,linear'
                label = (separator).join(line.split("\t"))
            else:
                # e.g. dict[2] = 'alpha=0.1, rho=2, data_loss=linear'
                label = [k + "=" + v for (k, v) in
                         zip(self._parameters_dic.keys(), line.split("\t"))]
                label = (separator).join(label)
            line_to_parameters_dic[i] = label

        return line_to_parameters_dic

    ##
    # Check that study was read to obtain information on parameters
    # \date       2017-08-05 22:44:12+0100
    #
    # \param      self  The object
    #
    def _check_that_study_was_read(self):
        if self._lines_params is None:
            raise UnboundLocalError("Execute 'read_study' first to get "
                                    "information on parameters.")
