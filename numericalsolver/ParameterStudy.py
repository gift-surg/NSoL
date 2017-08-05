##
# \file ParameterStudy.py
# \brief Abstract class to provide basis for solver specific parameter studies
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import itertools
from abc import ABCMeta, abstractmethod

import pythonhelper.PythonHelper as ph

from numericalsolver.ParameterStudyBase import ParameterStudyBase


class ParameterStudy(ParameterStudyBase):
    __metaclass__ = ABCMeta

    def __init__(self,
                 solver,
                 parameters,
                 monitor,
                 dir_output,
                 name):

        ParameterStudyBase.__init__(self, directory=dir_output, name=name)

        self._solver = solver
        self._parameters = parameters
        self._monitor = monitor

    def run(self):

        # Reset monitor and attach to solver in case this has not happened
        self._monitor.clear_x_list()
        self._solver.set_monitor(self._monitor)

        # Create files where output is written to
        self._create_file_parameters()
        self._create_files_measures()
        self._create_file_computational_time()

        time_start = ph.start_timing()

        self._run()

        # Get computational time in seconds
        self._computational_time = ph.stop_timing(time_start)

    def _run(self):

        dic_parameter = {}

        # Create list from itertools. Then total number of iterations known
        iterations = list(itertools.product(*self._parameters.values()))

        for i, vals in enumerate(iterations):

            ph.print_title("%s: Iteration %d/%d" %
                           (self._name, i+1, len(iterations)))

            for j, key in enumerate(self._parameters.keys()):

                # Update varying parameters, i.e. those which are not None
                if vals[j] is not None:
                    map(eval("self._solver.set_" + key), [vals[j]])

                # Get current parameter values
                dic_parameter[key] = \
                    eval("str(self._solver.get_" + key + "())")

                # Print current parameter values
                ph.print_info(key + " = %s" % (dic_parameter[key]))

            # Write current parameter values to file
            self._add_to_file_parameters(dic_parameter)

            # Execute solver
            self._solver.run()

            # Compute similarity measures for solver estimate iterations
            self._monitor.compute_measures()

            # Write all measure results to file for all iterations
            measures = self._monitor.get_measures()
            for measure in measures:
                nda = measures[measure].reshape(1, -1)
                self._add_to_file_measures(measure, nda)

            # Write required computational time
            self._add_to_file_computational_time(
                self._monitor.get_computational_time())

            # Clear monitor for next parameter selection
            self._monitor.clear_x_list()

    def _create_file_parameters(self):

        # Build header
        header = self._get_fileheader()
        header += "## " + ("\t").join(self._parameters.keys()) + "\n"

        # Write file
        ph.write_to_file(self._get_path_to_file_parameters(), header, "w")

    def _create_files_measures(self):

        measures = self._monitor.get_measures()
        for measure in measures.keys():

            # Build header
            header = self._get_fileheader()
            header += "## " + measure + " for iteration 0 to n\n"

            # Write file
            ph.write_to_file(self._get_path_to_file_measures(measure),
                             header, "w")

    def _create_file_computational_time(self):

        # Build header
        header = self._get_fileheader()
        header += "## " + "Computational time measured for n iterations" + "\n"

        # Write file
        ph.write_to_file(self._get_path_to_file_computational_time(),
                         header, "w")

    ##
    # Adds to file parameters.
    # \date       2017-08-04 22:06:58+0100
    #
    # \param      self        The object
    # \param      parameters  parameter dictionary
    #
    def _add_to_file_parameters(self, dic_parameters):
        text = ("\t").join(dic_parameters.values())
        text += "\n"
        ph.write_to_file(self._get_path_to_file_parameters(), text, "a")

    def _add_to_file_measures(self, measure, nda):
        ph.write_array_to_file(self._get_path_to_file_measures(measure), nda)

    def _add_to_file_computational_time(self, computational_time):
        text = str(computational_time)
        text += "\n"
        ph.write_to_file(
            self._get_path_to_file_computational_time(), text, "a")

    @abstractmethod
    def _get_fileheader(self):
        pass
