##
# \file parameter_study.py
# \brief Abstract class to provide basis for solver specific parameter studies
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import re
import six
import itertools
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph

from nsol.parameter_study import ParameterStudy
from nsol.reader_parameter_study import ReaderParameterStudy


##
# Abstract class to run parameter studies for solvers
# \date       2017-08-05 19:19:21+0100
#
class SolverParameterStudy(ParameterStudy):
    __metaclass__ = ABCMeta

    def __init__(self,
                 solver,
                 parameters,
                 observer,
                 dir_output,
                 name,
                 reconstruction_info,
                 append,
                 ):

        ParameterStudy.__init__(self, directory=dir_output, name=name)

        self._solver = solver
        self._parameters = parameters
        self._observer = observer
        self._reconstruction_info = reconstruction_info

        # Append (instead of overwrite) potentially existing study
        self._append = append

    ##
    # Run parameter study and write results to specified files
    # \date       2017-08-05 19:27:16+0100
    #
    # \param      self  The object
    #
    def run(self):

        # Reset observer and attach to solver in case this has not happened
        self._observer.set_name(self._name)
        self._observer.clear_x_list()
        self._solver.set_observer(self._observer)

        # Create or append to a previous study
        bool_prev_study = ph.file_exists(self._get_path_to_file_parameters())
        if not self._append or not bool_prev_study:
            self._create_file_parameters()
            self._create_files_measures()
            self._create_file_computational_time()

            # Overwrite append-flag in case set as a new study is created
            self._append = False
        else:
            ph.print_info("Append previous study ... ")
            self._check_that_studies_match()

        time_start = ph.start_timing()

        self._run()

        # Get computational time in seconds
        self._computational_time = ph.stop_timing(time_start)

    def get_computational_time(self):
        return self._computational_time

    ##
    # Gets the parameters specified for parameter study.
    # \date       2017-08-05 23:35:09+0100
    #
    # \param      self  The object
    #
    # \return     The parameters as dictionary.
    #
    def get_parameters(self):
        return self._parameters

    ##
    # Check that study can be appended
    # \date       2019-03-20 15:41:55+0000
    #
    # \param      self  The object
    #
    def _check_that_studies_match(self):

        def raise_error(h1, h2, info=""):
            msg = "Study cannot be appended as parameter settings do " \
                "not match: %s != %s%s" % (h1, h2, info)
            raise RuntimeError(msg)

        reader_parameter_study = ReaderParameterStudy(
            directory=self._directory, name=self._name)
        reader_parameter_study.read_study()
        header_prev_study = reader_parameter_study.get_file_header()
        header = self._get_fileheader()

        # split header information into its settings (but ignore date info etc)
        header_list = header.split(" ")[1:-2]
        header_prev_study_list = header_prev_study.split(" ")[1:-2]

        # check each individual option
        for i in range(len(header_list)):

            # eliminate "," at the end of string
            h1 = re.sub(",", "", header_list[i])
            h2 = re.sub(",", "", header_prev_study_list[i])

            # test if study name and non-numeric parameters matches,
            # e.g. TVL2 and minimizer=lsmr
            if h1 == h2:
                continue

            # test if numeric parameter values match, e.g. alpha=0.01
            if "=" in h1 and "=" in h2:
                h1_var, h1_val = h1.split("=")
                h2_var, h2_val = h2.split("=")

                if h1_var != h2_var:
                    raise_error(h1, h2)

                if ph.is_float(h1_val) and ph.is_float(h2_val):
                    try:
                        np.testing.assert_almost_equal(
                            float(h1_val), float(h2_val), decimal=6)
                        continue
                    except AssertionError as e:
                        raise_error(h1, h2, ". %s" % e)

            raise_error(h1, h2)

    def _run(self):

        dic_parameter = {}

        # Create list from itertools. Then total number of iterations known
        iterations = list(itertools.product(*self._parameters.values()))

        if self._append:
            # Retrieve previous reconstructions and iterations
            reader_parameter_study = ReaderParameterStudy(
                directory=self._directory, name=self._name)
            reader_parameter_study.read_study()
            previous_iterations = len(
                reader_parameter_study.get_parameters_to_line().keys())
            dic_x = dict(reader_parameter_study.get_reconstructions())
        else:
            previous_iterations = 0
            dic_x = {k: v for k, v in six.iteritems(self._reconstruction_info)}

        for i, vals in enumerate(iterations):

            ph.print_title("%s: Iteration %d/%d" %
                           (self._name, i + 1, len(iterations)))

            for j, key in enumerate(self._parameters.keys()):

                # Update varying parameters of solver
                map(eval("self._solver.set_" + key), [vals[j]])

                # Store current parameter values and print it on the screen
                dic_parameter[key] = eval(
                    "str(self._solver.get_" + key + "())")
                ph.print_info(key + " = %s" % (dic_parameter[key]))

            # Execute solver
            self._solver.run()

            # Compute similarity measures for solver estimate iterations
            self._observer.compute_measures()

            # Write all measure results to file for all iterations
            measures = self._observer.get_measures()
            for measure in measures:

                # Write measure results as line to the measure's file
                nda = measures[measure].reshape(1, -1)
                self._add_to_file_measures(measure, nda)

            # Write required computational time
            self._add_to_file_computational_time(
                self._observer.get_computational_time())

            # Write current parameter values to file
            self._add_to_file_parameters(dic_parameter)

            # Write last iteration of reconstruction to file
            # Data array is associated to line in parameters file
            var = str(i + previous_iterations)
            dic_x[var] = np.array(
                self._observer.get_x_list()[-1], dtype=np.float16)

            # Write results of all obtained arrays at each iteration
            # (Previous results will be overwritten at each iteration)
            self._write_to_file_reconstructions(dic_x)

            # Clear observer for next parameter selection
            self._observer.clear_x_list()

            # Reset solver to initial value
            self._solver.set_x0(self._solver.get_x0())

    ##
    # Creates file where all parameters configurations are stored.
    # \date       2017-08-05 19:25:04+0100
    #
    # \param      self  The object
    #
    def _create_file_parameters(self):

        # Build header
        header = self._get_fileheader()
        header += "## " + ("\t").join(self._parameters.keys()) + "\n"

        # Write file
        ph.write_to_file(self._get_path_to_file_parameters(), header, "w")

    ##
    # Creates a file for each chosen measure where all solver iterations are
    # stored for each parameter configuration
    # \date       2017-08-05 19:25:28+0100
    #
    # \param      self  The object
    #
    def _create_files_measures(self):

        measures = self._observer.get_measures()
        for measure in measures.keys():

            # Build header
            header = self._get_fileheader()
            header += "## " + measure + " for iteration 0 to n\n"

            # Write file
            ph.write_to_file(self._get_path_to_file_measures(measure),
                             header, "w")

    ##
    # Creates a file to computational time required to run each parameter
    # configuration.
    # \date       2017-08-05 19:26:09+0100
    #
    # \param      self  The object
    #
    def _create_file_computational_time(self):

        # Build header
        header = self._get_fileheader()
        header += "## " + "Computational time measured for n iterations" + "\n"

        # Write file
        ph.write_to_file(self._get_path_to_file_computational_time(),
                         header, "w")

    ##
    # Adds parameter configuration to parameter file.
    # \date       2017-08-04 22:06:58+0100
    #
    # \param      self            The object
    # \param      dic_parameters  dictionary holding the parameters and their
    #                             values
    #
    def _add_to_file_parameters(self, dic_parameters):
        text = ("\t").join(dic_parameters.values())
        text += "\n"
        ph.write_to_file(self._get_path_to_file_parameters(), text, "a")

    ##
    # Adds obtained results for all iterations to the measure's file
    # \date       2017-08-05 19:19:51+0100
    #
    # \param      self     The object
    # \param      measure  ID of measure which was evaluated
    # \param      nda      Numpy data array as (1 x N) data array
    #
    def _add_to_file_measures(self, measure, nda):
        ph.write_array_to_file(self._get_path_to_file_measures(measure), nda)

    ##
    # Adds measured computational time to file.
    # \date       2017-08-05 19:23:35+0100
    #
    # \param      self                The object
    # \param      computational_time  computational time as timedelta object
    #
    def _add_to_file_computational_time(self, computational_time):
        text = str(computational_time)
        text += "\n"
        ph.write_to_file(
            self._get_path_to_file_computational_time(), text, "a")

    ##
    # Adds obtained results for all iterations to the reconstruction's file
    # \date       2017-08-05 19:19:51+0100
    #
    # \param      self     The object
    # \param      measure  ID of measure which was evaluated
    # \param      nda      Numpy data array as (1 x N) data array
    #
    def _write_to_file_reconstructions(self, dic):
        np.savez_compressed(self._get_path_to_file_reconstructions(), **dic)
        ph.print_info("File '%s' written" %
                      (self._get_path_to_file_reconstructions()))

    ##
    # Get solver-specific header for all files to be written
    # \date       2017-08-05 19:24:23+0100
    #
    # \param      self  The object
    #
    # \return     The fileheader as tring.
    #
    @abstractmethod
    def _get_fileheader(self):
        pass
