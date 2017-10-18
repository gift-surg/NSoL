##
# \file solver.py
# \brief      Abstract class to define a numerical solver
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import numpy as np
import datetime
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph


##
# Abstract class to define a numerical solver
# \date       2017-07-20 23:17:11+0100
#
class Solver(object):
    __metaclass__ = ABCMeta

    ##
    # Store relevant information common for all numerical solvers
    # \date       2017-07-20 23:39:13+0100
    #
    # \param      self     The object
    # \param      x0       Initial value as 1D numpy array
    # \param      x_scale  Characteristic scale of each variable. Setting
    #                      x_scale is equivalent to reformulating the problem
    #                      in scaled variables ``xs = x / x_scale``
    # \param      verbose  Verbose output, bool
    #
    def __init__(self, x0, x_scale, verbose):
        self._x_scale = float(x_scale)
        self._x0 = np.array(x0, dtype=np.float64) / self._x_scale
        self._x = np.array(self._x0)
        self._verbose = verbose
        self._computational_time = datetime.timedelta(seconds=0)
        self._observer = None

    ##
    # Sets the characteristic scale for each variable.
    # \date       2017-08-04 19:09:20+0100
    #
    # \param      self     The object
    # \param      x_scale  Characteristic scale of each variable. Setting
    #                      x_scale is equivalent to reformulating the problem
    #                      in scaled variables ``xs = x / x_scale``
    #
    def set_x_scale(self, x_scale):
        self._x_scale = x_scale

    ##
    # Gets the characteristic scale for each variable.
    # \date       2017-08-04 19:10:13+0100
    #
    # \param      self  The object
    #
    # \return     the characteristic scale for each variable, float or array.
    #
    def get_x_scale(self):
        return self._x_scale

    ##
    # Sets the verbose.
    # \date       2017-08-05 18:26:15+0100
    #
    # \param      self     The object
    # \param      verbose  Turn on/off verbose; bool
    #
    def set_verbose(self, verbose):
        self._verbose = verbose

    ##
    # Gets the verbose.
    # \date       2017-08-05 18:26:44+0100
    #
    # \param      self  The object
    #
    # \return     boolean flag.
    #
    def get_verbose(self):
        return self._verbose

    ##
    # Sets the initial value.
    # \date       2017-09-06 17:32:06+0100
    #
    # \param      self  The object
    # \param      x0    Initial value as 1D numpy array
    #
    def set_x0(self, x0):
        self._x0 = np.array(x0, dtype=np.float64) / self._x_scale
        self._x = np.array(self._x0)

    ##
    # Gets the initial value.
    # \date       2017-08-05 18:30:48+0100
    #
    # \param      self  The object
    #
    # \return     Initial value as 1D numpy array.
    #
    def get_x0(self):
        return np.array(self._x0) * self._x_scale

    ##
    # Gets the obtained numerical estimate to the minimization problem
    # \date       2017-07-20 23:39:36+0100
    #
    # \param      self  The object
    #
    # \return     Numerical solution as 1D numpy array
    #
    def get_x(self):
        return np.array(self._x) * self._x_scale

    ##
    # Gets the computational time it took to obtain the numerical estimate.
    # \date       2017-07-20 23:40:17+0100
    #
    # \param      self  The object
    #
    # \return     The computational time as string
    #
    def get_computational_time(self):
        return self._computational_time

    ##
    # Sets the observer to monitor performance
    # \date       2017-08-05 18:31:18+0100
    #
    # \param      self     The object
    # \param      observer  The observer as Observer object
    #
    def set_observer(self, observer):
        self._observer = observer

    ##
    # Run the numerical solver to obtain the numerical estimate
    # \date       2017-07-20 23:41:08+0100
    #
    # \param      self  The object
    #
    def run(self):

        if self._x0.ndim != 1:
            raise ValueError("Initial value x0 must be a 1D array")

        time_start = ph.start_timing()

        # Execute solver
        self._run()

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)

        if self._verbose:
            ph.print_info("Required computational time: %s" %
                          (self.get_computational_time()))

        if self._observer is not None:
            self._observer.set_computational_time(
                self.get_computational_time())

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def print_statistics(self):
        pass
