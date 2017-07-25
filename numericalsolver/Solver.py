##
# \file Solver.py
# \brief      Abstract class to define a numerical solver
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import numpy as np
import time
import datetime
from abc import ABCMeta, abstractmethod

import pythonhelper.PythonHelper as ph

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
    # \param      verbose  Verbose output, bool
    #
    def __init__(self, x0, verbose):
        self._x0 = np.array(x0, dtype=np.float64)
        self._x = np.array(x0, dtype=np.float64)
        self._verbose = verbose
        self._computational_time = None
        self._monitor = None

    ##
    # Gets the obtained numerical estimate to the minimization problem
    # \date       2017-07-20 23:39:36+0100
    #
    # \param      self  The object
    #
    # \return     Numerical solution as 1D numpy array
    #
    def get_x(self):
        return np.array(self._x)

    ##
    # Gets the computational time it took to obtain the numerical estimate.
    # \date       2017-07-20 23:40:17+0100
    #
    # \param      self  The object
    #
    # \return     The computational time as string
    #
    def get_computational_time(self):
        return datetime.timedelta(seconds=self._computational_time)

    def set_monitor(self, monitor):
        self._monitor = monitor

    ##
    # Run the numerical solver to obtain the numerical estimate
    # \date       2017-07-20 23:41:08+0100
    #
    # \param      self  The object
    #
    def run(self):

        time_start = time.time()

        self._run()

        # Get computational time in seconds
        self._computational_time = time.time()-time_start

        if self._verbose:
            ph.print_info("Required computational time: %s" %
                          (self.get_computational_time()))

        if self._monitor is not None:
            self._monitor.set_computational_time(self.get_computational_time())

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def print_statistics(self):
        pass
