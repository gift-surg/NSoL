##
# \file Monitor.py
# \brief      Class to monitor the iterations of numerical solvers
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import numpy as np


##
# Class to monitor the iterations of numerical solvers
# \date       2017-07-23 01:48:54+0100
#
class Monitor(object):

    ##
    # Store name of solver to be monitored
    # \date       2017-07-23 00:04:55+0100
    #
    # \param      self  The object
    # \param      name  Name for the solver to monitor, string
    #
    def __init__(self, name):
        self._name = name
        self._x_list = []
        self._measures = []
        self._measures_names = []
        self._dic_measures = {}
        self._computational_time = None

    ##
    # Adds a data array to monitor
    # \date       2017-07-23 01:43:49+0100
    #
    # \param      self  The object
    # \param      x     numpy array
    #
    def add_x(self, x):
        self._x_list.append(x)

    ##
    # Sets the measures to be computed
    # \date       2017-07-23 01:44:14+0100
    #
    # \param      self          The object
    # \param      measures_dic  Dictionary with measures_dic = { name: f}
    #                           whereby f = lambda x: f(x) the measure to
    #                           evaluate
    #
    def set_measures(self, measures_dic):
        measures_names = measures_dic.keys()
        for i in range(0, len(measures_names)):
            self._measures_names.append(measures_names[i])
            self._measures.append(measures_dic[measures_names[i]])
            self._dic_measures.update({measures_names[i]: None})

    ##
    # Sets the computational time.
    # \date       2017-07-23 01:45:41+0100
    #
    # \param      self                The object
    # \param      computational_time  The computational time
    #
    def set_computational_time(self, computational_time):
        self._computational_time = computational_time

    ##
    # Calculates the measures.
    # \date       2017-07-23 01:45:50+0100
    #
    # \param      self  The object
    # \post       self._dic_measures contains information on all evaluated
    #             measures
    #
    # \return     The measures.
    #
    def compute_measures(self):
        N = len(self._x_list)
        res = np.zeros(N)
        for k, name in enumerate(self._measures_names):
            for i in range(0, N):
                res[i] = self._measures[k](self._x_list[i])
            self._dic_measures[name] = np.array(res)

    ##
    # Gets the list of collected data arrays
    # \date       2017-07-23 01:46:00+0100
    #
    # \return     list of numpy arrays.
    #
    def get_x_list(self):
        return self._x_list

    ##
    # Gets the name.
    # \date       2017-07-23 01:47:03+0100
    #
    # \param      self  The object
    #
    # \return     The name as string
    #
    def get_name(self):
        return self._name

    ##
    # Gets the measures.
    # \date       2017-07-23 01:47:11+0100
    #
    # \param      self  The object
    #
    # \return     The measures as dictionary with format {measure_name: array}.
    #
    def get_measures(self):
        return self._dic_measures

    ##
    # Gets the computational time.
    # \date       2017-07-23 01:48:40+0100
    #
    # \param      self  The object
    #
    # \return     The computational time.
    #
    def get_computational_time(self):
        return self._computational_time
