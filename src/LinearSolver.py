##
# \file LinearSolver.py
# \brief      Abstract class to define a numerical solver for linear
#             least-squares problems
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import numpy as np
from abc import ABCMeta, abstractmethod

from src.Solver import Solver

sys.path.insert(1, os.path.join(
    os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py'))
import utilities.lossFunctions as lf
import utilities.PythonHelper as ph

##
# Abstract class to define a numerical solver for linear least-squares
# problems.
#
# Abstract class to define a numerical solver for linear least-squares problems
# with (non-linear) data loss function rho and regularizer g, i.e.
# max_x [1/2 ||rho( Ax-b )||^2 + alpha g(x)]
# \date       2017-07-20 23:24:24+0100
#
class LinearSolver(Solver):
    __metaclass__ = ABCMeta

    ##
    # Store relevant information for linear solvers
    # \date       2017-07-20 23:27:49+0100
    #
    # \param      self       The object
    # \param      A          Function associated to linear operator A: X->Y;
    #                        x->A(x) with x being a 1D numpy array
    # \param      A_adj      Function associated to adjoint linear operator
    #                        A^*: Y->X; y->A^*(y)
    # \param      b          Right hand-side of linear system Ax = b as 1D
    #                        numpy array
    # \param      x0         Initial value as 1D numpy array
    # \param      alpha      Regularization parameter; scalar
    # \param      data_loss  Data loss function rho specified as string, e.g.
    #                        "linear", "soft_l1", "huber", "cauchy", "arctan".
    #
    def __init__(self, A, A_adj, b, x0, alpha, data_loss):

        Solver.__init__(self, x0=x0)

        self._A = A
        self._A_adj = A_adj
        self._b = b
        self._alpha = alpha
        self._data_loss = data_loss

    ##
    # Gets the total cost as 1/2 ||rho( Ax-b )||^2 + alpha g(x)
    # \date       2017-07-20 23:38:46+0100
    #
    # \param      self  The object
    #
    # \return     The total cost as scalar value
    #
    def get_total_cost(self):
        data_cost = self.get_cost_data_term()
        regularization_cost = self.get_cost_regularization_term()
        return data_cost + self._alpha * regularization_cost

    ##
    # Gets the cost of the data term, i.e. f(x) = 1/2 ||rho( Ax-b )||^2
    # \date       2017-07-20 23:56:30+0100
    #
    # \param      self  The object
    #
    # \return     The cost of the data term.
    #
    def get_cost_data_term(self):
        return self._get_cost_data_term(self._x)

    ##
    # Gets the ell2 cost of the data term, i.e.  1/2 || Ax-b ||^2
    # \date       2017-07-20 23:58:34+0100
    #
    # \param      self  The object
    #
    # \return     The ell2 cost data term.
    #
    def get_ell2_cost_data_term(self):
        return self._get_ell2_cost_data_term(self._x)

    ##
    # Gets the cost of the regularization term g(x)
    # \date       2017-07-21 00:00:26+0100
    #
    # \param      self  The object
    #
    # \return     The cost of the regularization term.
    #
    def get_cost_regularization_term(self):
        return self._get_cost_regularization_term(self._x)

    ##
    # Prints the statistics of the performed optimization
    # \date       2017-07-21 00:01:10+0100
    #
    # \param      self  The object
    # \param      fmt   Format for printing numerical values
    #
    def print_statistics(self, fmt="%.3e"):

        cost_data = self.get_cost_data_term()
        cost_data_ell2 = self.get_ell2_cost_data_term()
        cost_regularizer = self.get_cost_regularization_term()

        ph.print_subtitle("Summary Optimization")
        ph.print_debug_info("Computational time: %s" %
                            (self.get_computational_time()))
        ph.print_debug_info("Cost data term (f, loss=%s): " %
                            (self._data_loss) + fmt % (cost_data) +
                            " (ell2-cost: " + fmt % (cost_data_ell2) + ")")
        ph.print_debug_info(
            "Cost regularization term (g): " +
            fmt % (cost_regularizer))
        ph.print_debug_info(
            "Total cost (f + alpha g; alpha = %g" % (self._alpha) + "): " +
            fmt % (cost_data + self._alpha * cost_regularizer))

    def _get_cost_data_term(self, x):

        residual = self._A(x) - self._b
        cost = 0.5 * np.sum(lf.get_loss[self._data_loss](residual ** 2))

        return cost

    def _get_ell2_cost_data_term(self, x):

        residual = self._A(x) - self._b
        cost = 0.5 * np.sum(residual ** 2)

        return cost

    def _get_gradient_cost_data_term(self, x):

        residual = self._A(x) - self._b

        grad = self._A_adj(
            lf.get_gradient_loss[self._data_loss](residual ** 2) * residual)

        return grad

    @abstractmethod
    def _get_cost_regularization_term(self, x):
        pass
