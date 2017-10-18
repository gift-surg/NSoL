##
# \file linear_solver.py
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

import pysitk.python_helper as ph

from nsol.solver import Solver
from nsol.loss_functions import LossFunctions as lf


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
    # \param      self             The object
    # \param      A                Function associated to linear operator A:
    #                              X->Y; x->A(x) with x being a 1D numpy array
    # \param      A_adj            Function associated to adjoint linear
    #                              operator A^*: Y->X; y->A^*(y)
    # \param      b                Right hand-side of linear system Ax = b as
    #                              1D numpy array
    # \param      x0               Initial value as 1D numpy array
    # \param      alpha            Regularization parameter; scalar
    # \param      x_scale          Characteristic scale of each variable.
    #                              Setting x_scale is equivalent to
    #                              reformulating the problem in scaled
    #                              variables ``xs = x / x_scale``
    # \param      data_loss        Data loss function rho specified as string,
    #                              e.g. "linear", "soft_l1", "huber", "cauchy",
    #                              "arctan".
    # \param      data_loss_scale  Value of soft margin between inlier and
    #                              outlier residuals, default is 1.0. The loss
    #                              function is evaluated as rho_(f2) = C**2 *
    #                              rho(f2 / C**2), where C is data_loss_scale.
    #                              This parameter has no effect with
    #                              data_loss='linear', but for other loss
    #                              values it is of crucial importance.
    # \param      minimizer        String defining the used optimizer, i.e.
    #                              "lsmr", "least_squares" or any solver as
    #                              provided by scipy.optimize.minimize
    # \param      iter_max         Number of maximum iterations for used
    #                              minimizer, integer value
    # \param      verbose          Verbose output, bool
    #
    def __init__(self,
                 A,
                 A_adj,
                 b,
                 x0,
                 alpha,
                 x_scale,
                 data_loss,
                 data_loss_scale,
                 minimizer,
                 iter_max,
                 verbose):

        Solver.__init__(self, x0=x0, x_scale=x_scale, verbose=verbose)

        self._A = A
        self._A_adj = A_adj
        self._b = b / self._x_scale
        self._alpha = float(alpha)
        self._data_loss = data_loss
        self._data_loss_scale = float(data_loss_scale)
        self._minimizer = minimizer
        self._iter_max = iter_max

    ##
    # Gets the linear operator A
    # \date       2017-09-06 16:29:11+0100
    #
    # \param      self  The object
    #
    # \return     Forward operator as function: 1D -> 1D
    #
    def get_A(self):
        return self._A

    ##
    # Gets the adjoint linear operator A
    # \date       2017-09-06 16:29:11+0100
    #
    # \param      self  The object
    #
    # \return     Adjoint forward operator as function: 1D -> 1D
    #
    def get_A_adj(self):
        return self._A_adj

    ##
    # Gets the right hand-side b of linear system Ax = b
    # \date       2017-09-06 16:37:32+0100
    #
    # \param      self  The object
    #
    # \return     Right hand-side as 1D array
    #
    def get_b(self):
        return np.array(self._b) * self._x_scale

    ##
    # Sets the regularization parameter alpha.
    # \date       2017-08-04 18:54:24+0100
    #
    # \param      self   The object
    # \param      alpha  Regularization  parameter; scalar
    #
    def set_alpha(self, alpha):
        self._alpha = alpha

    ##
    # Gets the regularization parameter alpha.
    # \date       2017-08-04 18:54:59+0100
    #
    # \param      self  The object
    #
    # \return     scalar
    #
    def get_alpha(self):
        return self._alpha

    ##
    # Sets the data loss function specified as string
    # \date       2017-08-04 18:55:38+0100
    #
    # \param      self       The object
    # \param      data_loss  The data loss as defined in LossFunctions, string
    #
    def set_data_loss(self, data_loss):
        if data_loss not in lf.get_loss.keys():
            raise ValueError("data_loss must be in " +
                             str(lf.get_loss.keys()))
        self._data_loss = data_loss

    ##
    # Gets the data loss.
    # \date       2017-08-04 19:00:42+0100
    #
    # \param      self  The object
    #
    # \return     The data loss function specifier, string
    #
    def get_data_loss(self):
        return self._data_loss

    ##
    # Sets the data loss scale.
    # \date       2017-08-04 19:00:54+0100
    #
    # \param      self             The object
    # \param      data_loss_scale  Value of soft margin between inlier and
    #                              outlier residuals, default is 1.0. The loss
    #                              function is evaluated as rho_(f2) = C**2 *
    #                              rho(f2 / C**2), where C is data_loss_scale.
    #                              This parameter has no effect with
    #                              data_loss='linear', but for other loss
    #                              values it is of crucial importance.
    #
    def set_data_loss_scale(self, data_loss_scale):
        self._data_loss_scale = data_loss_scale

    ##
    # Gets the data loss scale.
    # \date       2017-08-04 19:01:57+0100
    #
    # \param      self  The object
    #
    # \return     The data loss scale; scalar
    #
    def get_data_loss_scale(self):
        return self._data_loss_scale

    ##
    # Sets the minimizer.
    # \date       2017-08-04 19:06:16+0100
    #
    # \param      self       The object
    # \param      minimizer  String defining the used optimizer, i.e. "lsmr",
    #                        "least_squares" or any solver as provided by
    #                        scipy.optimize.minimize
    #
    def set_minimizer(self, minimizer):
        self._minimizer = minimizer

    ##
    # Gets the minimizer.
    # \date       2017-08-04 19:06:58+0100
    #
    # \param      self  The object
    #
    # \return     The minimizer as string
    #
    def get_minimizer(self):
        return self._minimizer

    ##
    # Sets the number of maximum iterations.
    # \date       2017-08-04 19:07:23+0100
    #
    # \param      self      The object
    # \param      iter_max  Number of maximum iterations for used minimizer,
    #                       integer value
    #
    def set_iter_max(self, iter_max):
        self._iter_max = iter_max

    ##
    # Gets the iterator maximum.
    # \date       2017-08-04 19:08:06+0100
    #
    # \param      self  The object
    #
    # \return     Number of maximum iterations for used minimizer, integer
    #             value
    #
    def get_iter_max(self):
        return self._iter_max

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
        ph.print_info("Computational time: %s" %
                      (self.get_computational_time()))
        ph.print_info("Cost data term (f, loss=%s, scale=%g): " %
                      (self._data_loss, self._data_loss_scale) +
                      fmt % (cost_data) +
                      " (ell2-cost: " + fmt % (cost_data_ell2) + ")")
        ph.print_info(
            "Cost regularization term (g): " +
            fmt % (cost_regularizer))
        ph.print_info(
            "Total cost (f + alpha g; alpha = %g" % (self._alpha) + "): " +
            fmt % (cost_data + self._alpha * cost_regularizer))

    def _get_cost_data_term(self, x):

        residual = self._A(x) - self._b
        cost = 0.5 * np.sum(
            lf.get_loss[self._data_loss](f2=residual ** 2,
                                         f_scale=self._data_loss_scale))

        return cost

    def _get_ell2_cost_data_term(self, x):

        residual = self._A(x) - self._b
        cost = 0.5 * np.sum(residual ** 2)

        return cost

    def _get_gradient_cost_data_term(self, x):

        residual = self._A(x) - self._b

        grad = self._A_adj(
            lf.get_gradient_loss[self._data_loss](f2=residual ** 2,
                                                  f_scale=self._data_loss_scale
                                                  ) * residual)

        return grad

    @abstractmethod
    def _get_cost_regularization_term(self, x):
        pass
