##
# \file TikhonovLinearSolver.py
# \brief      Class to define a numerical solver for solving the linear
#             least-squares problems with Tikhonov regularization
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import scipy
import numpy as np

from numericalsolver.LinearSolver import LinearSolver

from definitions import EPS
import numericalsolver.lossFunctions as lf


##
# Class to estimate the unique minimizer of the convex minimization problem
# max_x [1/2 ||rho( Ax-b )||^2 + alpha/2 ||Bx-b_reg||^2].
# \date       2017-07-20 23:43:19+0100
#
class TikhonovLinearSolver(LinearSolver):

    ##
    # Store relevant information for solving a linear least-squares problem
    # with Tikhonov regularization
    # \date       2017-07-20 23:47:48+0100
    #
    # \param      self       The object
    # \param      A          Function associated to linear operator A: X->Y; x
    #                        -> A(x) with x being a 1D numpy array
    # \param      A_adj      Function associated to adjoint linear operator
    #                        A^*: Y -> X; y -> A^*(y)
    # \param      B          Function associated to the linear operator B:
    #                        X->Z; x->B(x) with x being a 1D numpy array
    # \param      B_adj      Function associated to adjoint linear operator
    #                        B^*: Z->X; z->B^*(z)
    # \param      b          Right hand-side of linear system Ax = b as 1D
    #                        numpy array
    # \param      alpha      Regularization parameter; scalar >= 0
    # \param      x0         Initial value as 1D numpy array
    # \param      b_reg      Right hand-side of linear system associated to the
    #                        regularizer, i.e. Bx = b_reg.
    # \param      minimizer  String defining the used optimizer, i.e. "lsmr",
    #                        "least_squares" or any solver as provided by
    #                        scipy.optimize.minimize
    # \param      data_loss  Data loss function rho specified as string, e.g.
    #                        "linear", "soft_l1", "huber", "cauchy", "arctan".
    # \param      iter_max   Number of maximum iterations for used minimizer,
    #                        integer value
    # \param      verbose    Verbose output, bool
    #
    def __init__(self, A, A_adj, B, B_adj, b, alpha, x0, b_reg=0,
                 minimizer="lsmr", data_loss="linear",
                 iter_max=10, verbose=0):

        super(self.__class__, self).__init__(
            A=A, A_adj=A_adj, b=b, x0=x0, alpha=alpha, data_loss=data_loss,
            verbose=verbose)

        self._B = B
        self._B_adj = B_adj
        self._b_reg = b_reg
        self._minimizer = minimizer
        self._iter_max = iter_max

    def _run(self):

        # Monitor output
        if self._monitor is not None:
            self._monitor.add_x(self._x)

        # Get augmented linear system
        A, b = self._get_augmented_linear_system(self._alpha)

        # Define residual function and its Jacobian
        residual = lambda x: A*x - b
        jacobian_residual = lambda x: A

        if self._minimizer == "lsmr" and self._data_loss != "linear":
            raise ValueError(
                "LSMR solver cannot be used with non-linear data loss")

        # Use scipy.sparse.linalg.lsmr
        elif self._minimizer == "lsmr" and self._data_loss == "linear":

            # Linear least-squares method
            self._x = scipy.sparse.linalg.lsmr(
                A, b,
                maxiter=self._iter_max,
                show=self._verbose,
                atol=0,
                btol=0)[0]

            # Clip negative values
            self._x = np.clip(self._x, 0, np.inf)

        # Use scipy.optimize.least_squares
        elif self._minimizer == "least_squares":
            # BE AWARE:
            # Loss function is applied to both data and regularization term!
            # Remark: it seems that least_squares solver does not cope with
            # non-linear loss. Maybe because of the use of sparse linear
            # operator?

            method = "trf"
            bounds = (0, np.inf)
            x0 = np.clip(self._x0, 0, np.inf)

            # Non-linear least squares algorithm
            self._x = scipy.optimize.least_squares(
                fun=residual,
                jac=jacobian_residual,
                jac_sparsity=jacobian_residual,
                x0=x0,
                method=method,
                tr_solver='lsmr',
                bounds=bounds,
                loss=self._data_loss,
                max_nfev=self._iter_max,
                verbose=2*self._verbose,
            ).x

        # Use scipy.optimize.minimize
        else:
            x0 = np.clip(self._x0, 0, np.inf)
            bounds = [[0, None]]*x0.size

            # Define cost function and its Jacobian
            cost = lambda x: \
                self._get_cost_data_term(x) + \
                self._alpha * self._get_cost_regularization_term(x)
            grad_cost = lambda x: \
                self._get_gradient_cost_data_term(x) + \
                self._alpha * self._get_gradient_cost_regularization_term(x)

            self._x = scipy.optimize.minimize(
                method=self._minimizer,
                fun=cost,
                jac=grad_cost,
                x0=x0,
                bounds=bounds,
                options={'maxiter': self._iter_max, 'disp': self._verbose}).x

        # Monitor output
        if self._monitor is not None:
            self._monitor.add_x(self._x)

    def _get_augmented_linear_system(self, alpha):

        # With regularization
        if alpha > EPS:

            # Define forward and backward operators
            A_fw = lambda x: self._A_augmented(x, np.sqrt(alpha))
            A_bw = lambda x: self._A_augmented_adj(x, np.sqrt(alpha))

            # Define right-hand side b
            b = np.zeros(self._x0.size + self._B(self._x0).size)
            b[0:self._b.size] = self._b
            b[self._b.size:] = np.sqrt(alpha) * self._b_reg

        # Without regularization
        else:

            # Define forward and backward operators
            A_fw = lambda x: self._A(x)
            A_bw = lambda x: self._A_adj(x)

            # Define right-hand side b
            b = self._b

        # Construct (sparse) linear operator A
        A = scipy.sparse.linalg.LinearOperator(
            shape=(b.size, self._x0.size),
            matvec=A_fw,
            rmatvec=A_bw)

        return A, b

    def _A_augmented(self, x, sqrt_alpha):

        A_augmented_x = np.concatenate((
            self._A(x),
            sqrt_alpha*self._B(x)))

        return A_augmented_x

    def _A_augmented_adj(self, x, sqrt_alpha):

        x_upper = x[:self._b.size]
        x_lower = x[self._b.size:]

        A_augmented_adj_x = self._A_adj(x_upper) + \
            sqrt_alpha * self._B_adj(x_lower)

        return A_augmented_adj_x

    def _get_cost_regularization_term(self, x):
        return 0.5 * np.sum(self._B(x)**2)

    def _get_gradient_cost_regularization_term(self, x):
        return self._B_adj(self._B(x))
