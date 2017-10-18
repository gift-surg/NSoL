##
# \file tikhonov_linear_solver.py
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

from nsol.linear_solver import LinearSolver
from nsol.definitions import EPS
from nsol.loss_functions import LossFunctions as lf


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
    # \param      self             The object
    # \param      A                Function associated to linear operator A:
    #                              X->Y; x -> A(x) with x being a 1D numpy
    #                              array
    # \param      A_adj            Function associated to adjoint linear
    #                              operator A^*: Y -> X; y -> A^*(y)
    # \param      b                Right hand-side of linear system Ax = b as
    #                              1D numpy array
    # \param      B                Function associated to the linear operator
    #                              B: X->Z; x->B(x) with x being a 1D numpy
    #                              array
    # \param      B_adj            Function associated to adjoint linear
    #                              operator B^*: Z->X; z->B^*(z)
    # \param      x0               Initial value as 1D numpy array
    # \param      alpha            Regularization parameter; scalar >= 0
    # \param      b_reg            Right hand-side of linear system associated
    #                              to the regularizer, i.e. Bx = b_reg.
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
    # \param      iter_max         The iterator maximum
    # \param      x_scale          Characteristic scale of each variable.
    #                              Setting x_scale is equivalent to
    #                              reformulating the problem in scaled
    #                              variables ``xs = x / x_scale``
    # \param      verbose          Verbose output, bool
    # \param      bounds           The bounds
    #
    def __init__(self,
                 A, A_adj,
                 b,
                 B, B_adj,
                 x0,
                 alpha=0.01,
                 b_reg=0,
                 data_loss="linear",
                 data_loss_scale=1,
                 minimizer="lsmr",
                 iter_max=10,
                 x_scale=1,
                 verbose=0,
                 bounds=(0, np.inf)):

        super(self.__class__, self).__init__(
            A=A, A_adj=A_adj, b=b, x0=x0, alpha=alpha, iter_max=iter_max,
            minimizer=minimizer, data_loss=data_loss,
            data_loss_scale=data_loss_scale, x_scale=x_scale, verbose=verbose)

        self._B = B
        self._B_adj = B_adj
        self._b_reg = b_reg / self._x_scale
        self._bounds = bounds

    ##
    # Gets the linear operator B
    # \date       2017-09-06 16:29:11+0100
    #
    # \param      self  The object
    #
    # \return     Forward operator as function: 1D -> 1D
    #
    def get_B(self):
        return self._B

    ##
    # Gets the adjoint linear operator B
    # \date       2017-09-06 16:29:11+0100
    #
    # \param      self  The object
    #
    # \return     Forward operator as function: 1D -> 1D
    #
    def get_B_adj(self):
        return self._B_adj

    def get_b_reg(self):
        return self._b_reg * self._x_scale

    def _run(self):

        if self._minimizer == "lsmr" and self._data_loss != "linear":
            raise ValueError(
                "lsmr solver cannot be used with non-linear data loss")

        elif self._minimizer == "lsq_linear" and self._data_loss != "linear":
            raise ValueError(
                "lsq_linear solver cannot be used with non-linear data loss")

        # Monitor output
        if self._observer is not None:
            self._observer.add_x(self.get_x())

        # Get augmented linear system
        A, b = self._get_augmented_linear_system(self._alpha)

        # Define residual function and its Jacobian
        residual = lambda x: A*x - b
        jacobian_residual = lambda x: A

        # Clip to bounds
        if self._bounds is not None:
            self._x0 = np.clip(self._x0, self._bounds[0], self._bounds[1])

        # Use scipy.sparse.linalg.lsmr
        if self._minimizer == "lsmr" and self._data_loss == "linear":

            # Linear least-squares method
            self._x = scipy.sparse.linalg.lsmr(
                A, b,
                maxiter=self._iter_max,
                show=self._verbose,
                atol=0,
                btol=0)[0]

            if self._bounds is not None:
                # Clip to bounds
                self._x = np.clip(self._x, self._bounds[0], self._bounds[1])

        # Use scipy.optimize.lsq_linear
        elif self._minimizer == "lsq_linear" and self._data_loss == "linear":

            # linear least-squares problem with bounds on the variables
            self._x = scipy.optimize.lsq_linear(
                A, b,
                max_iter=self._iter_max,
                lsq_solver='lsmr',
                lsmr_tol='auto',
                bounds=self._bounds,
                verbose=2*self._verbose,
            ).x

        # Use scipy.optimize.least_squares
        elif self._minimizer == "least_squares":
            # BE AWARE:
            # Loss function is applied to both data and regularization term!
            # Remark: it seems that least_squares solver does not cope with
            # non-linear loss. Maybe because of the use of sparse linear
            # operator?

            # Non-linear least squares algorithm
            self._x = scipy.optimize.least_squares(
                fun=residual,
                jac=jacobian_residual,
                jac_sparsity=jacobian_residual,
                x0=self._x0,
                # method="trf",
                tr_solver='lsmr',
                bounds=self._bounds,
                loss=self._data_loss,
                f_scale=self._data_loss_scale,
                max_nfev=self._iter_max,
                verbose=2*self._verbose,
            ).x

        # Use scipy.optimize.minimize
        else:
            bounds = [[self._bounds[0], self._bounds[1]]] * self._x0.size

            # Define cost function and its Jacobian
            if self._alpha > EPS:
                cost = lambda x: \
                    self._get_cost_data_term(x) + \
                    self._alpha * self._get_cost_regularization_term(x)
                grad_cost = lambda x: \
                    self._get_gradient_cost_data_term(x) + \
                    self._alpha * \
                    self._get_gradient_cost_regularization_term(x)

            else:
                cost = lambda x: self._get_cost_data_term(x)
                grad_cost = lambda x: self._get_gradient_cost_data_term(x)

            self._x = scipy.optimize.minimize(
                method=self._minimizer,
                fun=cost,
                jac=grad_cost,
                x0=self._x0,
                bounds=bounds,
                options={'maxiter': self._iter_max, 'disp': self._verbose}).x

        # Monitor output
        if self._observer is not None:
            self._observer.add_x(self.get_x())

    def _get_augmented_linear_system(self, alpha):

        # With regularization
        if alpha > EPS:

            # Define forward and backward operators
            A_fw = lambda x: self._A_augmented(x, np.sqrt(alpha))
            A_bw = lambda x: self._A_augmented_adj(x, np.sqrt(alpha))

            # Define right-hand side b
            b = np.zeros(A_fw(self._x0).size)
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
            sqrt_alpha * self._B(x)))

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
