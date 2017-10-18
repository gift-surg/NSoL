##
# \file admm_linear_solver.py
# \brief      Class to define a numerical solver for solving the linear
#             least-squares problems with TV regularization via the Alternating
#             Direction Method of Multipliers (ADMM) method
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import numpy as np

import pysitk.python_helper as ph

from nsol.linear_solver import LinearSolver
import nsol.tikhonov_linear_solver as tk
from nsol.prior_measures import PriorMeasures as prior_meas


##
# Class to estimate the unique minimizer to the convex minimization problem
# max_x [1/2 ||rho( Ax-b )||^2 + alpha g(Bx-b_reg)] with g(z) = TV(z) using
# ADMM.
# \date       2017-07-21 00:23:34+0100
#
class ADMMLinearSolver(LinearSolver):

    ##
    # Store relevant information for solving a linear least-squares problem
    # with TV regularization via ADMM
    # \date       2017-07-21 00:24:24+0100
    #
    # \param      self             The object
    # \param      A                Function associated with linear operator A:
    #                              X->Y; x->A(x) with x being a 1D numpy array
    # \param      A_adj            Function associated with adjoint linear
    #                              operator A^*: Y->X; y->A^*(y)
    # \param      b                Right hand-side of linear system Ax = b as
    #                              1D numpy array
    # \param      B                Function associated with linear operator B:
    #                              x->B(x) with x and B(x) 1D numpy arrays
    # \param      B_adj            Function associated with adjoint linear
    #                              operator B^*
    # \param      x0               Initial value as 1D numpy array
    # \param      dimension        Dimension of space as integer indicating
    #                              either 1D, 2D or 3D problems
    # \param      b_reg            Right hand-side of linear system associated
    #                              to the regularizer, i.e. Bx = b_reg.
    # \param      alpha            Regularization parameter; scalar > 0
    # \param      iter_max         Number of maximum iterations for used
    #                              minimizer, integer value
    # \param      minimizer        String defining the used optimizer, i.e.
    #                              "lsmr", "least_squares" or any solver as
    #                              provided by scipy.optimize.minimize
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
    # \param      rho              regularization parameter of augmented
    #                              Lagrangian term; scalar > 0
    # \param      iterations       Number of ADMM iterations, integer value
    # \param      x_scale          Characteristic scale of each variable.
    #                              Setting x_scale is equivalent to
    #                              reformulating the problem in scaled
    #                              variables ``xs = x / x_scale``
    # \param      verbose          Verbose output, bool
    #
    def __init__(self,
                 A, A_adj,
                 b,
                 B, B_adj,
                 x0,
                 dimension,
                 b_reg=0,
                 alpha=0.01,
                 iter_max=10,
                 minimizer="lsmr",
                 data_loss="linear",
                 data_loss_scale=1,
                 rho=0.5,
                 iterations=10,
                 x_scale=1,
                 verbose=0):

        super(self.__class__, self).__init__(
            A=A, A_adj=A_adj, b=b, x0=x0, alpha=alpha, iter_max=iter_max,
            minimizer=minimizer, data_loss=data_loss,
            data_loss_scale=data_loss_scale, x_scale=x_scale, verbose=verbose)

        self._B = B
        self._B_adj = B_adj
        self._b_reg = b_reg / self._x_scale
        self._dimension = dimension
        self._rho = float(rho)
        self._iterations = iterations

    ##
    # Sets the regularization parameter for augmented Lagrangian.
    # \date       2017-08-05 04:08:03+0100
    #
    # \param      self  The object
    # \param      rho   regularization  parameter of augmented Lagrangian term;
    #                   scalar > 0
    #
    def set_rho(self, rho):
        self._rho = rho

    ##
    # Gets the regularization parameter for augmented Lagrangian.
    # \date       2017-08-05 04:08:42+0100
    #
    # \param      self  The object
    #
    # \return     scalar > 0
    #
    def get_rho(self):
        return self._rho

    ##
    # Gets the dimension of space.
    # \date       2017-08-05 18:42:33+0100
    #
    # \param      self  The object
    #
    # \return     The dimension of space as int.
    #
    def get_dimension(self):
        return self._dimension

    ##
    # Sets the number of ADMM iterations.
    # \date       2017-08-05 18:59:12+0100
    #
    # \param      self        The object
    # \param      iterations  Number of ADMM iterations, integer value
    #
    def set_iterations(self, iterations):
        self._iterations = iterations

    ##
    # Gets the number of ADMM iterations.
    # \date       2017-08-05 18:59:39+0100
    #
    # \param      self  The object
    #
    # \return     Number of ADMM iterations, integer value.
    #
    def get_iterations(self):
        return self._iterations

    ##
    # Execute solver
    # \date       2017-08-05 19:00:53+0100
    #
    # \param      self  The object
    #
    def _run(self):

        # Monitor output
        if self._observer is not None:
            self._observer.add_x(self.get_x())

        v = self._B(self._x0) - self._b_reg
        w = np.zeros_like(v)

        for i in range(0, self._iterations):

            if self._verbose:
                ph.print_title("ADMM iteration %d/%d" %
                               (i+1, self._iterations))
            else:
                ph.print_info("ADMM iteration %d/%d" %
                              (i+1, self._iterations))

            self._x, v, w = self._perform_ADMM_iteration(self._x, v, w)

            # Monitor output
            if self._observer is not None:
                self._observer.add_x(self.get_x())

            # shape = (256, 256)

            # recon = np.array(self._x.reshape(*shape))
            # ph.show_arrays(recon, title="ADMM_iter%d"%(i+1), fig_number=1)

            # v_split = [h.reshape(*shape) for h in self._get_split(v, self._dimension)]
            # ph.show_arrays(v_split, title="v", cmap="jet", fig_number=2)

            # w_split = [h.reshape(*shape) for h in self._get_split(w, self._dimension)]
            # ph.show_arrays(w_split, title="w", cmap="jet", fig_number=3)

            # ph.pause()

    def _perform_ADMM_iteration(self, x, v, w):

        # 1) Update primal variable
        x = self._solve_tikhonov_least_squares(x, v, w)

        # Compute derivatives for steps 2 and 3
        Bx_plus_w_minus_b_reg = self._B(x) + w - self._b_reg

        # 2) Update auxiliary variable
        v = self._prox_g(Bx_plus_w_minus_b_reg,
                         tau=self._alpha / self._rho,
                         dimension=self._dimension)

        # 3) Update scaled dual variable
        w = Bx_plus_w_minus_b_reg - v

        return x, v, w

    def _solve_tikhonov_least_squares(self, x, v, w):

        b_reg = v - w + self._b_reg

        tikhonov = tk.TikhonovLinearSolver(
            A=self._A, A_adj=self._A_adj,
            B=self._B, B_adj=self._B_adj,
            b=self._b, b_reg=b_reg,
            alpha=self._rho,
            x0=x,
            x_scale=1,
            iter_max=self._iter_max,
            data_loss=self._data_loss,
            minimizer=self._minimizer,
            verbose=self._verbose)
        tikhonov.run()

        return tikhonov.get_x()

    def _prox_g(self, t, tau, dimension):
        t_split = self._get_split(t, dimension)
        t_norm = np.sqrt(self._get_squared_sum_of_split(t_split))

        ind = t_norm > tau

        v = np.zeros_like(t)
        m = t_split[0].shape[0]
        for i in range(0, dimension):
            v_tmp = v[i*m:(i+1)*m, ...]
            v_tmp[ind] = self._get_soft_threshold(tau, t_norm[ind]) * \
                t_split[i][ind] / t_norm[ind]
            v[i*m:(i+1)*m, ...] = v_tmp

        return v

    ##
    # Gets the split of (n, ...) numpy into d (n/d, ...) numpy arrays with d
    # being the dimension of space
    #
    # This recovers the respective x (y, z) variables from a single numpy array
    # \date       2017-07-21 00:38:54+0100
    #
    # \param      self       The object
    # \param      x          Numpy array
    # \param      dimension  Numpy array as integer
    #
    # \return     List of d numpy arrays corresponding to the split
    #
    def _get_split(self, x, dimension):
        x_split = np.array_split(x, dimension)
        return x_split

    ##
    # Given a split, return the squared sum
    # \date       2017-07-21 00:41:49+0100
    #
    # \param      self     The object
    # \param      x_split  List of split obtained via 'get_split'
    #
    # \return     The squared sum of the split.
    #
    def _get_squared_sum_of_split(self, x_split):

        # x_i ** 2
        tmp = x_split[0] ** 2

        # Compute sum_k x_k^2
        for i in range(1, len(x_split)):
            tmp += x_split[i]**2

        return tmp

    ##
    # Gets the soft threshold.
    #
    # The soft threshold is defined as
    # \f[ S_\ell(t) =  \max(|t|-\ell,0)\,\text{sgn}(t) = \begin{cases}
    # t-\ell,& \text{if } t>\ell \\ 0,& \text{if } |t|\le\ell \\ t+\ell,&
    # \text{if } t<\ell \end{cases}
    # \f]
    # \date       2017-07-21 00:37:13+0100
    #
    # \param      self  The object
    # \param      ell   threshold as scalar > 0
    # \param      t     array containing the values to be thresholded
    #
    # \return     The soft threshold.
    #
    def _get_soft_threshold(self, ell, t):
        return np.maximum(np.abs(t) - ell, 0) * np.sign(t)

    def _get_cost_regularization_term(self, x):
        return prior_meas.total_variation(x, self._B, self._dimension)
