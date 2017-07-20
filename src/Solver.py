##
# \file Solver.py
# \brief      Class to create linear operators used for blurring and
#             differentiation in both 2B and 3B
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import sys
import os
import numpy as np
import scipy
import time
import datetime
from abc import ABCMeta, abstractmethod

from definitions import EPS

sys.path.insert(1, os.path.join(
    os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py'))
import utilities.lossFunctions as lf
import utilities.PythonHelper as ph


class Solver(object):
    __metaclass__ = ABCMeta

    def __init__(self, x0):
        self._x0 = np.array(x0, dtype=np.float64)
        self._x = np.array(x0, dtype=np.float64)
        self._computational_time = None

    def get_x(self):
        return np.array(self._x)

    def get_computational_time(self):
        return datetime.timedelta(seconds=self._computational_time)

    def run(self):

        time_start = time.time()

        self._run()

        # Get computational time in seconds
        self._computational_time = time.time()-time_start

        print("Required computational time: %s" %
              (self.get_computational_time()))

    @abstractmethod
    def _run(self):
        pass


class LinearSolver(Solver):
    __metaclass__ = ABCMeta

    def __init__(self, A, A_adj, b, x0, alpha, data_loss):

        Solver.__init__(self, x0)

        self._A = A
        self._A_adj = A_adj
        self._b = b
        self._alpha = alpha
        self._data_loss = data_loss

    def get_total_cost(self):
        data_cost = self.get_cost_data_term()
        regularization_cost = self.get_cost_regularization_term()
        return data_cost + self._alpha * regularization_cost

    def get_cost_data_term(self):
        return self._get_cost_data_term(self._x)

    def get_cost_data_term_ell2(self):
        return self._get_cost_data_term_ell2(self._x)

    def get_cost_regularization_term(self):
        return self._get_cost_regularization_term(self._x)

    def print_statistics(self, fmt="%.3e"):

        cost_data = self.get_cost_data_term()
        cost_data_ell2 = self.get_cost_data_term_ell2()
        cost_regularizer = self.get_cost_regularization_term()

        ph.print_title("Summary Optimization")
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

    def _get_cost_data_term_ell2(self, x):

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


class TikhonovLinearSolver(LinearSolver):

    def __init__(self, A, A_adj, B, B_adj, b, alpha, x0, b_reg=0,
                 minimizer="lsmr", data_loss="linear",
                 iter_max=10):

        super(self.__class__, self).__init__(
            A=A, A_adj=A_adj, b=b, x0=x0, alpha=alpha, data_loss=data_loss)

        self._B = B
        self._B_adj = B_adj
        self._b_reg = b_reg
        self._minimizer = minimizer
        self._iter_max = iter_max

    def _run(self):

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
                A, b, maxiter=self._iter_max, show=True)[0]

            # Chop off negative values
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
                verbose=2,
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
                options={'maxiter': self._iter_max, 'disp': True}).x

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

        # No regularization
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


class ADMMLinearSolver(LinearSolver):

    def __init__(self, A, A_adj, b, x0,
                 D, D_adj, dimension,
                 alpha=0.05, data_loss="linear", iter_max=10,
                 rho=0.5, ADMM_iterations=10):

        super(self.__class__, self).__init__(
            A=A, A_adj=A_adj, b=b, x0=x0, alpha=alpha, data_loss=data_loss)

        self._D = D
        self._D_adj = D_adj
        self._dimension = dimension
        self._iter_max = iter_max
        self._rho = rho
        self._ADMM_iterations = ADMM_iterations

    def _run(self):

        v = self._D(self._x0)
        w = np.zeros_like(v)

        for i in range(0, self._ADMM_iterations):

            self._x, v, w = self._perform_ADMM_iteration(self._x, v, w)

            # shape = (256, 256)

            # recon = np.array(self._x.reshape(*shape))
            # ph.show_arrays(recon, title="ADMM_iter%d"%(i+1), fig_number=1)

            # v_split = [h.reshape(*shape) for h in self._get_split(v)]
            # ph.show_arrays(v_split, title="v", cmap="jet", fig_number=2)

            # w_split = [h.reshape(*shape) for h in self._get_split(w)]
            # ph.show_arrays(w_split, title="w", cmap="jet", fig_number=3)

            # ph.pause()

    def _perform_ADMM_iteration(self, x, v, w):

        # 1) Update primal variable using first-order Tikhonov regularization
        x = self._perform_ADMM_step_1_TK1_recon_solution(x, v, w, self._rho)

        # Compute derivatives for steps 2 and 3
        Dx = self._D(x)

        # 2) Update auxiliary variable
        v = self._perform_ADMM_step_2_auxiliary_variable(
            Dx, w, self._alpha / self._rho)

        # 3) Update scaled dual variable
        w = w + Dx - v

        return x, v, w

    def _perform_ADMM_step_1_TK1_recon_solution(self, x, v, w, rho):

        b_reg = v - w

        tikhonov = TikhonovLinearSolver(
            A=self._A, A_adj=self._A_adj,
            B=self._D, B_adj=self._D_adj,
            b=self._b, b_reg=b_reg,
            alpha=rho,
            x0=x,
            iter_max=self._iter_max)
        tikhonov.run()
        tikhonov.print_statistics()

        return tikhonov.get_x()

    def _perform_ADMM_step_2_auxiliary_variable(self, Dx, w, ell):

        v = np.zeros_like(Dx)

        # Compute t = Dx + w
        t = Dx + w
        t_split = self._get_split(t)
        t_norm = np.sqrt(self._get_squared_sum_of_split(t_split))

        ind = t_norm > ell

        m = t_split[0].shape[0]
        for i in range(0, self._dimension):
            v_tmp = v[i*m:(i+1)*m, ...]
            v_tmp[ind] = self._get_soft_threshold(ell, t_norm[ind]) * \
                t_split[i][ind] / t_norm[ind]
            v[i*m:(i+1)*m, ...] = v_tmp

        return v

    def _get_soft_threshold(self, ell, t):
        return np.maximum(np.abs(t) - ell, 0)*np.sign(t)

    def _get_cost_regularization_term(self, x):

        Dx_split = self._get_split(self._D(x))
        sum_Dx_i_squared = self._get_squared_sum_of_split(Dx_split)

        return np.sum(np.sqrt(sum_Dx_i_squared))

    def _get_split(self, x):
        x_split = np.array_split(x, self._dimension)
        return x_split

    def _get_squared_sum_of_split(self, x_split):

        # x_i ** 2
        tmp = x_split[0] ** 2

        # Compute sum_k x_k^2
        for i in range(1, self._dimension):
            tmp += x_split[i]**2

        return tmp
