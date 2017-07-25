##
# \file PrimalDualSolver.py
# \brief      Implementation of First-order primal-dual algorithm for convex
#             problems min_x [f(x) + alpha g(Bx)] with B being a continuous
#             linear operator.
#
# First-order primal-dual algorithm for convex problems as introduced in
# Chambolle, A. & Pock, T., 2011. A First-Order Primal-Dual Algorithm for
# Convex Problems with Applications to Imaging. Journal of Mathematical Imaging
# and Vision, 40(1), pp.120-145.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import numpy as np

from numericalsolver.Solver import Solver
import pythonhelper.PythonHelper as ph


##
# First-order primal-dual algorithm for convex problems
# min_x [f(x) + alpha g(Bx)] with B being a continuous linear operator.
#
class PrimalDualSolver(Solver):

    ##
    # Store all essential variables
    # \date       2017-07-20 22:30:07+0100
    #
    # \param      self         The object
    # \param      prox_f       Proximal operator of f; prox_f: (x, tau) ->
    #                          prox_f(x, tau) = min_y [1/2 ||x-y||^2 + tau
    #                          f(y)]
    # \param      prox_g_conj  Proximal operator of g' with g' being the
    #                          conjugate of a convex, lower-semicontinuous
    #                          function g; typically g acts as regularizer like
    #                          TV or Huber; prox_g_conj: (x, sigma) ->
    #                          prox_g_conj(x, sigma) = min_y [1/2 ||x-y||^2 +
    #                          sigma g'(y)]
    # \param      B            Function associated to continuous linear
    #                          operator B
    # \param      B_conj       The conjugate of the continuous linear operator
    #                          B
    # \param      L2           Squared operator norm of B, i.e. L2 = ||B||^2
    # \param      x0           Initial value, 1D numpy data array
    # \param      alpha        Regularization parameter alpha > 0.
    # \param      iterations   The iterations
    #
    def __init__(self,
                 prox_f,
                 prox_g_conj,
                 B,
                 B_conj,
                 L2,
                 x0,
                 alpha,
                 iterations=10,
                 verbose=0,
                 alg_type="ALG2",
                 ):

        Solver.__init__(self, x0=x0, verbose=verbose)

        # proximal operator of f
        self._prox_f = prox_f

        # proximal operator of g'
        self._prox_g_conj = prox_g_conj

        # Continuous linear operator B in regularizer term g(Bx)
        self._B = B

        # Conjugate operator of B, i.e. B'
        self._B_conj = B_conj

        # Squared operator norm of B, i.e. L2 = ||B||^2
        self._L2 = float(L2)

        # Regularization parameter in f(x) + alpha g(Bx)
        self._alpha = float(alpha)

        # Number of primal-dual iterations
        self._iterations = iterations

        self._alg_type = alg_type

        # parameter initialization depend on chosen method
        self._get_initial_tau_sigma = {
            "ALG2_AHMOD": self._get_initial_tau_sigma_alg2_ahmod,
            "ALG2": self._get_initial_tau_sigma_alg2,
            "ALG3": self._get_initial_tau_sigma_alg3,
        }

        # parameter updates depend on chosen method
        self._get_update_theta_tau_sigma = {
            "ALG2_AHMOD": self._get_update_theta_tau_sigma_alg2_ahmod,
            "ALG2": self._get_update_theta_tau_sigma_alg2,
            "ALG3": self._get_update_theta_tau_sigma_alg3,
        }

    ##
    # Prints the statistics of the performed optimization
    # \date       2017-07-21 00:01:10+0100
    #
    # \param      self  The object
    # \param      fmt   Format for printing numerical values
    #
    def print_statistics(self, fmt="%.3e"):
        pass

    def _run(self):

        # Monitor output
        if self._monitor is not None:
            self._monitor.add_x(self._x)

        # regularization parameter lambda as used in Chambolle2011
        lmbda = 1. / self._alpha

        # Dynamic step sizes for primal and dual variable, see p.127
        tau_n, sigma_n, gamma = self._get_initial_tau_sigma[
            self._alg_type](L2=self._L2, lmbda=lmbda)

        x_n = np.array(self._x0)
        x_mean = np.array(self._x0)
        p_n = 0

        for i in range(0, self._iterations):

            if self._verbose:
                ph.print_title("Primal-Dual iteration %d/%d" %
                               (i+1, self._iterations))
            else:
                ph.print_info("Primal-Dual iteration %d/%d" %
                              (i+1, self._iterations))

            # Update dual variable
            p_n = self._prox_g_conj(
                p_n + sigma_n * self._B(x_mean), sigma_n)

            # Update primal variable
            x_np1 = self._prox_f(x_n - tau_n * self._B_conj(p_n), tau_n*lmbda)

            # Update parameter
            theta_n, tau_n, sigma_n = self._get_update_theta_tau_sigma[
                self._alg_type](self._L2, gamma, tau_n, sigma_n)

            # Update mean variable
            x_mean = x_np1 + theta_n * (x_np1 - x_n)

            # Prepare for next iteration
            x_n = x_np1

            # Monitor output
            if self._monitor is not None:
                self._monitor.add_x(x_n)

        self._x = x_n

    ##
    # Gets the initial step sizes tau_0, sigma_0 and the Lipschitz parameter
    # gamma according to ALG2 method in Chambolle2011, p.133
    #
    # tau_0 and sigma_0 such that tau_0 * sigma_0 * L^2 = 1
    # \date       2017-07-18 17:57:33+0100
    #
    # \param      self   The object
    # \param      L2     Squared operator norm
    # \param      lmbda  Regularization parameter
    #
    # \return     tau0, sigma0, gamma
    #
    def _get_initial_tau_sigma_alg2(self, L2, lmbda):
        # Initial values according to ALG2 in Chambolle2011
        tau0 = 1. / np.sqrt(L2)
        sigma0 = 1. / (L2 * tau0)
        gamma = 0.35 * lmbda
        return tau0, sigma0, gamma

    ##
    # Gets the update of the variable relaxation parameter
    # \f$\theta_n\in[0,1]\f$ and the dynamic step sizes
    # \f$\tau_n,\,\sigma_n>0\f$ for the primal and dual variable, respectively.
    #
    # Update is performed according to ALG2 in Chambolle2011, p.133. It always
    # holds tau_n * sigma_n * L^2 = 1.
    # \date       2017-07-18 18:16:28+0100
    #
    # \param      self     The object
    # \param      L2       Squared operator norm
    # \param      gamma    Lipschitz parameter
    # \param      tau_n    Dynamic step size for primal variable
    # \param      sigma_n  Dynamic step size for dual variable
    #
    # \return     theta_n, tau_n, sigma_n update
    #
    def _get_update_theta_tau_sigma_alg2(self, L2, gamma, tau_n, sigma_n):
        theta_n = 1. / np.sqrt(1. + 2. * gamma * tau_n)
        tau_n = tau_n * theta_n
        sigma_n = sigma_n / theta_n
        return theta_n, tau_n, sigma_n

    ##
    # Gets the initial step sizes tau_0, sigma_0 and the Lipschitz parameter
    # gamma according to ALG2 method in Chambolle2011, p.136
    #
    # tau_0 and sigma_0 such that tau_0 * sigma_0 * L^2 = 1
    # \date       2017-07-18 17:57:33+0100
    #
    # \param      self   The object
    # \param      L2     Squared operator norm
    # \param      lmbda  Regularization parameter
    #
    # \return     tau0, sigma0, theta
    #
    def _get_initial_tau_sigma_alg3(self, L2, lmbda, huber_alpha=0.05):

        # Initial values according to ALG3 in Chambolle2011
        gamma = lmbda
        delta = huber_alpha
        mu = 2. * np.sqrt(gamma * delta / L2)

        # relaxation parameter in [1/(1+mu), 1]
        theta = 1. / (1. + mu)

        # step size dual variable
        sigma = mu / (2. * delta)

        # step size primal variable
        tau = mu / (2. * gamma)

        return tau, sigma, theta

    ##
    # Gets the update of the variable relaxation parameter
    # \f$\theta_n\in[0,1]\f$ and the dynamic step sizes
    # \f$\tau_n,\,\sigma_n>0\f$ for the primal and dual variable, respectively.
    #
    # Update is performed according to ALG2 in Chambolle2011, p.136. It always
    # holds tau_n * sigma_n * L^2 = 1.
    # \date       2017-07-18 18:16:28+0100
    #
    # \param      self     The object
    # \param      L2       Squared operator norm
    # \param      gamma    Lipschitz parameter
    # \param      tau_n    Dynamic step size for primal variable
    # \param      sigma_n  Dynamic step size for dual variable
    #
    # \return     theta_n, tau_n, sigma_n update
    #
    def _get_update_theta_tau_sigma_alg3(self, L2, gamma, tau_n, sigma_n):
        theta_n = gamma  # gamma is used as place holder for tau
        return theta_n, tau_n, sigma_n

    ##
    # Gets the initial step sizes tau_0, sigma_0 and the Lipschitz parameter
    # gamma according to AHMOD, i.e. Arrow-Hurwicz method, in Chambolle2011,
    # p.133
    #
    # tau_0 and sigma_0 such that tau_0 * sigma_0 * L^2 = 4
    # \date       2017-07-18 17:56:36+0100
    #
    # \param      self   The object
    # \param      L2     Squared operator norm
    # \param      lmbda  Regularization parameter
    #
    # \return     tau0, sigma0, gamma
    #
    def _get_initial_tau_sigma_alg2_ahmod(self, L2, lmbda):
        # Initial values according to AHMOD in Chambolle2011
        tau0 = 0.02
        sigma0 = 4. / (L2 * tau0)
        gamma = 0.35 * lmbda
        return tau0, sigma0, gamma

    ##
    # Gets the update of the variable relaxation parameter
    # \f$\theta_n\in[0,1]\f$ and the dynamic step sizes
    # \f$\tau_n,\,\sigma_n>0\f$ for the primal and dual variable, respectively.
    #
    # Update is performed according to AHMOD, i.e. Arrow-Hurwicz method, in
    # Chambolle2011, p.133. It always holds tau_n * sigma_n * L^2 = 4.
    # \date       2017-07-18 18:16:28+0100
    #
    # \param      self     The object
    # \param      L2       Squared operator norm
    # \param      gamma    Lipschitz parameter
    # \param      tau_n    Dynamic step size for primal variable
    # \param      sigma_n  Dynamic step size for dual variable
    #
    # \return     theta_n, tau_n, sigma_n update
    #
    def _get_update_theta_tau_sigma_alg2_ahmod(self,
                                               L2, gamma, tau_n, sigma_n):
        theta_n = 1. / np.sqrt(1. + 2. * gamma * tau_n)
        tau_n = tau_n * theta_n
        sigma_n = sigma_n / theta_n
        return 0., tau_n, sigma_n
