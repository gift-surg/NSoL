##
# \file proximalOperators.py
# \brief      Collection of proximal operators for primal and dual variables
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import numpy as np

import nsol.tikhonov_linear_solver as tk


class ProximalOperators(object):

    # #########################################################################
    # ------------------------------Primal Variables--------------------------

    ##
    # Proximal operator for g(x) = 1/2 ||Ax-b||^2
    #
    # Approximation provided by solving the minimization problem via Tikhonov
    # solver
    # \date       2017-07-23 01:32:15+0100
    #
    # \param      x         1D numpy array
    # \param      tau       step size; scalar >= 0
    # \param      A         Function handle lambda x: A(x) with x being 1D
    #                       numpy array
    # \param      A_adj     Function handle for adjoint operator lambda x:
    #                       A^*(x)
    # \param      b         1D numpy array
    # \param      x0        initial value for minimization problem as 1D numpy
    #                       array
    # \param      iter_max  The iterator maximum
    # \param      verbose   Verbose output; bool
    # \param      x_scale   Characteristic scale of each variable. Setting
    #                       x_scale is equivalent to reformulating the problem
    #                       in scaled variables ``xs = x / x_scale``
    #
    # \return     prox_g(x, tau) as numpy array
    #
    @staticmethod
    def prox_linear_least_squares(x,
                                  tau,
                                  A,
                                  A_adj,
                                  b,
                                  x0,
                                  iter_max=10,
                                  verbose=0,
                                  data_loss="linear",
                                  data_loss_scale=1,
                                  minimizer="lsmr",
                                  x_scale=1,
                                  bounds=(0, np.inf),
                                  ):

        # Identity
        I = lambda x: x.flatten()

        tikhonov = tk.TikhonovLinearSolver(
            A=A, A_adj=A_adj,
            B=I, B_adj=I,
            x0=x0/float(x_scale),
            b=b/float(x_scale),
            b_reg=x,
            alpha=1./tau,
            iter_max=iter_max,
            verbose=verbose,
            x_scale=x_scale,
            data_loss=data_loss,
            data_loss_scale=data_loss_scale,
            minimizer=minimizer,
            bounds=bounds,
        )
        tikhonov.run()
        return tikhonov.get_x()

    ##
    # Proximal operator for ell1 denoising, i.e. g(x) = ||x - x0||_1.
    #
    # See Chambolle2011, p.135
    # \date       2017-07-21 01:52:01+0100
    #
    # \param      x        numpy array
    # \param      tau      step size; scalar >= 0
    # \param      x0       observed; numpy array
    # \param      x_scale  Characteristic scale of each variable. Setting
    #                      x_scale is equivalent to reformulating the problem
    #                      in scaled variables ``xs = x / x_scale``
    #
    # \return     prox_{\tau g}(x) as numpy array
    #
    @staticmethod
    def prox_ell1_denoising(x, tau, x0, x_scale=1.):
        x0 = x0 / float(x_scale)
        return x0 + np.maximum(np.abs(x-x0) - tau, 0) * np.sign(x-x0)

    ##
    # Proximal operator for least squares denoising, i.e. g(x) = 1/2 ||x -
    # x0||_2^2
    #
    # With g(x) = 1/2 * ||x - x0||_2^2 it holds that prox_{\tau g}(x) = (x +
    # \tau*x0) / (1 + \tau*x0). See Chambolle2011, p.133
    # \date       2017-07-21 01:52:01+0100
    #
    # \param      x        numpy array
    # \param      tau      step size; scalar >= 0
    # \param      x0       observed; numpy array
    # \param      x_scale  Characteristic scale of each variable. Setting
    #                      x_scale is equivalent to reformulating the problem
    #                      in scaled variables ``xs = x / x_scale``
    #
    # \return     prox_{\tau g}(x) as numpy array
    #
    @staticmethod
    def prox_ell2_denoising(x, tau, x0, x_scale=1.):
        x0 = x0 / float(x_scale)
        return (x + tau * x0) / (1. + tau)

    # #########################################################################
    # -------------------------------Dual Variables---------------------------

    ##
    # Proximal operator for conjugate of TV functional.
    #
    # With g(x) = TV(x) = || ||\nabla x||_2 ||_1 the conjugate operator reads
    # g'(p) = \delta_P(p)
    # See Chambolle2011, p.133
    # \date       2017-07-23 17:58:04+0100
    #
    # \param      x      numpy array
    # \param      sigma  step size dual variable; scalar >= 0
    #
    # \return     prox_{\sigma g'}(x) as numpy array
    #
    @staticmethod
    def prox_tv_conj(x, sigma):
        return x / np.maximum(1, np.abs(x))

    ##
    # Proximal operator for conjugate of Huber functional.
    #
    # With g(x) = | ||\nabla x||_2 |_\gamma the conjugate operator reads
    # g'(p) = \delta_P(p) + gamma/2 ||p||_2^2
    # See Chambolle2011. p.136
    # \date       2017-07-23 18:03:56+0100
    #
    #
    # \param      x      numpy array
    # \param      sigma  step size dual variable; scalar >= 0
    #
    # \return     prox_{\sigma g'}(x) as numpy array
    #
    @staticmethod
    def prox_huber_conj(x, sigma, gamma=0.05):
        x /= (1. + sigma * gamma)
        return x / np.maximum(1, np.abs(x))
