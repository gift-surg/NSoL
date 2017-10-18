##
# \file optimize.py
# \brief      Class with static functions to wrap the functionality of scipy
#             optimizers
#
# Not used for now. Might be relevant at some point if interfaces from one
# solver to the other shall be wrapped. E.g. convert from residuals
# (least_squares) to cost functions (minimize) etc
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import scipy
import numpy as np

class Optimize(object):

    ##
    # Gets the approximate solution lsmr (Linear least-squares method)
    # \date       2017-07-24 10:24:36+0100
    #
    # \param      self      The object
    # \param      A         {matrix, sparse matrix, ndarray, LinearOperator}
    #                       Matrix A in the linear system.
    # \param      b         array_like, shape (m,) Vector b in the linear
    #                       system.
    # \param      max_iter  Maximum number of iterations, int
    # \param      verbose   verbose output, bool
    # \param      conlim    condition limit, float; `lsmr` terminates if an
    #                       estimate of ``cond(A)`` exceeds `conlim`.
    # \param      atol      Stopping tolerances, float
    # \param      btol      Stopping tolerances, float
    # \param      damp      Damping factor for regularized least-squares, float
    #
    # \return     The approximate solution as 1D numpy array.
    #
    @staticmethod
    def get_approximate_solution_lsmr(A, b,
                                      max_iter=None,
                                      verbose=True,
                                      conlim=1e8,
                                      atol=1e-6,
                                      btol=1e-6,
                                      damp=0.0,
                                      ):

        x = scipy.sparse.linalg.lsmr(
            A, b,
            maxiter=max_iter,
            show=verbose,
            atol=atol,
            btol=btol,
            conlim=conlim,
            damp=damp
        )[0]

        return x

    ##
    # Gets the approximate solution lsq linear (linear least-squares problem
    # with bounds)
    # \date       2017-07-24 10:33:24+0100
    #
    # \param      self        The object
    # \param      A           array_like, sparse matrix of LinearOperator,
    #                         shape (m, n)
    # \param      b           array_like, shape (m,)
    # \param      bounds      2-tuple of array_like
    # \param      method      Method to perform minimization. 'trf' or 'bvls'
    # \param      tol         tolerance, float
    # \param      lsq_solver  {None, 'exact', 'lsmr'}
    # \param      lsmr_tol    Tolerance parameters 'atol' and 'btol' for
    #                         `scipy.sparse.linalg.lsmr`
    # \param      max_iter    Maximum number of iterations before termination,
    #                         int
    # \param      verbose     Verbose output, bool
    #
    # \return     The approximate solution as 1D numpy array of shape (m,).
    #
    @staticmethod
    def get_approximate_solution_lsq_linear(A, b,
                                            bounds=(-np.inf, np.inf),
                                            method="trf",
                                            tol=1e-10,
                                            lsq_solver=None,
                                            lsmr_tol=None,
                                            max_iter=None,
                                            verbose=2,
                                            ):
        res = scipy.optimize.lsq_linear(
            A, b,
            bounds=bounds,
            method=method,
            tol=tol,
            lsq_solver=lsq_solver,
            lsmr_tol=lsmr_tol,
            max_iter=max_iter,
            verbose=verbose
        )

        return res.x

    ##
    # Gets the approximate solution nnls (FORTRAN non-negative least squares
    # solver).
    # \date       2017-07-24 10:48:06+0100
    #
    # \param      self  The object
    # \param      A     numpy array of shape (m, n)
    # \param      b     numpy array of shape (m,)
    #
    # \return     The approximate solution as 1D numpy array of shape (n,).
    #
    @staticmethod
    def get_approximate_solution_nnls(A, b):
        return scipy.optimize.nnls(A, b)

    ##
    # Gets the approximate solution least squares (non-linear least-squares
    # problem with bounds).
    # \date       2017-07-24 10:54:18+0100
    #
    # \param      self          The object
    # \param      fun           Function which computes the vector of
    #                           residuals, callable
    # \param      x0            initial value, array_like
    # \param      jac           {'2-point', '3-point', 'cs', callable}. Method
    #                           of computing the Jacobian matrix (an m-by-n
    #                           matrix, where element (i, j) is the partial
    #                           derivative of f[i] with respect to x[j]),
    # \param      bounds        Lower and upper bounds on independent
    #                           variables, 2-tuple of array_like
    # \param      method        {'trf', 'dogbox', 'lm'}. Algorithm to perform
    #                           minimization
    # \param      ftol          float. Tolerance for termination by the change
    #                           of the cost function
    # \param      xtol          float. Tolerance for termination by the change
    #                           of the independent variables
    # \param      gtol          float. Tolerance for termination by the norm of
    #                           the gradient
    # \param      x_scale       array_like or 'jac'. Characteristic scale of
    #                           each variable
    # \param      loss          str or callable. Determines the loss function.
    #                           {'linear', 'soft_l1', 'huber', 'cauchy',
    #                           'arctan'}
    # \param      f_scale       float. Value of soft margin between inlier and
    #                           outlier residuals
    # \param      diff_step     None or array_like. Determines the relative
    #                           step size for the finite difference
    #                           approximation of the Jacobian
    # \param      tr_solver     {None, 'exact', 'lsmr'}. Method for solving
    #                           trust-region subproblems, relevant only for
    #                           'trf' and 'dogbox' methods
    # \param      tr_options    dict. Keyword options passed to trust-region
    #                           solver
    # \param      jac_sparsity  {None, array_like, sparse matrix}. Defines the
    #                           sparsity structure of the Jacobian matrix for
    #                           finite difference estimation, its shape must be
    #                           (m, n)
    # \param      max_nfev      None or int. Maximum number of function
    #                           evaluations before the termination
    # \param      verbose       Verbose output, bool
    # \param      args          tuple and dict. Additional arguments passed to
    #                           `fun` and `jac`
    # \param      kwargs        tuple and dict. Additional arguments passed to
    #                           `fun` and `jac`
    #
    # \return     The approximate solution least squares as 1D numpy array
    #
    @staticmethod
    def get_approximate_solution_least_squares(fun, x0,
                                               jac='2-point',
                                               bounds=(-np.inf, np.inf),
                                               method='trf',
                                               ftol=1e-8,
                                               xtol=1e-8,
                                               gtol=1e-8,
                                               x_scale=1.0,
                                               loss='linear',
                                               f_scale=1.0,
                                               diff_step=None,
                                               tr_solver=None,
                                               tr_options={},
                                               jac_sparsity=None,
                                               max_nfev=None,
                                               verbose=2,
                                               args=(),
                                               kwargs={}):

        res = scipy.optimize.least_squares(
            fun, x0,
            jac=jac,
            bounds=bounds,
            method=method,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            loss=loss,
            f_scale=f_scale,
            diff_step=diff_step,
            tr_solver=tr_solver,
            tr_options=tr_options,
            jac_sparsity=jac_sparsity,
            max_nfev=max_nfev,
            verbose=verbose,
            args=args,
            kwargs=kwargs,
        )

        return res.x

    ##
    # Gets the approximate solution minimize.
    # \date       2017-07-24 11:28:20+0100
    #
    # \param      self         The object
    # \param      fun          callable. Objective function
    # \param      x0           ndarray. Initial value.
    # \param      args         Tuple. Extra arguments passed to the objective
    #                          function and its derivatives (Jacobian, Hessian)
    # \param      method       str or callable.
    # \param      jac          bool or callable. Jacobian (gradient) of
    #                          objective function
    # \param      hess         callable, optional. Hessian (matrix of
    #                          second-order derivatives) of objective function
    #                          or Hessian of objective function times an
    #                          arbitrary vector p
    # \param      hessp        callable, optional. Hessian (matrix of
    #                          second-order derivatives) of objective function
    #                          or Hessian of objective function times an
    #                          arbitrary vector p
    # \param      bounds       sequence
    # \param      constraints  dict or sequence of dict. Constraints definition
    #                          (only for COBYLA and SLSQP)
    # \param      tol          float. Tolerance for termination
    # \param      callback     callable. Called after each iteration
    # \param      options      dict. A dictionary of solver options
    #
    # \return     The approximate solution minimize.
    #
    @staticmethod
    def get_approximate_solution_minimize(fun, x0,
                                          args=(),
                                          method=None,
                                          jac=None,
                                          hess=None,
                                          hessp=None,
                                          bounds=None,
                                          constraints=(),
                                          tol=None,
                                          callback=None,
                                          options=None,
                                          ):
        res = scipy.optimize.minimize(
            fun=fun,
            x0=x0,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
        )

        return res.x
