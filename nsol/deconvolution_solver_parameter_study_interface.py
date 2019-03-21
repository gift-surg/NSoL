##
# \file InterfaceDeconvolutionParameterStudy.py
# \brief      Common interface to run deconvolution reconstruction and
#             parameter study
#
# Idea is to provide a simple interface so that the solver/parameter study for
# the deconvolution problem ||Ax - b|| + alpha g(x) can be performed in any
# additional project outside nsol for the reconstruction methods:
#   -# TK0L2: least-squares optimization with zeroth-order Tikhonov reg.
#   -# TK1L2: least-squares optimization with first-order Tikhonov reg.
#   -# TVL2: least-squares optimization with isotropic total variation reg.
#   -# HuberL2: least-squares optimization with Huber norm reg.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Sept 2017
#

import os
import sys
import scipy.io
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import nsol.tikhonov_linear_solver as tk
import nsol.admm_linear_solver as admm
import nsol.primal_dual_solver as pd
import nsol.observer as Observer
import nsol.tikhonov_linear_solver_parameter_study as tkparam
import nsol.primal_dual_solver_parameter_study as pdparam
import nsol.admm_linear_solver_parameter_study as admmparam
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures
from nsol.proximal_operators import ProximalOperators as prox
from nsol.prior_measures import PriorMeasures as prior_meas
from nsol.loss_functions import LossFunctions as loss_fun


##
# Interface to get desired TK0L2/TK1L2/TVL2/HuberL2 solver and associated
# (similarity) measures for facilitated analysis.
# \date       2017-09-13 21:24:08+0100
#
class DeconvolutionSolverStudyInterface(object):

    ##
    # Store parameters to get solver.
    #
    # Note that not all variables are used for all solvers! E.g. data_loss, rho
    # \date       2017-09-13 21:29:01+0100
    #
    # \param      self                 The object
    # \param      A                    Function associated with linear operator
    #                                  A: X->Y; x -> A(x) with x being a 1D
    #                                  numpy array
    # \param      A_adj                Function associated with adjoint linear
    #                                  operator A^*: Y -> X; y -> A^*(y)
    # \param      D                    Function associated with the
    #                                  differential/gradient operator D: X->Z;
    #                                  x->D(x) with x being a 1D numpy array
    # \param      D_adj                Function associated with adjoint linear
    #                                  operator B^*: Z->X; z->B^*(z)
    # \param      b                    Right hand-side of linear system Ax = b
    #                                  as 1D numpy array
    # \param      x0                   Initial value as 1D numpy array
    # \param      alpha                The alpha
    # \param      x_scale              Characteristic scale of each variable.
    #                                  Setting x_scale is equivalent to
    #                                  reformulating the problem in scaled
    #                                  variables ``xs = x / x_scale``
    # \param      iter_max             The iterator maximum
    # \param      iterations           The iterations
    # \param      minimizer            String defining the used optimizer, i.e.
    #                                  "lsmr", "least_squares" or any solver as
    #                                  provided by scipy.optimize.minimize
    # \param      measures             The measures
    # \param      reconstruction_type  The reconstruction type
    # \param      dimension            The dimension
    # \param      rho                  regularization parameter of augmented
    #                                  Lagrangian term; scalar > 0
    # \param      x_ref                Reference data array; required for the
    #                                  evaluation of (optional) similarity
    #                                  measures
    # \param      data_loss            Data loss function rho specified as
    #                                  string, e.g. "linear", "soft_l1",
    #                                  "huber", "cauchy", "arctan".
    # \param      data_loss_scale      Value of soft margin between inlier and
    #                                  outlier residuals, default is 1.0. The
    #                                  loss function is evaluated as rho_(f2) =
    #                                  C**2 * rho(f2 / C**2), where C is
    #                                  data_loss_scale. This parameter has no
    #                                  effect with data_loss='linear', but for
    #                                  other loss values it is of crucial
    #                                  importance.
    # \param      tv_solver            Decide on the used TV solver. Either
    #                                  "PD" (Primal-Dual) or "ADMM"
    # \param      verbose              Verbose output, bool
    #
    def __init__(self,
                 A,
                 A_adj,
                 D,
                 D_adj,
                 b,
                 x0,
                 alpha,
                 x_scale,
                 iter_max,
                 iterations,
                 minimizer,
                 measures,
                 reconstruction_type,
                 dimension,
                 L2=8,
                 rho=0.5,
                 x_ref=None,
                 x_ref_mask=None,
                 data_loss="linear",
                 data_loss_scale=1,
                 tv_solver="PD",
                 verbose=0,
                 append=0,
                 ):

        self._A = A
        self._A_adj = A_adj
        self._D = D
        self._D_adj = D_adj
        self._b = b
        self._x0 = x0
        self._alpha = alpha
        self._data_loss = data_loss
        self._data_loss_scale = data_loss_scale
        self._x_scale = x_scale
        self._iter_max = iter_max
        self._iterations = iterations
        self._minimizer = minimizer
        self._measures = measures
        self._reconstruction_type = reconstruction_type
        self._x_ref = x_ref
        self._x_ref_mask = x_ref_mask
        self._dimension = dimension
        self._tv_solver = tv_solver
        self._L2 = L2
        self._rho = rho
        self._verbose = verbose

        # Append (instead of overwrite) potentially existing study
        self._append = append

        self._set_up_solver = {
            "TK0L2": self._set_up_solver_TK0L2,
            "TK1L2": self._set_up_solver_TK1L2,
            "TVL2": self._set_up_solver_TVL2,
            "HuberL2": self._set_up_solver_HuberL2,
        }

        self._append_reg_and_data_costs = {
            "TK0L2": self._append_reg_and_data_costs_TK0L2,
            "TK1L2": self._append_reg_and_data_costs_TK1L2,
            "TVL2": self._append_reg_and_data_costs_TVL2,
            "HuberL2": self._append_reg_and_data_costs_HuberL2,
        }

    def set_up_solver(self):
        self._solver = self._set_up_solver[self._reconstruction_type]()

    def set_up_measures(self):

        # Add all selected measures and append reg./prior and data costs
        if self._x_ref is not None:

            if not isinstance(self._x_ref, np.ndarray):
                raise ValueError("Reference x_ref must be of type 1D np.array")

            if self._x_ref.shape != self._x0.shape:
                raise ValueError(
                    "Initial value x0 and reference x_ref arrays "
                    "must be of same shape")

            # Reduce evaluation to mask
            if self._x_ref_mask is not None:
                if self._x_ref.shape != self._x_ref_mask.shape:
                    raise ValueError(
                        "Reference x_ref and reference mask x_ref_mask arrays "
                        "must be of same shape")
                indices = np.where(self._x_ref_mask > 0)

            # Evaluate on entire array
            else:
                indices = np.where(self._x_ref != np.inf)

            measures_dic = {
                m: lambda x, m=m:
                SimilarityMeasures.similarity_measures[m](
                    x[indices], self._x_ref[indices])
                for m in self._measures}
        else:
            measures_dic = {}
        self._append_reg_and_data_costs[self._reconstruction_type](
            measures_dic)

        self._measures_dic = measures_dic

    def get_solver(self):
        if self._solver is None:
            raise RuntimeError("Run 'set_up_solver' first")
        return self._solver

    def get_measures(self):
        if self._measures_dic is None:
            raise RuntimeError("Run 'set_up_measures' first")
        return self._measures_dic

    def _set_up_solver_TK0L2(self):

        solver = tk.TikhonovLinearSolver(
            A=self._A,
            A_adj=self._A_adj,
            B=lambda x: x.flatten(),
            B_adj=lambda x: x.flatten(),
            b=self._b,
            alpha=self._alpha,
            x0=self._x0,
            x_scale=self._x_scale,
            data_loss=self._data_loss,
            data_loss_scale=self._data_loss_scale,
            iter_max=self._iter_max,
            minimizer=self._minimizer,
            verbose=self._verbose,
        )
        return solver

    def _set_up_solver_TK1L2(self):

        solver = tk.TikhonovLinearSolver(
            A=self._A,
            A_adj=self._A_adj,
            B=self._D,
            B_adj=self._D_adj,
            b=self._b,
            alpha=self._alpha,
            x0=self._x0,
            x_scale=self._x_scale,
            data_loss=self._data_loss,
            data_loss_scale=self._data_loss_scale,
            iter_max=self._iter_max,
            minimizer=self._minimizer,
            verbose=self._verbose,
        )
        return solver

    def _set_up_solver_TVL2(self):

        if self._tv_solver == "PD":
            prox_f = lambda x, tau: prox.prox_linear_least_squares(
                x=x, tau=tau,
                A=self._A,
                A_adj=self._A_adj,
                b=self._b,
                x0=self._x0,
                iter_max=self._iter_max,
                data_loss=self._data_loss,
                data_loss_scale=self._data_loss_scale,
                x_scale=self._x_scale)
            solver = pd.PrimalDualSolver(
                prox_f=prox_f,
                prox_g_conj=prox.prox_tv_conj,
                B=self._D,
                B_conj=self._D_adj,
                L2=self._L2,
                alpha=self._alpha,
                x0=self._x0,
                iterations=self._iterations,
                # alg_type=alg_type,
                x_scale=self._x_scale,
                verbose=self._verbose,
            )

        elif self._tv_solver == "ADMM":
            solver = admm.ADMMLinearSolver(
                A=self._A,
                A_adj=self._A_adj,
                b=self._b,
                B=self._D,
                B_adj=self._D_adj,
                alpha=self._alpha,
                x0=self._x0,
                x_scale=self._x_scale,
                data_loss=self._data_loss,
                data_loss_scale=self._data_loss_scale,
                rho=self._rho,
                iterations=self._iterations,
                dimension=self._dimension,
                iter_max=self._iter_max,
                verbose=self._verbose,
            )

        return solver

    def _set_up_solver_HuberL2(self):
        prox_f = lambda x, tau: prox.prox_linear_least_squares(
            x=x, tau=tau,
            A=self._A,
            A_adj=self._A_adj,
            b=self._b,
            x0=self._x0,
            iter_max=self._iter_max,
            x_scale=self._x_scale)
        solver = pd.PrimalDualSolver(
            prox_f=prox_f,
            prox_g_conj=prox.prox_huber_conj,
            B=self._D,
            B_conj=self._D_adj,
            L2=self._L2,
            alpha=self._alpha,
            x0=self._x0,
            iterations=self._iterations,
            # alg_type=alg_type,
            x_scale=self._x_scale,
            verbose=self._verbose,
        )
        return solver

    def _append_reg_and_data_costs_TK0L2(self, measures_dic):
        measures_dic["Reg"] = \
            lambda x: prior_meas.zeroth_order_tikhonov(x)
        measures_dic["Data"] = \
            lambda x: loss_fun.get_ell2_cost_from_residual(
                self._A(x) - self._b,
                loss=self._data_loss,
                f_scale=self._data_loss_scale)

    def _append_reg_and_data_costs_TK1L2(self, measures_dic):
        measures_dic["Reg"] = \
            lambda x: prior_meas.first_order_tikhonov(x, self._D)
        measures_dic["Data"] = \
            lambda x: loss_fun.get_ell2_cost_from_residual(
                self._A(x) - self._b,
                loss=self._data_loss,
                f_scale=self._data_loss_scale)

    def _append_reg_and_data_costs_TVL2(self, measures_dic):
        measures_dic["Reg"] = \
            lambda x: prior_meas.total_variation(x, self._D, self._dimension)
        measures_dic["Data"] = \
            lambda x: loss_fun.get_ell2_cost_from_residual(
                self._A(x) - self._b,
                loss=self._data_loss,
                f_scale=self._data_loss_scale)

    def _append_reg_and_data_costs_HuberL2(self, measures_dic):
        measures_dic["Reg"] = \
            lambda x: prior_meas.huber(x, self._D, self._dimension)
        measures_dic["Data"] = \
            lambda x: loss_fun.get_ell2_cost_from_residual(
                self._A(x) - self._b,
                loss=self._data_loss,
                f_scale=self._data_loss_scale)


class DeconvolutionParameterStudyInterface(DeconvolutionSolverStudyInterface):

    ##
    # Store parameters to get solver
    # \date       2017-09-13 21:29:01+0100
    #
    # \param      self                 The object
    # \param      A                    Function associated with linear operator
    #                                  A: X->Y; x -> A(x) with x being a 1D
    #                                  numpy array
    # \param      A_adj                Function associated with adjoint linear
    #                                  operator A^*: Y -> X; y -> A^*(y)
    # \param      D                    Function associated with the
    #                                  differential/gradient operator D: X->Z;
    #                                  x->D(x) with x being a 1D numpy array
    # \param      D_adj                Function associated with adjoint linear
    #                                  operator B^*: Z->X; z->B^*(z)
    # \param      b                    Right hand-side of linear system Ax = b
    #                                  as 1D numpy array
    # \param      x0                   Initial value as 1D numpy array
    # \param      alpha                The alpha
    # \param      x_scale              Characteristic scale of each variable.
    #                                  Setting x_scale is equivalent to
    #                                  reformulating the problem in scaled
    #                                  variables ``xs = x / x_scale``
    # \param      iter_max             The iterator maximum
    # \param      iterations           The iterations
    # \param      minimizer            String defining the used optimizer, i.e.
    #                                  "lsmr", "least_squares" or any solver as
    #                                  provided by scipy.optimize.minimize
    # \param      measures             The measures
    # \param      dimension            The dimension
    # \param      reconstruction_type  The reconstruction type
    # \param      dir_output           The dir output
    # \param      parameters           The parameters
    # \param      name                 The name
    # \param      reconstruction_info  The reconstruction information
    # \param      rho                  regularization parameter of augmented
    #                                  Lagrangian term; scalar > 0
    # \param      x_ref                Reference data array; required for the
    #                                  evaluation of (optional) similarity
    #                                  measures
    # \param      data_loss            Data loss function rho specified as
    #                                  string, e.g. "linear", "soft_l1",
    #                                  "huber", "cauchy", "arctan".
    # \param      data_loss_scale      Value of soft margin between inlier and
    #                                  outlier residuals, default is 1.0. The
    #                                  loss function is evaluated as rho_(f2) =
    #                                  C**2 * rho(f2 / C**2), where C is
    #                                  data_loss_scale. This parameter has no
    #                                  effect with data_loss='linear', but for
    #                                  other loss values it is of crucial
    #                                  importance.
    # \param      tv_solver            Decide on the used TV solver. Either
    #                                  "PD" (Primal-Dual) or "ADMM"
    # \param      verbose              Verbose output, bool
    #
    def __init__(self,
                 A,
                 A_adj,
                 D,
                 D_adj,
                 b,
                 x0,
                 alpha,
                 x_scale,
                 iter_max,
                 iterations,
                 minimizer,
                 measures,
                 dimension,
                 reconstruction_type,
                 dir_output,
                 parameters,
                 name,
                 reconstruction_info,
                 L2=8,
                 rho=0.5,
                 x_ref=None,
                 x_ref_mask=None,
                 data_loss="linear",
                 data_loss_scale=1,
                 tv_solver="PD",
                 verbose=0,
                 append=False,
                 ):

        super(self.__class__, self).__init__(
            A=A,
            A_adj=A_adj,
            D=D,
            D_adj=D_adj,
            b=b,
            x0=x0,
            alpha=alpha,
            data_loss=data_loss,
            data_loss_scale=data_loss_scale,
            x_scale=x_scale,
            iter_max=iter_max,
            iterations=iterations,
            minimizer=minimizer,
            measures=measures,
            reconstruction_type=reconstruction_type,
            L2=L2,
            rho=rho,
            x_ref=x_ref,
            x_ref_mask=x_ref_mask,
            dimension=dimension,
            tv_solver=tv_solver,
            verbose=verbose,
            append=append,
        )
        self._name = name
        self._parameters = parameters
        self._reconstruction_info = reconstruction_info
        self._dir_output = dir_output

        self._solver = None
        self._parameter_study = None

        self._set_up_parameter_study = {
            "TK0L2": self._set_up_parameter_study_TKL2,
            "TK1L2": self._set_up_parameter_study_TKL2,
            "TVL2": self._set_up_parameter_study_TVL2,
            "HuberL2": self._set_up_parameter_study_HuberL2,
        }

    def set_up_parameter_study(self):
        self.set_up_solver()
        self.set_up_measures()

        observer = Observer.Observer()
        observer.set_measures(self._measures_dic)

        self._parameter_study = self._set_up_parameter_study[self._reconstruction_type](
            self._solver, observer,
        )

    def get_parameter_study(self):
        return self._parameter_study

    # def run(self):
    #     self._parameter_study.run()
    #     ph.print_info("Computational Time for Parameter Study: %s" %
    #                   self._parameter_study.get_computational_time())

    def _set_up_parameter_study_TKL2(self, solver, observer):
        parameter_study = tkparam.TikhonovLinearSolverParameterStudy(
            solver, observer,
            dir_output=self._dir_output,
            parameters=self._parameters,
            name=self._name,
            reconstruction_info=self._reconstruction_info,
            append=self._append,
        )
        return parameter_study

    def _set_up_parameter_study_TVL2(self, solver, observer):

        if self._tv_solver == "PD":
            parameter_study = pdparam.PrimalDualSolverParameterStudy(
                solver, observer,
                dir_output=self._dir_output,
                parameters=self._parameters,
                name=self._name,
                reconstruction_info=self._reconstruction_info,
                append=self._append,
            )
        elif self._tv_solver == "ADMM":
            parameter_study = admmparam.ADMMLinearSolverParameterStudy(
                solver, observer,
                dir_output=self._dir_output,
                parameters=self._parameters,
                name=self._name,
                reconstruction_info=self._reconstruction_info,
                append=self._append,
            )
        return parameter_study

    def _set_up_parameter_study_HuberL2(self, solver, observer):
        parameter_study = pdparam.PrimalDualSolverParameterStudy(
            solver, observer,
            dir_output=self._dir_output,
            parameters=self._parameters,
            name=self._name,
            reconstruction_info=self._reconstruction_info,
            append=self._append,
        )
        return parameter_study
