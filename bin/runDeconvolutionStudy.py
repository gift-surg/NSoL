#!/usr/bin/python

##
# \file runDenoisingStudy.py
# \brief      Run TK0L2/TK1L2/TVL2/HuberL2 deconvolution
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Sept 2017
#

import os
import sys
import scipy.io
import numpy as np
import SimpleITK as sitk

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

import numericalsolver.LinearOperators as LinearOperators
import numericalsolver.TikhonovLinearSolver as tk
import numericalsolver.ADMMLinearSolver as admm
import numericalsolver.PrimalDualSolver as pd
import numericalsolver.Observer as Observer
import numericalsolver.DataReader as dr
import numericalsolver.DataWriter as dw
import numericalsolver.TikhonovLinearSolverParameterStudy as tkparam
import numericalsolver.PrimalDualSolverParameterStudy as pdparam
import numericalsolver.ADMMLinearSolverParameterStudy as admmparam
from numericalsolver.SimilarityMeasures import SimilarityMeasures as \
    SimilarityMeasures
from numericalsolver.ProximalOperators import ProximalOperators as prox
from numericalsolver.PriorMeasures import PriorMeasures as prior_meas
import numericalsolver.InputArgparser as InputArgparser

if __name__ == '__main__':

    input_parser = InputArgparser.InputArgparser(
        description="Run TK0L2/TK1L2/TVL2/HuberL2 deconvolution",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_observation(required=True)
    input_parser.add_reference(required=False)
    input_parser.add_dir_output(required=True)
    input_parser.add_study_name()
    input_parser.add_reconstruction_type(default="TVL2")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_blur(default=1)
    input_parser.add_solver(default="PD")
    input_parser.add_iterations(default=50)
    input_parser.add_rho(default=0.1)
    input_parser.add_verbose(default=0)
    input_parser.add_iter_max(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_alpha_range(default=[0.001, 0.01, 0.001])
    input_parser.add_data_losses(
        # default=["linear", "arctan"]
    )
    input_parser.add_data_loss_scale_range(
        # default=[0.1, 1.5, 0.5]
    )
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    data_reader = dr.DataReader(args.observation)
    data_reader.read_data()
    observed_nda = data_reader.get_data()

    if args.reference is not None:
        data_reader = dr.DataReader(args.reference)
        data_reader.read_data()
        reference_nda = data_reader.get_data()
        x_ref = reference_nda.flatten()

    if args.blur > 0:
        sigma = np.atleast_1d(args.blur)
        if sigma.ndim != observed_nda.ndim:
            try:
                cov = np.diag(np.ones(observed_nda.ndim)) * sigma**2
            except:
                raise IOError(
                    "Blur information must be either 1- or d-dimensional")

    # ------------------------------Set Up Solver------------------------------
    dimension = observed_nda.ndim

    b = observed_nda.flatten()
    x0 = observed_nda.flatten()
    x_scale = np.max(observed_nda)

    # Get spacing for blurring operator
    if data_reader.get_image_sitk() is None:
        spacing = np.ones(observed_nda.ndim)
    else:
        spacing = np.array(data_reader.get_image_sitk().GetSpacing())

    # Get linear operators
    linear_operators = eval(
        "LinearOperators.LinearOperators%dD" % dimension)(spacing=spacing)
    A, A_adj = linear_operators.get_gaussian_blurring_operators(cov)
    grad, grad_adj = linear_operators.get_gradient_operators()

    # A: X \rightarrow Y and D: X \rightarrow Z
    X_shape = observed_nda.shape
    Y_shape = A(observed_nda).shape
    Z_shape = grad(observed_nda).shape

    A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
    A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

    D_1D = lambda x: grad(x.reshape(*X_shape)).flatten()
    D_adj_1D = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

    if args.reconstruction_type in ["TK0L2", "TK1L2"]:
        if args.reconstruction_type == "TK0L2":
            D_1D = lambda x: x.flatten()
            D_adj_1D = lambda x: x.flatten()

        solver = tk.TikhonovLinearSolver(
            A=A_1D, A_adj=A_adj_1D,
            B=D_1D, B_adj=D_adj_1D,
            b=b,
            x0=x0,
            x_scale=x_scale,
            iter_max=args.iter_max,
            minimizer=args.minimizer,
            verbose=args.verbose,
        )

    else:
        if args.reconstruction_type == "TVL2":
            prox_g_conj = prox.prox_tv_conj

        elif args.reconstruction_type == "HuberL2":
            prox_g_conj = prox.prox_huber_conj

        else:
            raise ValueError("Deconvolution type '%s' not known" %
                             args.reconstruction_type)

        prox_f = lambda x, tau: prox.prox_linear_least_squares(
            x=x, tau=tau, iter_max=args.iter_max,
            A=A_1D, A_adj=A_adj_1D, b=b, x0=x0, x_scale=x_scale)

        if args.solver == "PD":
            solver = pd.PrimalDualSolver(
                prox_f=prox_f,
                prox_g_conj=prox_g_conj,
                B=D_1D,
                B_conj=D_adj_1D,
                L2=8,
                x0=x0,
                iterations=args.iterations,
                # alg_type=alg_type,
                x_scale=x_scale,
                verbose=args.verbose,
            )
        elif args.solver == "ADMM":
            if args.reconstruction_type != "TVL2":
                raise ValueError("ADMM only works for TVL2")
            solver = admm.ADMMLinearSolver(
                A=lambda x: x.flatten(), A_adj=lambda x: x.flatten(),
                b=b,
                B=D_1D, B_adj=D_adj_1D,
                x0=x0,
                rho=args.rho,
                iterations=args.iterations,
                dimension=dimension,
                iter_max=args.iter_max,
                x_scale=x_scale,
                verbose=args.verbose,
            )
        else:
            raise ValueError("Solver '%s' not known" % args.solver)

    # ---------------------------Similarity Measures---------------------------
    if args.reference is not None:
        measures_dic = {
            m: lambda x, m=m:
            SimilarityMeasures.similarity_measures[m](x, x_ref)
            for m in args.measures}

        if args.reconstruction_type == "TK0L2":
            measures_dic["Reg"] = \
                lambda x: prior_meas.zeroth_order_tikhonov(x)
            measures_dic["Data"] = \
                lambda x: SimilarityMeasures.sum_of_squared_differences(
                    x, x_ref)

        elif args.reconstruction_type == "TK1L2":
            measures_dic["Reg"] = \
                lambda x: prior_meas.first_order_tikhonov(x, D_1D)
            measures_dic["Data"] = \
                lambda x: SimilarityMeasures.sum_of_squared_differences(
                    x, x_ref)

        elif args.reconstruction_type == "TVL2":
            measures_dic["Reg"] = \
                lambda x: prior_meas.total_variation(x, D_1D, dimension)
            measures_dic["Data"] = \
                lambda x: SimilarityMeasures.sum_of_squared_differences(
                    x, x_ref)

        elif args.reconstruction_type == "HuberL2":
            measures_dic["Reg"] = \
                lambda x: prior_meas.huber(x, D_1D, dimension)
            measures_dic["Data"] = \
                lambda x: SimilarityMeasures.sum_of_squared_differences(
                    x, x_ref)

    observer = Observer.Observer()
    observer.set_measures(measures_dic)
    solver.set_observer(observer)

    # ----------------------------Set Up Parameters----------------------------
    parameters = {}
    parameters["alpha"] = np.arange(*args.alpha_range)
    if args.data_losses is not None:
        parameters["data_loss"] = args.data_losses
    if args.data_loss_scale_range is not None:
        parameters["data_loss_scale"] = np.arange(*args.data_loss_scale_range)

    # -------------------------Set Up Parameter Study-------------------------
    if args.study_name is None:
        name = args.reconstruction_type
    else:
        name = args.study_name

    reconstruction_info_dic = {
        "shape": X_shape,
    }

    if args.reconstruction_type in ["TK0L2", "TK1L2"]:
        parameter_study = tkparam.TikhonovLinearSolverParameterStudy(
            solver, observer,
            dir_output=args.dir_output,
            parameters=parameters,
            name=name,
            reconstruction_info_dic=reconstruction_info_dic,
        )
    elif args.solver == "PD":
        parameter_study = pdparam.PrimalDualSolverParameterStudy(
            solver, observer,
            dir_output=args.dir_output,
            parameters=parameters,
            name=name,
            reconstruction_info_dic=reconstruction_info_dic,
        )
    else:
        parameter_study = admmparam.ADMMLinearSolverParameterStudy(
            solver, observer,
            dir_output=args.dir_output,
            parameters=parameters,
            name=name,
            reconstruction_info_dic=reconstruction_info_dic,
        )

    # Run parameter study
    parameter_study.run()
