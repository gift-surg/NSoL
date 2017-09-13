#!/usr/bin/python

##
# \file runDenoisingStudy.py
# \brief      Run parameter study for denoising problem. Result can be
#             visualized by showParameterStudy.py
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
import numericalsolver.ADMMLinearSolver as admm
import numericalsolver.PrimalDualSolver as pd
import numericalsolver.Observer as Observer
import numericalsolver.DataReader as dr
import numericalsolver.DataWriter as dw
import numericalsolver.PrimalDualSolverParameterStudy as pdparam
import numericalsolver.ADMMLinearSolverParameterStudy as admmparam
from numericalsolver.SimilarityMeasures import SimilarityMeasures as \
    SimilarityMeasures
from numericalsolver.ProximalOperators import ProximalOperators as prox
from numericalsolver.PriorMeasures import PriorMeasures as PriorMeasures
import numericalsolver.InputArgparser as InputArgparser

if __name__ == '__main__':

    input_parser = InputArgparser.InputArgparser(
        description="Run denoising algorithm study",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_observation(required=True)
    input_parser.add_reference(required=False)
    input_parser.add_dir_output(required=True)
    input_parser.add_study_name()
    input_parser.add_reconstruction_type(default="TVL2")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    # input_parser.add_solver(default="PD")
    input_parser.add_iterations(default=200)
    input_parser.add_rho(default=0.1)
    input_parser.add_verbose(default=0)

    input_parser.add_alpha_range(default=[0.01, 0.05, 0.005])
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

    # ------------------------------Set Up Solver------------------------------
    dimension = observed_nda.ndim

    b = observed_nda.flatten()
    x0 = observed_nda.flatten()
    x_scale = np.max(observed_nda)

    linear_operators = eval(
        "LinearOperators.LinearOperators%dD()" % (dimension))
    grad, grad_adj = linear_operators.get_gradient_operators()

    # A: X \rightarrow Y and D: X \rightarrow Z
    X_shape = observed_nda.shape
    Z_shape = grad(observed_nda).shape
    D_1D = lambda x: grad(x.reshape(*X_shape)).flatten()
    D_adj_1D = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

    if args.reconstruction_type == "TVL1":
        prox_f = lambda x, tau: prox.prox_ell1_denoising(
            x, tau, x0=b, x_scale=x_scale)
        prox_g_conj = prox.prox_tv_conj

    elif args.reconstruction_type == "TVL2":
        prox_f = lambda x, tau: prox.prox_ell2_denoising(
            x, tau, x0=b, x_scale=x_scale)
        prox_g_conj = prox.prox_tv_conj

    elif args.reconstruction_type == "HuberL1":
        prox_f = lambda x, tau: prox.prox_ell1_denoising(
            x, tau, x0=b, x_scale=x_scale)
        prox_g_conj = prox.prox_huber_conj

    elif args.reconstruction_type == "HuberL2":
        prox_f = lambda x, tau: prox.prox_ell2_denoising(
            x, tau, x0=b, x_scale=x_scale)
        prox_g_conj = prox.prox_huber_conj

    else:
        raise ValueError("Denoising '%s' type not known" %
                         args.reconstruction_type)

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

    # ---------------------------Similarity Measures---------------------------
    if args.reference is not None:
        measures_dic = {
            m: lambda x, m=m:
            SimilarityMeasures.similarity_measures[m](x, x_ref)
            for m in args.measures}

        if args.reconstruction_type == "TVL1":
            measures_dic["Reg"] = \
                lambda x: PriorMeasures.total_variation(x, D_1D, dimension)
            measures_dic["Data"] = lambda x: np.sum(np.abs(x - x_ref))

        elif args.reconstruction_type == "TVL2":
            measures_dic["Reg"] = \
                lambda x: PriorMeasures.total_variation(x, D_1D, dimension)
            measures_dic["Data"] = \
                lambda x: SimilarityMeasures.sum_of_squared_differences(
                    x, x_ref)

        elif args.reconstruction_type == "HuberL1":
            measures_dic["Reg"] = \
                lambda x: PriorMeasures.huber(x, D_1D, dimension)
            measures_dic["Data"] = lambda x: np.sum(np.abs(x - x_ref))

        elif args.reconstruction_type == "HuberL2":
            measures_dic["Reg"] = \
                lambda x: PriorMeasures.huber(x, D_1D, dimension)
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

    parameter_study = pdparam.PrimalDualSolverParameterStudy(
        solver, observer,
        dir_output=args.dir_output,
        parameters=parameters,
        name=name,
        reconstruction_info_dic={
            "shape": X_shape,
        }
    )

    # Run parameter study
    parameter_study.run()
