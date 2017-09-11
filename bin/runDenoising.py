#!/usr/bin/python

##
# \file compareSolver.py
# \brief      Playground file to test solvers for Yijing's application
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
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
from numericalsolver.SimilarityMeasures import SimilarityMeasures as \
    SimilarityMeasures
from numericalsolver.ProximalOperators import ProximalOperators as prox
from numericalsolver.PriorMeasures import PriorMeasures as prior_meas
import numericalsolver.InputArgparser as InputArgparser

if __name__ == '__main__':

    input_parser = InputArgparser.InputArgparser(
        description="Run denoising algorithms",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_observation(required=True)
    input_parser.add_reference(required=False)
    input_parser.add_result(required=False)
    input_parser.add_reconstruction_type(default="TVL1")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_tv_solver(default="PD")
    input_parser.add_iterations(default=50)
    input_parser.add_rho(default=0.1)
    input_parser.add_alpha(default=1/0.7)
    input_parser.add_dir_output_figures(default=None)
    input_parser.add_verbose(default=0)
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

        data_nda.append(reference_nda)
        data_labels.append("reference")
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
        alpha=args.alpha,
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
        observer = Observer.Observer()
        observer.set_measures(measures_dic)
    else:
        observer = None
    solver.set_observer(observer)

    # ---------------------------Run Reconstruction---------------------------
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    print("Computational time %s: %s" %
          (args.reconstruction_type, solver.get_computational_time()))

    if args.result is not None:
        data_writer = dw.DataWriter(recon_nda, args.result)
        data_writer.write_data()

    # ----------------------------Visualize Results----------------------------
    title_prefix = args.reconstruction_type + \
        " (" + r"$\alpha=%g$)" % args.alpha
    filename_prefix = \
        ("_").join([args.reconstruction_type, "alpha%g" % args.alpha])

    data_nda = []
    data_nda.append(observed_nda)
    data_nda.append(recon_nda)
    data_labels = []
    data_labels.append("observed")
    data_labels.append(title_prefix)

    ph.show_arrays(
        data_nda,
        title=data_labels,
        fig_number=None,
        cmap="jet",
        use_same_scaling=True,
        directory=args.dir_output_figures,
        filename=filename_prefix+"_comparison.pdf",
        save_figure=0 if args.dir_output_figures is None else 1,
    )

    if args.reference is not None:
        linestyles = ["-", ":", "-", "-."] * 10
        observer.compute_measures()

        for k in measures_dic.keys():
            title = title_prefix
            x = []
            y = []
            legend = []
            y.append(observer.get_measures()[k])
            legend.append(k)
            x.append(range(0, len(y[-1])))
            ph.show_curves(
                y=y,
                x=x,
                xlabel="iteration",
                labels=legend,
                title=title,
                linestyle=linestyles,
                markers=ph.MARKERS,
                markevery=1,
                directory=args.dir_output_figures,
                filename=filename_prefix+"_"+k+".pdf",
                save_figure=0 if args.dir_output_figures is None else 1,
            )
