#!/usr/bin/python

##
# \file runDenoising.py
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
    input_parser.add_result(required=False)
    input_parser.add_blur(default=1)
    input_parser.add_reconstruction_type(default="TVL2")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_iterations(default=50)
    input_parser.add_solver(default="PD")
    input_parser.add_rho(default=0.1)
    input_parser.add_alpha(default=[0.01])
    input_parser.add_iter_max(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_dir_output_figures(default=None)
    input_parser.add_verbose(default=0)
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    observers = [None] * len(args.alpha)
    recons = [None] * len(args.alpha)
    data_nda = []
    data_labels = []

    if args.result is not None and len(args.alpha) > 1:
        raise ValueError("Result can only be written for one alpha")

    # --------------------------------Read Data--------------------------------
    data_reader = dr.DataReader(args.observation)
    data_reader.read_data()
    observed_nda = data_reader.get_data()
    data_nda.append(observed_nda)
    data_labels.append("observed:\n%s" % os.path.basename(args.observation))

    if args.reference is not None:
        data_reader = dr.DataReader(args.reference)
        data_reader.read_data()
        reference_nda = data_reader.get_data()
        data_nda.append(reference_nda)
        data_labels.append("reference:\n%s" % os.path.basename(args.reference))
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

    for i, alpha in enumerate(args.alpha):
        title_prefix = args.reconstruction_type + \
            " (" + r"$\alpha=%g$)" % alpha
        solver.set_alpha(alpha)

        # ---------------------------Similarity Measures-----------------------
        if args.reference is not None:
            measures_dic = {
                m: lambda x, m=m:
                SimilarityMeasures.similarity_measures[m](x, x_ref)
                for m in args.measures}
            observers[i] = Observer.Observer()
            observers[i].set_measures(measures_dic)
        else:
            observers[i] = None
        solver.set_observer(observers[i])

        # ---------------------------Run Reconstruction------------------------
        solver.run()
        recons[i] = np.array(solver.get_x().reshape(*X_shape))
        print("Computational time %s: %s" %
              (args.reconstruction_type, solver.get_computational_time()))

        data_nda.append(recons[i])
        data_labels.append(title_prefix)

        if args.result is not None:
            data_writer = dw.DataWriter(
                recons[i], args.result, data_reader.get_image_sitk())
            data_writer.write_data()

    # ----------------------------Visualize Results----------------------------

    filename_prefix = args.reconstruction_type

    ph.show_arrays(
        data_nda,
        title=data_labels,
        fig_number=None,
        # cmap="jet",
        cmap="Greys_r",
        use_same_scaling=True,
        directory=args.dir_output_figures,
        filename=args.reconstruction_type+"_comparison.pdf",
        save_figure=0 if args.dir_output_figures is None else 1,
        fontsize=8,
    )

    if args.reference is not None:
        linestyles = ["-", ":", "-", "-."] * 10

        for ob in observers:
            ob.compute_measures()

        for k in measures_dic.keys():
            x = []
            y = []
            legend = []
            for i, alpha in enumerate(args.alpha):
                title = args.reconstruction_type + ": %s" % k
                y.append(observers[i].get_measures()[k])
                legend.append(r"$\alpha=%g$" % alpha)
                x.append(range(0, len(y[-1])))
            ph.show_curves(
                y=y,
                x=x,
                xlabel="iteration",
                labels=legend,
                title=title,
                linestyle=linestyles,
                markers=ph.MARKERS,
                markevery=10,
                directory=args.dir_output_figures,
                filename=filename_prefix+"_"+k+".pdf",
                save_figure=0 if args.dir_output_figures is None else 1,
            )
