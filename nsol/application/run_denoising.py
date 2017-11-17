#!/usr/bin/python

##
# \file run_denoising.py
# \brief      Run TVL1/TVL2/HuberL1/HuberL2 denoising
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

import nsol.linear_operators as LinearOperators
import nsol.admm_linear_solver as admm
import nsol.primal_dual_solver as pd
import nsol.observer as Observer
import nsol.data_reader as dr
import nsol.data_writer as dw
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures
from nsol.proximal_operators import ProximalOperators as prox
from nsol.prior_measures import PriorMeasures as prior_meas
import nsol.input_argparser as InputArgparser


def main():

    input_parser = InputArgparser.InputArgparser(
        description="Run TVL1/TVL2/HuberL1/HuberL2 denoising",
    )
    input_parser.add_observation(required=True)
    input_parser.add_result(required=False)
    input_parser.add_reference(required=False)
    input_parser.add_reconstruction_type(default="TVL2")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_iterations(default=50)
    input_parser.add_solver(default="PD")
    input_parser.add_rho(default=0.1)
    input_parser.add_alpha(default=[0.03])
    input_parser.add_dir_output_figures(default=None)
    input_parser.add_verbose(default=0)
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    observers = [None] * len(args.alpha)
    recons = [None] * len(args.alpha)
    data_nda = []
    data_labels = []

    if len(args.alpha) > 1 and args.result is not None:
        ph.print_warning("Multiple alphas overwrite result")
    elif len(args.alpha) == 1 and args.result is None:
        raise IOError("'--result' must be specified")

    if args.result is None:
        ph.print_warning("No output ('--result') provided")

    # --------------------------------Read Data--------------------------------
    data_reader = dr.DataReader(args.observation)
    data_reader.read_data()
    observed_nda = data_reader.get_data()
    dimension = observed_nda.ndim

    data_nda.append(observed_nda)
    if dimension < 3:
        data_labels.append("observed:\n%s" %
                           os.path.basename(args.observation))
    elif dimension == 3:
        data_labels.append(
            os.path.basename(args.observation).split(".")[0])

    if args.reference is not None:
        data_reader = dr.DataReader(args.reference)
        data_reader.read_data()
        reference_nda = data_reader.get_data()
        data_nda.append(reference_nda)

        if dimension < 3:
            data_labels.append("reference:\n%s" %
                               os.path.basename(args.reference))
        elif dimension == 3:
            data_labels.append(
                os.path.basename(args.reference).split(".")[0])

        x_ref = reference_nda.flatten()

    # ------------------------------Set Up Solver------------------------------
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
        raise ValueError("Denoising type '%s' not known" %
                         args.reconstruction_type)

    for i, alpha in enumerate(args.alpha):
        if dimension < 3:
            title_prefix = args.reconstruction_type + \
                " (" + r"$\alpha=%g$)" % alpha
        elif dimension == 3:
            title_prefix = ("%s_alpha=%g" % (
                args.reconstruction_type, alpha)).replace(".", "p")

        if args.solver == "PD":
            solver = pd.PrimalDualSolver(
                prox_f=prox_f,
                prox_g_conj=prox_g_conj,
                B=D_1D,
                B_conj=D_adj_1D,
                L2=8,
                x0=x0,
                alpha=alpha,
                iterations=args.iterations,
                # alg_type=alg_type,
                x_scale=x_scale,
                verbose=args.verbose,
            )

        # ADMM only for TVL2
        # elif args.solver == "ADMM":
        #     solver = admm.ADMMLinearSolver(
        #         A=lambda x: x.flatten(), A_adj=lambda x: x.flatten(),
        #         b=b,
        #         B=D_1D, B_adj=D_adj_1D,
        #         x0=x0,
        #         alpha=alpha,
        #         rho=args.rho,
        #         iterations=args.iterations,
        #         dimension=dimension,
        #         # iter_max=iter_max,
        #         x_scale=x_scale,
        #         verbose=args.verbose,
        #     )
        else:
            raise ValueError("Solver '%s' not known" % args.solver)

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
        data_nda.append(recons[i])
        data_labels.append(title_prefix)

        if args.result is not None:
            data_writer = dw.DataWriter(
                recons[i], args.result, data_reader.get_image_sitk())
            data_writer.write_data()

    # ----------------------------Visualize Results----------------------------
    if args.verbose:
        filename_prefix = args.reconstruction_type

        if dimension == 2:
            ph.show_arrays(
                data_nda,
                title=data_labels,
                fig_number=None,
                cmap="jet",
                use_same_scaling=True,
                directory=args.dir_output_figures,
                filename=args.reconstruction_type+"_comparison.pdf",
                save_figure=0 if args.dir_output_figures is None else 1,
                fontsize=8,
            )
        elif dimension == 3:
            images_sitk = [None] * len(data_nda)
            observation_sitk = data_reader.get_image_sitk()
            for i, nda in enumerate(data_nda):
                images_sitk[i] = sitk.GetImageFromArray(nda)
                images_sitk[i].CopyInformation(observation_sitk)
            sitkh.show_sitk_image(images_sitk, label=data_labels)

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

    return 0

if __name__ == '__main__':
    main()
