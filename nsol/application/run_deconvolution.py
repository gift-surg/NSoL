#!/usr/bin/python

##
# \file run_denoising.py
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

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import nsol.linear_operators as LinearOperators
import nsol.data_reader as dr
import nsol.data_writer as dw
import nsol.observer as Observer
import nsol.input_argparser as InputArgparser
import nsol.deconvolution_solver_parameter_study_interface as interface


def main():

    input_parser = InputArgparser.InputArgparser(
        description="Run TK0L2/TK1L2/TVL2/HuberL2 deconvolution",
    )
    input_parser.add_observation(required=True)
    input_parser.add_result(required=False)
    input_parser.add_reference(required=False)
    input_parser.add_blur(default=1)
    input_parser.add_reconstruction_type(default="TVL2")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_iterations(default=50)
    input_parser.add_solver(default="PD")
    input_parser.add_rho(default=0.1)
    input_parser.add_alpha(default=[0.01])
    input_parser.add_data_loss(default="linear")
    input_parser.add_data_loss_scale(default=1)
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
    else:
        x_ref = None

    if args.blur > 0:
        sigma = np.atleast_1d(args.blur)
        if sigma.ndim != observed_nda.ndim:
            try:
                cov = np.diag(np.ones(observed_nda.ndim)) * sigma**2
            except:
                raise IOError(
                    "Blur information must be either 1- or d-dimensional")

    # ------------------------------Set Up Solver------------------------------
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

    solver_interface = interface.DeconvolutionSolverStudyInterface(
        A=A_1D,
        A_adj=A_adj_1D,
        D=D_1D,
        D_adj=D_adj_1D,
        b=b,
        x0=x0,
        alpha=args.alpha[0],
        x_scale=x_scale,
        data_loss=args.data_loss,
        data_loss_scale=args.data_loss_scale,
        iter_max=args.iter_max,
        iterations=args.iterations,
        minimizer=args.minimizer,
        measures=args.measures,
        dimension=dimension,
        reconstruction_type=args.reconstruction_type,
        rho=args.rho,
        x_ref=x_ref,
        tv_solver=args.solver,
        verbose=args.verbose,
    )
    solver_interface.set_up_solver()
    solver_interface.set_up_measures()

    solver = solver_interface.get_solver()
    measures_dic = solver_interface.get_measures()

    # -----------------------------Run Reconstruction--------------------------
    for i, alpha in enumerate(args.alpha):
        ph.print_subtitle("Iteration %d/%d" % (i+1, len(args.alpha)))
        if dimension < 3:
            title_prefix = args.reconstruction_type + \
                " (" + r"$\alpha=%g$)" % alpha
        elif dimension == 3:
            title_prefix = ("%s_alpha=%g" % (
                args.reconstruction_type, alpha)).replace(".", "p")
        solver.set_alpha(alpha)

        observers[i] = Observer.Observer()
        observers[i].set_measures(measures_dic)
        solver.set_observer(observers[i])

        solver.run()
        recons[i] = np.array(solver.get_x().reshape(*X_shape))
        print("\nComputational time %s: %s" %
              (args.reconstruction_type, solver.get_computational_time()))

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
                # cmap="jet",
                cmap="Greys_r",
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

        # if args.reference is not None:
        linestyles = ["-", ":", "-", "-."] * 10

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
