#!/usr/bin/python

##
# \file run_denoising_study.py
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
import nsol.deconvolution_solver_parameter_study_interface as \
    deconv_interface


def main():

    input_parser = InputArgparser.InputArgparser(
        description="Run TK0L2/TK1L2/TVL2/HuberL2 deconvolution",
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
    input_parser.add_iter_max(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_alpha(default=[0.01])
    input_parser.add_data_loss(default="linear")
    input_parser.add_data_loss_scale(default=1)
    input_parser.add_verbose(default=0)

    # Range for parameter sweeps
    input_parser.add_alpha_range(default=[0.0001, 0.05, 10])
    input_parser.add_data_losses(
        # default=["linear", "arctan"]
    )
    input_parser.add_data_loss_scale_range(
        # default=[0.1, 1.5, 2]
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

    # ----------------------------Set Up Parameters----------------------------
    parameters = {}
    parameters["alpha"] = np.linspace(
        args.alpha_range[0], args.alpha_range[1], int(args.alpha_range[2]))
    if args.data_losses is not None:
        parameters["data_loss"] = args.data_losses
    if args.data_loss_scale_range is not None:
        parameters["data_loss_scale"] = np.linspace(
            args.data_loss_scale_range[0],
            args.data_loss_scale_range[1],
            int(args.data_loss_scale_range[2]))

    # -------------------------Set Up Parameter Study-------------------------
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

    if args.study_name is None:
        name = args.reconstruction_type
    else:
        name = args.study_name

    reconstruction_info = {
        "shape": X_shape,
    }

    parameter_study_interface = \
        deconv_interface.DeconvolutionParameterStudyInterface(
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
            dir_output=args.dir_output,
            parameters=parameters,
            name=name,
            reconstruction_info=reconstruction_info,
            x_ref=x_ref,
            tv_solver=args.solver,
            verbose=args.verbose,
        )
    parameter_study_interface.set_up_parameter_study()
    parameter_study = parameter_study_interface.get_parameter_study()

    # Run parameter study
    parameter_study.run()

    print("\nComputational time for Deconvolution Parameter Study %s: %s" %
          (name, parameter_study.get_computational_time()))

    return 0

if __name__ == '__main__':
    main()
