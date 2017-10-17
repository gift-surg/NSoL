#!/usr/bin/python

##
# \file addNoise.py
# \brief      Tool to smooth data and add noise to it
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Sept 2017
#

import os
import sys
import numpy as np
import SimpleITK as sitk

import pysitk.PythonHelper as ph
import pysitk.SimpleITKHelper as sitkh

import nsol.InputArgparser as InputArgparser
import nsol.DataReader as dr
import nsol.DataWriter as dw
import nsol.LinearOperators as LinearOperators
import nsol.Noise as Noise

from nsol.definitions import ALLOWED_NOISE_TYPES


if __name__ == '__main__':

    input_parser = InputArgparser.InputArgparser(
        description="Tool to smooth data and add noise to it",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_filename(required=True)
    input_parser.add_result(required=True)
    input_parser.add_noise(default="gaussian")
    input_parser.add_noise_level(default=0.05)
    input_parser.add_blur(default=[1])
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    data_reader = dr.DataReader(args.filename)
    data_reader.read_data()
    nda = data_reader.get_data()

    # --------------------------------Blur Data--------------------------------
    if args.blur[0] > 0:
        sigma = np.atleast_1d(args.blur)
        if sigma.ndim != nda.ndim:
            try:
                cov = np.diag(np.ones(nda.ndim)) * sigma**2
            except:
                raise IOError(
                    "Blur information must be either 1- or d-dimensional")

        linear_operators = eval(
            "LinearOperators.LinearOperators%dD()" % nda.ndim)
        A = linear_operators.get_gaussian_blurring_operators(cov)[0]
        nda = A(nda)

    # --------------------------------Add Noise--------------------------------
    noise = Noise.Noise(nda, seed=1)

    if args.noise == "gaussian":
        noise.add_gaussian_noise(noise_level=args.noise_level, sigma=1)

    elif args.noise == "poisson":
        noise.add_poisson_noise(noise_level=args.noise_level, lmbda=1)

    elif args.noise == "uniform":
        noise.add_uniform_noise(noise_level=args.noise_level)

    elif args.noise == "s&p":
        noise.add_salt_and_pepper_noise(salt_vs_pepper=0.5, amount=0.1)

    elif args.noise in [None, "none", "None"]:
        pass

    else:
        raise IOError("Noise type '%s' not known" % args.noise)

    nda = noise.get_noisy_data()

    # -------------------------------Write Data-------------------------------
    data_writer = dw.DataWriter(nda, args.result, data_reader.get_image_sitk())
    data_writer.write_data()
