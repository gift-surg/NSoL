#!/usr/bin/python

##
# \file foo.py
# \brief      Playground file
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import numpy as np
import SimpleITK as sitk

import src.Kernels as Kernels
import src.LinearOperators as LinearOperators
import src.Solver as Solver

sys.path.insert(1, os.path.join(
    os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR']), 'src', 'py'))
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh

from definitions import DIR_TEST

dimension = 1
dimension = 2
# dimension = 3

solver_type = "TK"
solver_type = "ADMM"

noise = 10
cov = np.diag(np.ones(dimension)) * 2

if dimension == 1:
    cov = cov[0][0]
    original_nda = np.zeros(50)
    original_nda[10] = 10
    original_nda[20] = 100
    original_nda[23] = 150

elif dimension == 2:
    filename = os.path.join(DIR_TEST, "2D_Lena_256.png")
    # filename = os.path.join(DIR_TEST, "2D_Lena_512.png")
    original_nda = ph.read_image(filename).astype(np.float64)

elif dimension == 3:
    filename = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
    original_sitk = sitk.ReadImage(filename, sitk.sitkFloat64)
    original_nda = sitk.GetArrayFromImage(original_sitk)


linear_operators = eval("LinearOperators.LinearOperators%dD()" % (dimension))

A, A_adj = linear_operators.get_gaussian_blurring_operators(cov)
grad, grad_adj = linear_operators.get_gradient_operators()

blurred_nda = A(original_nda)
blurred_noisy_nda = blurred_nda + noise * np.random.rand(*original_nda.shape)

D = lambda x: grad(x)
D_adj = lambda x: grad_adj(x)

I = lambda x: x
I_adj = lambda x: x

# A: X \rightarrow Y and D: X \rightarrow Z
X_shape = original_nda.shape
Y_shape = A(original_nda).shape
Z_shape = D(original_nda).shape

A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

D_1D = lambda x: D(x.reshape(*X_shape)).flatten()
D_adj_1D = lambda x: D_adj(x.reshape(*Z_shape)).flatten()

b = blurred_noisy_nda.flatten()
x0 = blurred_noisy_nda.flatten()

if dimension == 1:
    ph.show_curves(
        [original_nda, blurred_nda, blurred_noisy_nda],
        labels=["original", "blurred", "blurred+noise"],
        fig_number=1)

elif dimension == 2:
    ph.show_arrays(
        [original_nda, blurred_nda, blurred_noisy_nda],
        title=["original", "blurred", "blurred+noise"],
        fig_number=1)

elif dimension == 3:
    blurred_sitk = sitk.GetImageFromArray(blurred_nda)
    blurred_sitk.CopyInformation(original_sitk)

    blurred_noisy_sitk = sitk.GetImageFromArray(blurred_noisy_nda)
    blurred_noisy_sitk.CopyInformation(original_sitk)

    sitkh.show_sitk_image(
        [original_sitk, blurred_sitk, blurred_noisy_sitk],
        label=["original", "blurred", "blurred+noise"])


if solver_type == "TK":
    solver = Solver.TikhonovLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        B=D_1D, B_adj=D_adj_1D,
        b=b,
        alpha=0.05,
        x0=x0,
        # minimizer="lsmr",
        # minimizer="least_squares",
        # minimizer="L-BFGS-B",
        # data_loss="linear",
        # data_loss="huber",
        # data_loss="soft_l1",
        iter_max=10,
    )

elif solver_type == "ADMM":
    solver = Solver.ADMMLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        b=b,
        D=D_1D, D_adj=D_adj_1D,
        x0=x0,
        alpha=5,
        rho=0.5,
        ADMM_iterations=10,
        dimension=dimension,
        iter_max=10,
        # data_loss="huber",
    )

solver.run()
solver.print_statistics()
recon_nda = solver.get_x().reshape(*X_shape)

data = [original_nda, blurred_nda, blurred_noisy_nda, recon_nda]
labels = ["original", "blurred", "blurred+noise", "recon"]

if dimension == 1:
    ph.show_curves(data, labels=labels, fig_number=1)

elif dimension == 2:
    ph.show_arrays(data, title=labels, fig_number=1)

elif dimension == 3:
    recon_sitk = sitk.GetImageFromArray(recon_nda)
    recon_sitk.CopyInformation(original_sitk)

    sitkh.show_sitk_image(data, label=labels)
