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

import src.Kernels as Kernels
import src.LinearOperators as LinearOperators
import src.Solver as Solver

sys.path.insert(1, os.path.join(
    os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR']), 'src', 'py'))
import utilities.PythonHelper as ph

from definitions import DIR_TEST

filename = os.path.join(DIR_TEST, "2D_Lena_512.png")
original_nda = ph.read_image(filename).astype(np.float64)

cov = np.zeros((2, 2))
cov[0, 0] = 2
cov[1, 1] = 2
noise = 10

linear_operators = LinearOperators.LinearOperators2D()

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

# b = blurred_noisy_nda.flatten()
# x0 = blurred_noisy_nda.flatten()

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

solver = Solver.ADMMLinearSolver(
    A=A_1D, A_adj=A_adj_1D,
    b=b,
    D=D_1D, D_adj=D_adj_1D,
    x0=x0,
    alpha=0.05,
    rho=0.5,
    ADMM_iterations=2,
    dimension=len(original_nda.shape),
    iter_max=10,
)

solver.run()
solver.print_statistics()
# recon_nda = solver.get_x().reshape(*X_shape)

# ph.show_arrays(
#     [original_nda, blurred_nda, blurred_noisy_nda, recon_nda],
#     title=["original", "blurred", "blurred+noise", "recon"],
#     fig_number=None)
