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
import src.TikhonovLinearSolver as tk
import src.ADMMLinearSolver as admm
import src.PrimalDualSolver as pd

sys.path.insert(1, os.path.join(
    os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR']), 'src', 'py'))
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh

from definitions import DIR_TEST


def prox_tau_linear_least_squares(x, tau, A, A_adj, B, B_adj, b, x0):
    tikhonov = tk.TikhonovLinearSolver(
        A=A, A_adj=A_adj, B=B, B_adj=B_adj, b=b, x0=x0, b_reg=x, alpha=1./tau)
    tikhonov.run()
    return tikhonov.get_x()


##
# Proximal operator for least squares denoising.
#
# With g(x) = 1/2 * ||x - x0||^2 it holds that
# prox_{\tau g}(x) = (x + \tau*x0) / (1 + \tau*x0)
# \date       2017-07-21 01:52:01+0100
#
# \param      x     numpy array
# \param      tau   step size; scalar >= 0
# \param      x0    observed; numpy array
#
# \return     prox_{\tau g}(x) as numpy array
#
def prox_tau_least_squares_denoising(x, tau, x0):
    return (x + tau * x0) / (1. + tau)


def prox_sigma_tv_conj(x, sigma):
    return x / np.maximum(1, np.abs(x))


def prox_sigma_huber_conj(x, sigma, alpha):
    x /= (1. + sigma * alpha)
    return x / np.maximum(1, np.abs(x))


dimension = 1
dimension = 2
# dimension = 3

solver_TK = 0
solver_ADMM = 1
solver_PrimalDual = 1

flag_normalize = 1
flag_blurring = 0
flag_noise = 1
noise = 0.1

iter_max = 10
alpha = 0.03
rho = 1
ADMM_iterations = 100
PD_iterations = 100

cov = np.diag(np.ones(dimension)) * 1.5

if dimension == 1:
    cov = cov[0][0]
    original_nda = np.zeros(50)
    original_nda[5] = 10
    original_nda[16] = 100
    original_nda[23] = 150
    original_nda[30] = 50

elif dimension == 2:
    filename = os.path.join(DIR_TEST, "2D_Lena_256.png")
    # filename = os.path.join(DIR_TEST, "2D_Brain_Source.png")
    original_nda = ph.read_image(filename).astype(np.float64)

elif dimension == 3:
    filename = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
    original_sitk = sitk.ReadImage(filename)
    original_nda = sitk.GetArrayFromImage(original_sitk).astype(np.float64)

if flag_normalize:
    original_nda = original_nda / original_nda.max()

linear_operators = eval("LinearOperators.LinearOperators%dD()" % (dimension))

A, A_adj = linear_operators.get_gaussian_blurring_operators(cov)
grad, grad_adj = linear_operators.get_gradient_operators()

if flag_blurring:
    blurred_nda = A(original_nda)
else:
    blurred_nda = original_nda

if flag_noise:
    blurred_noisy_nda = blurred_nda + noise * blurred_nda.max() * \
        np.random.rand(*original_nda.shape)
else:
    blurred_noisy_nda = blurred_nda


# A: X \rightarrow Y and D: X \rightarrow Z
X_shape = original_nda.shape
Y_shape = A(original_nda).shape
Z_shape = D(original_nda).shape

b = blurred_noisy_nda.flatten()
x0 = blurred_noisy_nda.flatten()

D = lambda x: grad(x)
D_adj = lambda x: grad_adj(x)

I_1D = lambda x: x
I_adj_1D = lambda x: x

A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

D_1D = lambda x: D(x.reshape(*X_shape)).flatten()
D_adj_1D = lambda x: D_adj(x.reshape(*Z_shape)).flatten()

prox_tau_f = lambda x, tau: prox_tau_linear_least_squares(
    x=x, tau=tau,
    A=A_1D, A_adj=A_adj_1D, B=I_1D, B_adj=I_adj_1D, b=b, x0=x0)

prox_tau_f = lambda x, tau: prox_tau_least_squares_denoising(x, tau, x0=b)

prox_sigma_g_conj = lambda x, sigma: prox_sigma_tv_conj(
    x=x, sigma=sigma)
# prox_sigma_g_conj = lambda x, sigma: prox_sigma_huber_conj(
#     x=x, sigma=sigma,
#     alpha=0.05)

data_nda = [original_nda]
data_labels = ["original"]

# data_nda.append(blurred_nda)
# data_labels.append("blurred")

data_nda.append(blurred_noisy_nda)
data_labels.append("blurred+noise")

if dimension == 1:
    ph.show_curves(data_nda, labels=data_labels, fig_number=1)

elif dimension == 2:
    ph.show_arrays(data_nda, title=data_labels, fig_number=1)

elif dimension == 3:

    original_sitk = sitk.GetImageFromArray(original_nda)
    data_sitk = [original_sitk]

    # blurred_sitk = sitk.GetImageFromArray(blurred_nda)
    # data_sitk.append(blurred_sitk)

    blurred_noisy_sitk = sitk.GetImageFromArray(blurred_noisy_nda)
    data_sitk.append(blurred_noisy_sitk)

    ph.killall_itksnap()
    sitkh.show_sitk_image(data_sitk, label=data_labels)

if solver_TK:
    solver = tk.TikhonovLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        B=D_1D, B_adj=D_adj_1D,
        b=b,
        alpha=alpha,
        x0=x0,
        # minimizer="lsmr",
        # minimizer="least_squares",
        # minimizer="L-BFGS-B",
        # data_loss="linear",
        # data_loss="huber",
        # data_loss="soft_l1",
        iter_max=iter_max,
    )
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append("TK")

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

if solver_ADMM:
    solver = admm.ADMMLinearSolver(
        # A=A_1D, A_adj=A_adj_1D,
        A=I_1D, A_adj=I_adj_1D,
        b=b,
        D=D_1D, D_adj=D_adj_1D,
        x0=x0,
        alpha=alpha,
        rho=rho,
        ADMM_iterations=ADMM_iterations,
        dimension=dimension,
        iter_max=iter_max,
        # data_loss="huber",
    )
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append("TV-ADMM")

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

if solver_PrimalDual:
    solver = pd.PrimalDualSolver(
        prox_tau_f=prox_tau_f,
        prox_sigma_g_conj=prox_sigma_g_conj,
        K=D_1D,
        K_conj=D_adj_1D,
        L2=8,
        x0=x0,
        alpha=alpha,
        iterations=PD_iterations,
    )
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append("TV-PrimalDual")

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

if dimension == 1:
    ph.show_curves(data_nda, labels=data_labels, fig_number=1)

elif dimension == 2:
    ph.show_arrays(data_nda, title=data_labels, fig_number=1)

elif dimension == 3:
    sitkh.show_sitk_image(data_sitk, label=data_labels)
