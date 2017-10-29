##
# \file compare_solver.py
# \brief      Playground file to show-case use of solvers for denoising and
#             deconvolution applications.
#
# Visualizations are provided to compare solver performances. Run script in
# IPython (or Python) interactive shell to keep visualization windows open.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import nsol.linear_operators as lin_op
import nsol.noise as noise
import nsol.tikhonov_linear_solver as tk
import nsol.admm_linear_solver as admm
import nsol.primal_dual_solver as pd
import nsol.observer as obs
from nsol.proximal_operators import ProximalOperators as prox
from nsol.similarity_measures import SimilarityMeasures as sim_meas
from nsol.definitions import DIR_TEST

# Define dimension of example to be computed
# dimension = 1
dimension = 2
# dimension = 3

# Indicate which solver to compute
solver_TK = 1
solver_ADMM = 1
solver_PrimalDual = 1

# Add noise
flag_noise = 1
noise_level = 0.05

# Add blurring
flag_blurring = 1
sigma2 = 1

# Verbose solver output
verbose = 0

# Regularization parameter (including approximate suggestions for its values)
# Approximate values for L2-deblurring: 0.01 (Gaussian)
# Approximate values for L2-denoising: 0.6 (Gaussian)
# Approximate values for L1-denoising: 0.6 (Salt and pepper)
# alpha = 0.6
alpha = 0.01

# Maximum iterations for scipy-solver (only TK and ADMM)
iter_max = 10

# ADMM: regularization parameter for augmented Lagrangian term
rho = 0.1

# Iterations for ADMM solver
ADMM_iterations = 50

# Iterations for Primal-Dual solver
PD_iterations = 50

# Choose similarity measures; acronyms as defined in similarity_measures.py
measures = [
    "RMSE",
    "NCC",
    "NMI",
    "PSNR",
    "SSIM",
]

# ---------------------------------Read input---------------------------------
cov = np.diag(np.ones(dimension)) * sigma2
if dimension == 1:
    cov = cov[0][0]
    original_nda = np.ones(50) * 50
    original_nda[5] = 10
    original_nda[16] = 100
    original_nda[23] = 150
    original_nda[30] = 20

elif dimension == 2:
    # filename = os.path.join(DIR_TEST, "2D_Lena_256.png")
    # filename = os.path.join(DIR_TEST, "2D_House_256.png")
    # filename = os.path.join(DIR_TEST, "2D_Cameraman_256.png")
    filename = os.path.join(DIR_TEST, "2D_BrainWeb.png")
    original_nda = ph.read_image(filename).astype(np.float64)

elif dimension == 3:
    filename = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
    original_sitk = sitk.ReadImage(filename)
    original_nda = sitk.GetArrayFromImage(original_sitk).astype(np.float64)

# --------------------------------Corrupt Data--------------------------------
linear_operators = eval("lin_op.LinearOperators%dD()" % (dimension))

A, A_adj = linear_operators.get_gaussian_blurring_operators(cov)
grad, grad_adj = linear_operators.get_gradient_operators()

if flag_blurring:
    blurred_nda = A(original_nda)
else:
    blurred_nda = original_nda

noise_ = noise.Noise(blurred_nda, seed=1)
if flag_noise:
    noise_.add_gaussian_noise(noise_level=noise_level)
    # noise_.add_poisson_noise(noise_level=noise_level)
    # noise_.add_uniform_noise(noise_level=noise_level)
    # noise_.add_salt_and_pepper_noise()
blurred_noisy_nda = noise_.get_noisy_data()

# ------------------------------Optimizer Set-up------------------------------

# Right hand-side of linear equation; measurement
b = blurred_noisy_nda.flatten()

# Initial value
x0 = blurred_noisy_nda.flatten()

# Scaling of data arrays for solvers
x_scale = np.max(original_nda)

# Define operator spaces: Blurring-op A: X -> Y and differential-op D: X -> Z
X_shape = original_nda.shape
Y_shape = A(original_nda).shape
Z_shape = grad(original_nda).shape

# Operator defs for solver on flattened arrays, i.e. always 1D to 1D mapping
if flag_blurring:
    # Deconvolution problem
    A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
    A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()
else:
    # Denoising problem
    A_1D = lambda x: x.flatten()
    A_adj_1D = lambda x: x.flatten()

# Differential operator
D_1D = lambda x: grad(x.reshape(*X_shape)).flatten()
D_adj_1D = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

# Definitions of data loss for Proximal Operator (for Primal-Dual only)
if flag_blurring:
    # L2-deconvolution
    prox_f = lambda x, tau: prox.prox_linear_least_squares(
        x=x, tau=tau,
        A=A_1D, A_adj=A_adj_1D, b=b, x0=x0, x_scale=x_scale)
else:
    # L2-denoising
    prox_f = lambda x, tau: prox.prox_ell2_denoising(
        x, tau, x0=b, x_scale=x_scale)

    # L1-denoising
    # prox_f = lambda x, tau: prox.prox_ell1_denoising(
    #     x, tau, x0=b, x_scale=x_scale)


data_labels = ["original", "blurred+noise"]
data_nda = [original_nda, blurred_noisy_nda]

# Convert array into SimpleITK objects for visualization purposes
if dimension == 3:
    original_sitk = sitk.GetImageFromArray(original_nda)
    blurred_noisy_sitk = sitk.GetImageFromArray(blurred_noisy_nda)

    data_sitk = [original_sitk, blurred_noisy_sitk]

# -------------------------------Reconstructions-------------------------------

# Define observer to record performances of solvers. It records the obtained
# reconstructions of each iteration
observers = []

# Define similarity measures to be evaluated by observer
x_ref = original_nda.flatten()
measures_dic = {
    m: lambda x, m=m:
    sim_meas.similarity_measures[m](x, x_ref) for m in measures
}

# Tikhonov Solver (only L2 data loss possible)
if solver_TK:
    name = "TK"
    minimizer = "L-BFGS-B"
    observer_tk = obs.Observer(name)
    observer_tk.set_measures(measures_dic)
    observers.append(observer_tk)
    solver = tk.TikhonovLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        B=D_1D, B_adj=D_adj_1D,
        b=b,
        alpha=alpha,
        x0=x0,
        iter_max=iter_max,
        minimizer=minimizer,
        verbose=verbose,
        x_scale=x_scale,
    )
    solver.set_observer(observer_tk)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

# ADMM Solver (only L2 data loss possible)
if solver_ADMM:
    name = "ADMM-TV"
    observer_admm = obs.Observer(name)
    observer_admm.set_measures(measures_dic)
    observers.append(observer_admm)
    solver = admm.ADMMLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        b=b,
        B=D_1D, B_adj=D_adj_1D,
        x0=x0,
        alpha=alpha,
        rho=rho,
        iterations=ADMM_iterations,
        dimension=dimension,
        iter_max=iter_max,
        x_scale=x_scale,
        verbose=verbose,
    )
    solver.set_observer(observer_admm)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

# Primal Dual Solver (L1 and L2 data losses possible)
if solver_PrimalDual:
    name = "PrimalDual-TV-ALG2"
    observer_pd = obs.Observer(name)
    observer_pd.set_measures(measures_dic)
    observers.append(observer_pd)
    solver = pd.PrimalDualSolver(
        prox_f=prox_f,
        prox_g_conj=prox.prox_tv_conj,
        B=D_1D,
        B_conj=D_adj_1D,
        L2=8,
        x0=x0,
        alpha=alpha,
        iterations=PD_iterations,
        alg_type="ALG2",
        verbose=verbose,
        x_scale=x_scale,
    )
    solver.set_observer(observer_pd)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

# ------------------------------Visualize Results------------------------------
linestyles = ["-", ":", "-", "-."] * 10

# Show visual comparison of reconstructions
if dimension == 1:
    ph.show_curves(data_nda, labels=data_labels, fig_number=1,
                   linestyle=linestyles, markers=ph.MARKERS)
elif dimension == 2:
    ph.show_arrays(data_nda, title=data_labels, fig_number=1)
elif dimension == 3:
    sitkh.show_sitk_image(data_sitk, label=data_labels)

# Evaluate ground-truth similarity of reconstructions recorded by observer
for m in range(0, len(observers)):
    observers[m].compute_measures()
    print("Computational time %s: %s" %
          (observers[m].get_name(), observers[m].get_computational_time()))

# Visualize similarity results
for k in measures_dic.keys():
    title = k
    x = []
    y = []
    legend = []
    for m in range(0, len(observers)):
        y.append(observers[m].get_measures()[title])
        legend.append(observers[m].get_name())
        x.append(range(0, len(y[-1])))
    ph.show_curves(y, x=x, xlabel="iteration", labels=legend, title=title,
                   linestyle=linestyles, markers=ph.MARKERS, markevery=1)
