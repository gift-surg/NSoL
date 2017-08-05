#!/usr/bin/python

##
# \file compareSolver.py
# \brief      Playground file
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import numpy as np
import SimpleITK as sitk

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

import numericalsolver.LinearOperators as LinearOperators
import numericalsolver.Noise as Noise
import numericalsolver.TikhonovLinearSolver as tk
import numericalsolver.TikhonovParameterStudy as tkparam
import numericalsolver.ParameterStudyReader as psreader
import numericalsolver.ADMMLinearSolver as admm
import numericalsolver.PrimalDualSolver as pd
import numericalsolver.Monitor as monitor
from numericalsolver.ProximalOperators import ProximalOperators as prox
from numericalsolver.SimilarityMeasures import SimilarityMeasures as sim_meas
from numericalsolver.PriorMeasures import PriorMeasures as prior_meas

from numericalsolver.definitions import DIR_TEST


np.random.seed(seed=1)

# dimension = 1
dimension = 2
# dimension = 3

solver_TK = 1
solver_ADMM = 0
solver_PrimalDual = 0

verbose = 0
flag_normalize = 1
max_intensity_normalization = 10000
flag_blurring = 1
flag_noise = 1
noise_level = 0.05

iter_max = 10
alpha = 0.01  # Denoising (S&P) 0.6; default: 0.01
rho = 0.5
ADMM_iterations = 50
PD_iterations = 50

cov = np.diag(np.ones(dimension)) * 1.5

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

if flag_normalize:
    original_nda = original_nda / original_nda.max()
    original_nda *= max_intensity_normalization

linear_operators = eval("LinearOperators.LinearOperators%dD()" % (dimension))

A, A_adj = linear_operators.get_gaussian_blurring_operators(cov)
grad, grad_adj = linear_operators.get_gradient_operators()

if flag_blurring:
    blurred_nda = A(original_nda)
else:
    blurred_nda = original_nda

noise = Noise.Noise(blurred_nda)
if flag_noise:
    # noise.add_gaussian_noise(noise_level=noise_level)
    noise.add_poisson_noise(noise_level=noise_level)
    # noise.add_uniform_noise(noise_level=noise_level)
    # noise.add_salt_and_pepper_noise()
blurred_noisy_nda = noise.get_noisy_data()


# A: X \rightarrow Y and D: X \rightarrow Z
X_shape = original_nda.shape
Y_shape = A(original_nda).shape
Z_shape = grad(original_nda).shape

b = blurred_noisy_nda.flatten()
x0 = blurred_noisy_nda.flatten()

x_scale = np.max(original_nda)

I_1D = lambda x: x.flatten()
I_adj_1D = lambda x: x.flatten()

A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

D_1D = lambda x: grad(x.reshape(*X_shape)).flatten()
D_adj_1D = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

if flag_blurring:
    prox_f = lambda x, tau: prox.prox_linear_least_squares(
        x=x, tau=tau,
        A=A_1D, A_adj=A_adj_1D, b=b, x0=x0, x_scale=x_scale)
else:
    prox_f = lambda x, tau: prox.prox_ell1_denoising(x, tau, x0=b)


if not flag_blurring:
    A_1D = I_1D
    A_adj_1D = I_adj_1D

data_nda = [original_nda]
data_labels = ["original"]

# data_nda.append(blurred_nda)
# data_labels.append("blurred")

data_nda.append(blurred_noisy_nda)
data_labels.append("blurred+noise")

# if dimension == 1:
#     ph.show_curves(data_nda, labels=data_labels, fig_number=1)

# elif dimension == 2:
#     ph.show_arrays(data_nda, title=data_labels, fig_number=1)

# elif dimension == 3:

#     original_sitk = sitk.GetImageFromArray(original_nda)
#     data_sitk = [original_sitk]

#     # blurred_sitk = sitk.GetImageFromArray(blurred_nda)
#     # data_sitk.append(blurred_sitk)

#     blurred_noisy_sitk = sitk.GetImageFromArray(blurred_noisy_nda)
#     data_sitk.append(blurred_noisy_sitk)

#     ph.killall_itksnap()
#     sitkh.show_sitk_image(data_sitk, label=data_labels)

x_ref = original_nda.flatten()

# -----------------------------Similarity Measures-----------------------------
ssd = lambda x: sim_meas.sum_of_squared_differences(x, x_ref)
mse = lambda x: sim_meas.mean_squared_error(x, x_ref)
rmse = lambda x: sim_meas.root_mean_square_error(x, x_ref)
psnr = lambda x: sim_meas.peak_signal_to_noise_ratio(x, x_ref)
ssim = lambda x: sim_meas.structural_similarity(x, x_ref)
ncc = lambda x: sim_meas.normalized_cross_correlation(x, x_ref)
mi = lambda x: sim_meas.mutual_information(x, x_ref)
nmi = lambda x: sim_meas.normalized_mutual_information(x, x_ref)

tk1 = lambda x: prior_meas.first_order_tikhonov(x, D_1D)
tv = lambda x: prior_meas.total_variation(x, D_1D, dimension)

monitors = []
measures_dic = {
    "SSD": ssd,
    # "RMSE": rmse,
    # "PSNR": psnr,
    # "SSIM": ssim,
    # "NCC": ncc,
    # "MI": mi,
    # "NMI": nmi,
    "TK1": tk1,
    "TV": tv,
}

# ----------------------------------Tikhonov----------------------------------
if solver_TK:
    name = "TK"
    minimizer = "L-BFGS-B"
    monitor_tk = monitor.Monitor(name)
    monitor_tk.set_measures(measures_dic)
    monitors.append(monitor_tk)
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
    solver.set_monitor(monitor_tk)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)


# if solver_TK:
#     minimizer = "L-BFGS-B"
#     # for data_loss_scale in [1, 5, 10, 20, 40]:
#     for data_loss_scale in [1]:
#         for data_loss in ["arctan", "cauchy"]:
#             name = "TK-loss" + data_loss + "_scale" + str(data_loss_scale)
#             monitor_tk = monitor.Monitor(name)
#             monitor_tk.set_measures(measures_dic)
#             monitors.append(monitor_tk)
#             solver = tk.TikhonovLinearSolver(
#                 A=A_1D, A_adj=A_adj_1D,
#                 B=D_1D, B_adj=D_adj_1D,
#                 b=b,
#                 alpha=alpha,
#                 x0=x0,
#                 minimizer=minimizer,
#                 data_loss=data_loss,
#                 data_loss_scale=data_loss_scale,
#                 iter_max=iter_max,
#                 verbose=verbose,
#             )
#             solver.set_monitor(monitor_tk)
#             solver.run()
#             recon_nda = solver.get_x().reshape(*X_shape)
#             data_nda.append(recon_nda)
#             data_labels.append(name)

# if solver_TK:
#     name = "TK_least_squares"
#     monitor_tk = monitor.Monitor(name)
#     monitor_tk.set_measures(measures_dic)
#     monitors.append(monitor_tk)
#     solver = tk.TikhonovLinearSolver(
#         A=A_1D, A_adj=A_adj_1D,
#         B=D_1D, B_adj=D_adj_1D,
#         b=b,
#         alpha=alpha,
#         x0=x0,
#         iter_max=iter_max,
#         verbose=verbose,
#         # data_loss="huber",
#         minimizer="least_squares",
#     )
#     solver.set_monitor(monitor_tk)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_TK:
#     name = "TK_lsq_linear"
#     monitor_tk = monitor.Monitor(name)
#     monitor_tk.set_measures(measures_dic)
#     monitors.append(monitor_tk)
#     solver = tk.TikhonovLinearSolver(
#         A=A_1D, A_adj=A_adj_1D,
#         B=D_1D, B_adj=D_adj_1D,
#         b=b,
#         alpha=alpha,
#         x0=x0,
#         iter_max=iter_max,
#         verbose=verbose,
#         minimizer="lsq_linear",
#     )
#     solver.set_monitor(monitor_tk)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_TK:
#     data_loss = "linear"
#     name = "TK-loss" + data_loss
#     monitor_tk = monitor.Monitor(name)
#     monitor_tk.set_measures(measures_dic)
#     monitors.append(monitor_tk)
#     solver = tk.TikhonovLinearSolver(
#         A=A_1D, A_adj=A_adj_1D,
#         B=D_1D, B_adj=D_adj_1D,
#         b=b,
#         alpha=alpha,
#         x0=x0,
#         minimizer="CG",
#         data_loss=data_loss,
#         iter_max=iter_max,
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_tk)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_TK:
#     data_loss = "arctan"
#     minimizer = "CG"
#     name = "TK-loss" + data_loss
#     monitor_tk = monitor.Monitor(name)
#     monitor_tk.set_measures(measures_dic)
#     monitors.append(monitor_tk)
#     solver = tk.TikhonovLinearSolver(
#         A=A_1D, A_adj=A_adj_1D,
#         B=D_1D, B_adj=D_adj_1D,
#         b=b,
#         alpha=alpha,
#         x0=x0,
#         minimizer=minimizer,
#         data_loss=data_loss,
#         iter_max=iter_max,
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_tk)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# ------------------------------------ADMM------------------------------------
if solver_ADMM:
    name = "ADMM-TV"
    monitor_admm = monitor.Monitor(name)
    monitor_admm.set_measures(measures_dic)
    monitors.append(monitor_admm)
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
    solver.set_monitor(monitor_admm)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

# if solver_ADMM:
#     data_loss = "soft_l1"
#     name = "ADMM-TV-" + data_loss
#     monitor_admm = monitor.Monitor(name)
#     monitor_admm.set_measures(measures_dic)
#     monitors.append(monitor_admm)
#     solver = admm.ADMMLinearSolver(
#         A=A_1D, A_adj=A_adj_1D,
#         b=b,
#         B=D_1D, B_adj=D_adj_1D,
#         x0=x0,
#         alpha=alpha,
#         rho=rho,
#         iterations=ADMM_iterations,
#         dimension=dimension,
#         iter_max=iter_max,
#         data_loss=data_loss,
#         minimizer="TNC",
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_admm)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)


# ---------------------------------Primal Dual---------------------------------
if solver_PrimalDual:
    name = "PrimalDual-TV-ALG2"
    monitor_pd = monitor.Monitor(name)
    monitor_pd.set_measures(measures_dic)
    monitors.append(monitor_pd)
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
    solver.set_monitor(monitor_pd)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

# if solver_PrimalDual:
#     name = "PrimalDual-TV-ALG2_AHMOD"
#     monitor_pd = monitor.Monitor(name)
#     monitor_pd.set_measures(measures_dic)
#     monitors.append(monitor_pd)
#     solver = pd.PrimalDualSolver(
#         prox_f=prox_f,
#         prox_g_conj=prox.prox_tv_conj,
#         B=D_1D,
#         B_conj=D_adj_1D,
#         L2=8,
#         x0=x0,
#         alpha=alpha,
#         iterations=PD_iterations,
#         alg_type="ALG2_AHMOD",
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_pd)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_PrimalDual:
#     name = "PrimalDual-TV-ALG3"
#     monitor_pd = monitor.Monitor(name)
#     monitor_pd.set_measures(measures_dic)
#     monitors.append(monitor_pd)
#     solver = pd.PrimalDualSolver(
#         prox_f=prox_f,
#         prox_g_conj=prox.prox_tv_conj,
#         B=D_1D,
#         B_conj=D_adj_1D,
#         L2=8,
#         x0=x0,
#         alpha=alpha,
#         iterations=PD_iterations,
#         alg_type="ALG3",
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_pd)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_PrimalDual:
#     name = "PrimalDual-Huber-ALG2"
#     monitor_pd = monitor.Monitor(name)
#     monitor_pd.set_measures(measures_dic)
#     monitors.append(monitor_pd)
#     solver = pd.PrimalDualSolver(
#         prox_f=prox_f,
#         prox_g_conj=prox.prox_huber_conj,
#         B=D_1D,
#         B_conj=D_adj_1D,
#         L2=8,
#         x0=x0,
#         alpha=alpha,
#         iterations=PD_iterations,
#         alg_type="ALG2",
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_pd)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_PrimalDual:
#     name = "PrimalDual-Huber-ALG2_AHMOD"
#     monitor_pd = monitor.Monitor(name)
#     monitor_pd.set_measures(measures_dic)
#     monitors.append(monitor_pd)
#     solver = pd.PrimalDualSolver(
#         prox_f=prox_f,
#         prox_g_conj=prox.prox_huber_conj,
#         B=D_1D,
#         B_conj=D_adj_1D,
#         L2=8,
#         x0=x0,
#         alpha=alpha,
#         iterations=PD_iterations,
#         alg_type="ALG2_AHMOD",
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_pd)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)

# if solver_PrimalDual:
#     name = "PrimalDual-Huber-ALG3"
#     monitor_pd = monitor.Monitor(name)
#     monitor_pd.set_measures(measures_dic)
#     monitors.append(monitor_pd)
#     solver = pd.PrimalDualSolver(
#         prox_f=prox_f,
#         prox_g_conj=prox.prox_huber_conj,
#         B=D_1D,
#         B_conj=D_adj_1D,
#         L2=8,
#         x0=x0,
#         alpha=alpha,
#         iterations=PD_iterations,
#         alg_type="ALG3",
#         verbose=verbose,
#     )
#     solver.set_monitor(monitor_pd)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

#     if dimension == 3:
#         recon_sitk = sitk.GetImageFromArray(recon_nda)
#         recon_sitk.CopyInformation(original_sitk)
#         data_sitk.append(recon_sitk)


# ------------------------------Visualize Results------------------------------
linestyles = ["-", ":", "-", "-."] * 10
if dimension == 1:
    ph.show_curves(data_nda, labels=data_labels, fig_number=1,
                   linestyle=linestyles, markers=ph.MARKERS)

elif dimension == 2:
    ph.show_arrays(data_nda, title=data_labels, fig_number=1)

elif dimension == 3:
    sitkh.show_sitk_image(data_sitk, label=data_labels)

for m in range(0, len(monitors)):
    monitors[m].compute_measures()
    print("Computational time %s: %s" %
          (monitors[m].get_name(), monitors[m].get_computational_time()))

for k in measures_dic.keys():
    title = k
    x = []
    y = []
    legend = []
    for m in range(0, len(monitors)):
        y.append(monitors[m].get_measures()[title])
        legend.append(monitors[m].get_name())
        x.append(range(0, len(y[-1])))
    ph.show_curves(y, x=x, xlabel="iteration", labels=legend, title=title,
                   linestyle=linestyles, markers=ph.MARKERS, markevery=1)
