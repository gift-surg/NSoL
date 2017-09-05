#!/usr/bin/python

##
# \file compareSolver.py
# \brief      Playground file to test solvers for Yijing's application
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import sys
import scipy.io
import numpy as np
import SimpleITK as sitk

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

import numericalsolver.LinearOperators as LinearOperators
import numericalsolver.Noise as Noise
import numericalsolver.TikhonovLinearSolver as tk
import numericalsolver.ADMMLinearSolver as admm
import numericalsolver.PrimalDualSolver as pd
import numericalsolver.Observer as observer
from numericalsolver.ProximalOperators import ProximalOperators as prox
from numericalsolver.SimilarityMeasures import SimilarityMeasures as sim_meas
from numericalsolver.PriorMeasures import PriorMeasures as prior_meas

from numericalsolver.definitions import DIR_TEST


np.random.seed(seed=1)

dimension = 2

solver_TK = 0
solver_ADMM = 0
solver_PrimalDual = 1

verbose = 0

iter_max = 10
alpha = 1/0.7
rho = 0.1
ADMM_iterations = 50
PD_iterations = 500

filename = "/Users/mebner/Dropbox/Yijing Data set/040117-HumanJM02Mes.mat"
dic = scipy.io.loadmat(filename)
observed_nda = np.array(dic["Mescpv"])

filename_de = "/Users/mebner/Dropbox/Yijing Data set/040117-HumanJM02DeMes.mat"
dic = scipy.io.loadmat(filename_de)
observed_de_nda = np.array(dic["DeMescpv"])

linear_operators = eval("LinearOperators.LinearOperators%dD()" % (dimension))

grad, grad_adj = linear_operators.get_gradient_operators()

# A: X \rightarrow Y and D: X \rightarrow Z
X_shape = observed_nda.shape
# Y_shape = A(observed_nda).shape
Z_shape = grad(observed_nda).shape

b = observed_nda.flatten()
x0 = observed_nda.flatten()
x_scale = np.max(observed_nda)

I_1D = lambda x: x.flatten()
I_adj_1D = lambda x: x.flatten()

# A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
# A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

D_1D = lambda x: grad(x.reshape(*X_shape)).flatten()
D_adj_1D = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

# if flag_blurring:
#     prox_f = lambda x, tau: prox.prox_linear_least_squares(
#         x=x, tau=tau,
#         A=A_1D, A_adj=A_adj_1D, b=b, x0=x0, x_scale=x_scale)
# else:
prox_f_ell1 = lambda x, tau: prox.prox_ell1_denoising(
    x, tau, x0=b, x_scale=x_scale)
prox_f_ell2 = lambda x, tau: prox.prox_ell2_denoising(
    x, tau, x0=b, x_scale=x_scale)


A_1D = I_1D
A_adj_1D = I_adj_1D

data_nda = []
data_labels = []

data_nda.append(observed_nda)
data_labels.append("observed")
data_nda.append(observed_de_nda)
data_labels.append("TVL1_Yijing")

# ph.show_arrays(data_nda, title=data_labels, fig_number=None,
#                cmap="jet", use_same_scaling=True)

x_ref = observed_de_nda.flatten()

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

observers = []
measures_dic = {
    # "SSD": ssd,
    "RMSE": rmse,
    "PSNR": psnr,
    "SSIM": ssim,
    "NCC": ncc,
    # "MI": mi,
    # "NMI": nmi,
    # "TK1": tk1,
    # "TV": tv,
}

# ----------------------------------Tikhonov----------------------------------
if solver_TK:
    name = "TK"
    minimizer = "L-BFGS-B"
    observer_tk = observer.Observer(name)
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

# if solver_TK:
#     minimizer = "L-BFGS-B"
#     # for data_loss_scale in [1, 5, 10, 20, 40]:
#     for data_loss_scale in [1]:
#         for data_loss in ["arctan", "cauchy"]:
#             name = "TK-loss" + data_loss + "_scale" + str(data_loss_scale)
#             observer_tk = observer.Observer(name)
#             observer_tk.set_measures(measures_dic)
#             observers.append(observer_tk)
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
#             solver.set_observer(observer_tk)
#             solver.run()
#             recon_nda = solver.get_x().reshape(*X_shape)
#             data_nda.append(recon_nda)
#             data_labels.append(name)

# ------------------------------------ADMM------------------------------------
if solver_ADMM:
    name = "ADMM-TV"
    observer_admm = observer.Observer(name)
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


# if solver_ADMM:
#     data_loss = "soft_l1"
#     name = "ADMM-TV-" + data_loss
#     observer_admm = observer.Observer(name)
#     observer_admm.set_measures(measures_dic)
#     observers.append(observer_admm)
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
#     solver.set_observer(observer_admm)
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
    alg_type = "ALG2"
    name = "PrimalDual-TV-L1_" + alg_type
    observer_pd = observer.Observer(name)
    observer_pd.set_measures(measures_dic)
    observers.append(observer_pd)
    solver = pd.PrimalDualSolver(
        prox_f=prox_f_ell1,
        prox_g_conj=prox.prox_tv_conj,
        # prox_g_conj=prox.prox_huber_conj,
        B=D_1D,
        B_conj=D_adj_1D,
        L2=8,
        x0=x0,
        alpha=alpha,
        iterations=PD_iterations,
        alg_type=alg_type,
        verbose=verbose,
        x_scale=x_scale,
    )
    solver.set_observer(observer_pd)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

# if solver_PrimalDual:
#     alg_type = "ALG2"
#     name = "PrimalDual-Huber-L1_" + alg_type
#     observer_pd = observer.Observer(name)
#     observer_pd.set_measures(measures_dic)
#     observers.append(observer_pd)
#     solver = pd.PrimalDualSolver(
#         prox_f=prox_f_ell1,
#         prox_g_conj=prox.prox_huber_conj,
#         B=D_1D,
#         B_conj=D_adj_1D,
#         L2=8,
#         x0=x0,
#         alpha=alpha,
#         iterations=PD_iterations,
#         alg_type=alg_type,
#         verbose=verbose,
#         x_scale=x_scale,
#     )
#     solver.set_observer(observer_pd)
#     solver.run()
#     recon_nda = solver.get_x().reshape(*X_shape)
#     data_nda.append(recon_nda)
#     data_labels.append(name)

# ------------------------------Visualize Results------------------------------
ph.show_arrays(data_nda, title=data_labels, fig_number=None,
               cmap="jet", use_same_scaling=True)

linestyles = ["-", ":", "-", "-."] * 10
for m in range(0, len(observers)):
    observers[m].compute_measures()
    print("Computational time %s: %s" %
          (observers[m].get_name(), observers[m].get_computational_time()))

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
