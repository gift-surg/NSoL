#!/usr/bin/python

##
# \file perform_parameter_study.py
# \brief      Playground file
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import sys
import re
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import nsol.linear_operators as LinearOperators
import nsol.noise as Noise
import nsol.tikhonov_linear_solver as tk
import nsol.tikhonov_linear_solver_parameter_study as tkparam
import nsol.admm_linear_solver_parameter_study as admmparam
import nsol.primal_dual_solver_parameter_study as pdparam
import nsol.reader_parameter_study as readerps
import nsol.admm_linear_solver as admm
import nsol.primal_dual_solver as pd
import nsol.observer as observer
from nsol.proximal_operators import ProximalOperators as prox
from nsol.similarity_measures import SimilarityMeasures as sim_meas
from nsol.prior_measures import PriorMeasures as prior_meas

from nsol.definitions import DIR_TEST
from nsol.definitions import DIR_ROOT


np.random.seed(seed=1)

# dimension = 1
dimension = 2
# dimension = 3

solver_TK = 1
solver_ADMM = 0
solver_PrimalDual = 0

verbose = 0
flag_normalize = 1
max_intensity_normalization = 10
flag_blurring = 1
flag_noise = 1
noise_level = 0.05
directory = os.path.join(DIR_ROOT, "results", "ParameterStudy")
directory = os.path.join("/tmp/", "results", "ParameterStudy")

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


# -----------------------------Similarity Measures-----------------------------
x_ref = original_nda.flatten()
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
    "SSD": ssd,
    "RMSE": rmse,
    "PSNR": psnr,
    "SSIM": ssim,
    "NCC": ncc,
    # "MI": mi,
    "NMI": nmi,
    "TK1": tk1,
    "TV": tv,
}

observer_tk = observer.Observer()
observer_tk.set_measures(measures_dic)

solver_tk = tk.TikhonovLinearSolver(
    A=A_1D, A_adj=A_adj_1D,
    B=D_1D, B_adj=D_adj_1D,
    b=b,
    alpha=alpha,
    x0=x0,
    iter_max=iter_max,
    verbose=verbose,
    x_scale=x_scale,
    minimizer="L-BFGS-B",
)

observer_admm = observer.Observer()
observer_admm.set_measures(measures_dic)

solver_admm = admm.ADMMLinearSolver(
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

observer_pd = observer.Observer()
observer_pd.set_measures(measures_dic)

solver_pd = pd.PrimalDualSolver(
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

# ---------------------------------Run Studies---------------------------------
if solver_TK:
    name_tk = "Tikhonov"
    tikhonov_parameter_study = tkparam.TikhonovLinearSolverParameterStudy(
        solver_tk, observer_tk,
        dir_output=directory,
        name=name_tk,
    )
    tikhonov_parameter_study.run()

if solver_ADMM:
    name_admm = "ADMM"
    admm_parameter_study = admmparam.ADMMLinearSolverParameterStudy(
        solver_admm, observer_admm,
        dir_output=directory,
        name=name_admm,
    )
    # admm_parameter_study.run()

if solver_PrimalDual:
    name_pd = "PrimalDual"
    pd_parameter_study = pdparam.PrimalDualSolverParameterStudy(
        solver_pd, observer_pd,
        dir_output=directory,
        name=name_pd,
    )
    # pd_parameter_study.run()
ph.exit()
# --------------------------------Read Studies--------------------------------
if solver_TK:
    parameter_study_reader = readerps.ReaderParameterStudy(
        directory=directory, name=name_tk)
    parameter_study_reader.read_study()

    nda_SSD = parameter_study_reader.get_results("SSD")
    nda_TK1 = parameter_study_reader.get_results("TK1")

    parameters_dic = parameter_study_reader.get_parameters()
    parameters_to_line_dic = parameter_study_reader.get_parameters_to_line()
    line_to_parameter_labels_dic = parameter_study_reader.\
        get_line_to_parameter_labels()

    print parameters_dic
    print parameters_to_line_dic
    print line_to_parameter_labels_dic

    # Get lines in result files associated to 'alpha'
    p = {k: (parameters_dic[k] if k == 'alpha' else parameters_dic[
             k][0]) for k in parameters_dic.keys()}
    lines = parameter_study_reader.get_lines_to_parameters(p)
    print lines

    x = [nda_SSD[i, -1] for i in range(nda_SSD.shape[0])]
    y = [nda_TK1[i, -1] for i in range(nda_TK1.shape[0])]
    label_sub = {
        "alpha": "$a$",
        "data_loss": "r",
        "data_loss_scale": "f",
    }
    label = []
    for i in range(nda_SSD.shape[0]):
        ell = line_to_parameter_labels_dic[i]
        for k in label_sub.keys():
            ell = re.sub(k, label_sub[k], ell)
        label.append(ell)
    # label = [label_sub[] for i in range(nda_SSD.shape[0])]

    ph.show_curves(y, x=x,
                   xlabel="SSD",
                   ylabel="TK1",
                   labels=label,
                   markers=ph.MARKERS*100,
                   markevery=1,
                   # y_axis_style="loglog",
                   )
if solver_ADMM:
    parameter_study_reader = readerps.ReaderParameterStudy(
        directory=directory, name=name_admm)
    parameter_study_reader.read_study()

    nda_SSD = parameter_study_reader.get_results("SSD")
    nda_TV = parameter_study_reader.get_results("TV")
    nda_NCC = parameter_study_reader.get_results("NCC")

    parameters_dic = parameter_study_reader.get_parameters()
    line_to_parameter_labels_dic = parameter_study_reader.\
        get_line_to_parameter_labels()

    x = [nda_SSD[i, -1] for i in range(nda_SSD.shape[0])]
    y = [nda_TV[i, -1] for i in range(nda_TV.shape[0])]
    label = []
    for i in range(nda_SSD.shape[0]):
        ell = line_to_parameter_labels_dic[i]
        label.append(ell)
    # label = [label_sub[] for i in range(nda_SSD.shape[0])]

    ph.show_curves(y, x=x,
                   xlabel="SSD",
                   ylabel="TV",
                   labels=label,
                   markers=ph.MARKERS*100,
                   markevery=1,
                   # y_axis_style="loglog",
                   )

    i = 0
    ph.show_curves(nda_NCC[i, :],
                   xlabel="iterations",
                   ylabel="NCC",
                   # labels=label,
                   labels=line_to_parameter_labels_dic[i],
                   markers=ph.MARKERS*100,
                   markevery=1,
                   # y_axis_style="loglog",
                   )

if solver_PrimalDual:
    parameter_study_reader = readerps.ReaderParameterStudy(
        directory=directory, name=name_pd)
    parameter_study_reader.read_study()

    nda_SSD = parameter_study_reader.get_results("SSD")
    nda_TV = parameter_study_reader.get_results("TV")
    nda_NCC = parameter_study_reader.get_results("NCC")

    parameters_dic = parameter_study_reader.get_parameters()
    line_to_parameter_labels_dic = parameter_study_reader.\
        get_line_to_parameter_labels()

    x = [nda_SSD[i, -1] for i in range(nda_SSD.shape[0])]
    y = [nda_TV[i, -1] for i in range(nda_TV.shape[0])]
    label = []
    for i in range(nda_SSD.shape[0]):
        ell = line_to_parameter_labels_dic[i]
        label.append(ell)
    # label = [label_sub[] for i in range(nda_SSD.shape[0])]

    ph.show_curves(y, x=x,
                   xlabel="SSD",
                   ylabel="TV",
                   labels=label,
                   markers=ph.MARKERS*100,
                   markevery=1,
                   # y_axis_style="loglog",
                   )

    i = 0
    ph.show_curves(nda_NCC[i, :],
                   xlabel="iterations",
                   ylabel="NCC",
                   # labels=label,
                   labels=line_to_parameter_labels_dic[i],
                   markers=ph.MARKERS*100,
                   markevery=1,
                   # y_axis_style="loglog",
                   )